'''
本代码将测试ResViTM_BCELoss模型。输入是1024*1024的图片，输出是0或1，0表示正常，1表示异常。
使用read_data_meta.py中的main函数读取数据集，并使用ResViTM_BCELoss模型进行训练。
模型保存在model_output文件夹下
文件名形如
ResViTM_BCELoss-20250409010540-{loss:.4f}.pth
'''



import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import random
import copy
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加项目根目录到 sys.path
sys.path.append(os.path.join(current_dir, '../..'))



class PatchEmbedding(nn.Module):
    def __init__(self, img_size=1024, patch_size=16, in_channels=1, embed_dim=768):
        """
        图像分块嵌入
        Args:
            img_size: 输入图像大小
            patch_size: 分块大小
            in_channels: 输入通道数
            embed_dim: 嵌入维度
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # 使用卷积层实现分块嵌入
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2)  # [B, embed_dim, n_patches]
        x = x.transpose(1, 2)  # [B, n_patches, embed_dim]
        return x


import torchvision.models as models
class PartialPretrainedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 加载一个预训练的ResNet18模型
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # 2. (可选但推荐) 冻结所有预训练权重，我们暂时只把它当作一个固定的特征提取器
        # for param in resnet18.parameters():
        #     param.requires_grad = False

        # 3. 截取我们需要的层，从头开始直到 layer3 结束
        #    - resnet18.children() 会返回所有顶层模块
        #    - `*list(...)[:-3]` 表示我们取除了最后3个模块（avgpool, fc）之外的所有层
        #      对于ResNet18，剩下的就是 stem, layer1, layer2, layer3
        feature_extractor_layers = list(resnet18.children())[:-3] # 去掉layer4, avgpool, fc
        self.features = nn.Sequential(*feature_extractor_layers)
        
        # 4. ResNet18的layer3输出通道是256。我们用1x1卷积将其降维到1通道。
        #    这一层的权重是随机初始化的，需要我们自己训练。
        self.final_conv = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        # 注意：预训练模型需要3通道输入，所以要先复制通道
        x = x.repeat(1, 3, 1, 1) # [B, 1, H, W] -> [B, 3, H, W]
        
        out = self.features(x)
        out = self.final_conv(out)
        return out

class ResViTM_BCELoss(nn.Module):
    def __init__(self, img_size=1024, patch_size=16, in_channels=1, num_classes=2,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        """
        ResViTM_BCELoss模型，增加了元数据处理功能
        Args:
            img_size: 输入图像大小
            patch_size: 分块大小
            in_channels: 输入通道数（灰度图为1）
            num_classes: 分类数
            embed_dim: 嵌入维度
            depth: Transformer块数量
            num_heads: 注意力头数
            mlp_ratio: MLP隐藏层维度与嵌入维度的比例
            dropout: dropout比例
        """
        super().__init__()
        
        # 1. 实例化我们新的、基于ResNet18的CNN部分
        self.cnn = PartialPretrainedCNN()
        
        # 2. 计算下采样倍数
        # ResNet18到layer3为止，总共下采样了 2(stem_conv) * 2(stem_pool) * 2(layer2) * 2(layer3) = 16倍
        # 这和我们之前的MiniResNet设计保持了一致！
        self.img_size_after_cnn = img_size // 16
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            img_size=self.img_size_after_cnn,
            patch_size=patch_size // 16,
            in_channels=1,
            embed_dim=embed_dim
        )
        
        # 可学习的类别token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim)
        )
        
        self.pos_drop = nn.Dropout(dropout)
        
        # TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 层归一化
        self.norm = nn.LayerNorm(embed_dim)
        
        # 分类头
        # self.head = nn.Linear(embed_dim, num_classes)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, 1)
        )
        
        self.meta_net=nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embed_dim) # 输出维度与ViT特征维度相同，便于融合
        )

        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        # 初始化位置编码
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # 初始化线性层
        self.apply(self._init_weights_linear)
        
    def _init_weights_linear(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def forward(self, x, meta_data):
        """
        前向传播
        Args:
            x: 图像输入 [B, C, H, W]
            meta_data: 元数据输入 [B, 5] - [gender0, gender1, age0, age1, age2]
        """
        # CNN特征提取
        x = self.cnn(x)  # [B, in_channels, img_size/8, img_size/8]
        
        # Patch Embedding
        x = self.patch_embed(x)  # [B, n_patches, embed_dim]
        
        # 添加类别token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_token, x), dim=1)  # [B, n_patches+1, embed_dim]
        
        # 添加位置编码
        x = x + self.pos_embed  # [B, n_patches+1, embed_dim]
        x = self.pos_drop(x)
        
        # Transformer编码器
        x = self.transformer(x)
        
        # 层归一化
        x = self.norm(x)
        
        # 使用类别token的输出进行分类
        cls_features = x[:, 0]  # [B, embed_dim]
        
        # 融合特征
        meta_features = self.meta_net(meta_data)  # [B, embed_dim]
        cls_features = cls_features + meta_features
        
        # 分类
        x = self.head(cls_features).squeeze(1)  # [B, num_classes]
        
        return x


def save_model(model, loss, timestamp, epoch, model_path='model_output', end=""):
    """保存模型"""
    os.makedirs(model_path, exist_ok=True)
    loss_str = f"{loss:.4f}"
    model_name = f'ResViTM_BCELoss-{timestamp}-{epoch}-{loss_str}{end}.pth'
    save_path = os.path.join(model_path, model_name)
    
    torch.save(model.state_dict(), save_path)
    print(f'模型已保存至: {save_path}')
    return save_path


def plot_history(train_losses, val_losses, train_accs, val_accs, timestamp):
    """绘制训练历史"""
    plt.figure(figsize=(12, 5))
    
    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    os.makedirs('model_output', exist_ok=True)
    plt.savefig(f'model_output/ResViTM_BCELoss-{timestamp}.png')
    # plt.show()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, device='cuda'):
        super().__init__()
        # 计算正负样本比例
        neg_count = 3126
        pos_count = 873
        
        # 设置alpha权重
        pos_weight = min(neg_count/pos_count, 2.0)  # 限制最大权重为2
        self.alpha = torch.tensor([1.0, pos_weight]).to(device)
        
        # gamma值控制难样本的关注度
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha[targets] * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()




def train_ResViTM_BCELoss(data_lists, num_epochs, batch_size, learning_rate, resume_training, model_path):
    report=""
    """训练ResViTM_BCELoss模型"""
    # 合并数据列表
    all_data = []
    for data_list in data_lists:
        all_data.extend(data_list)
    
    print(f"合并后的数据总量: {len(all_data)}")
    
    # 划分训练集和验证集
    train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)
    print(f"训练集大小: {len(train_data)}, 验证集大小: {len(val_data)}")
    
    # 数据增强
    from utils.augmenter import ReproducibleAugmenter
    augmentation_seed = 42 

    # 1. 初始化增强器
    #    - 为训练集创建一个增强器实例
    #    - 你可以控制为每张图片衍生出多少张，这里以1张为例
    #    - 你也可以选择只对阳性样本增强
    train_augmenter = ReproducibleAugmenter(
        seed=augmentation_seed, 
        num_augmentations_per_image=2, # 每个样本额外生成2个
        use_rotation=True,
        use_cropping=True,
        use_brightness=True
    )
    
    # 2. 对训练集应用增强
    #    - is_positive_only=True 表示只对 "positive" == 1 的图片增强
    #    - 传入debug路径可以检查增强效果
    augmented_train_data = train_augmenter.augment(
        train_data, 
        is_positive_only=False,
        debug_save_path=None
    )

    # (可选) 对验证集不进行增强，或者使用不同的、更少的增强
    # 这里我们不对验证集进行增强，以保证评估的一致性
    # 如果需要，可以创建另一个augmenter实例
    # val_augmenter = ReproducibleAugmenter(seed=augmentation_seed, num_augmentations_per_image=0)
    # augmented_val_data = val_augmenter.augment(val_data)
    # print(f"验证集大小: {len(augmented_val_data)}") # 此时大小应与val_data相同
    
    print(f"增强后的训练集大小: {len(augmented_train_data)}")

    start_epoch = 0


    # 初始化模型
    model = ResViTM_BCELoss(
        img_size=1024,
        patch_size=16, # best is 16
        in_channels=1,  # 灰度图像
        num_classes=2,  # 二分类

        embed_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.2
    )

    # 如果继续训练，加载已保存的模型并解析文件名获取epoch信息
    if resume_training and model_path:
        print(f"加载模型: {model_path}")
        model.load_state_dict(torch.load(model_path))
        
        # 从文件名中提取epoch信息
        # 格式: ResViTM_BCELoss-timestamp-epoch-loss.pth
        try:
            filename = os.path.basename(model_path)
            parts = filename.split('-')
            # 假设epoch是第5个部分(索引4)
            start_epoch = int(parts[4])
            print(f"从文件名解析的epoch: {start_epoch}")
            # 使用原始模型的时间戳
            timestamp = parts[2]
        except Exception as e:
            print(f"无法从文件名解析epoch信息: {e}")
            print("将从epoch 0开始训练")
            start_epoch = 0
            
        print(f"模型加载成功，将从第 {start_epoch+1} 个epoch继续训练")
    
    # 调整总训练轮数，确保至少再训练num_epochs轮
    total_epochs = start_epoch + num_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    print(f"使用设备: {device}")
     # 计算训练集中正负样本数量
    neg_count = sum(1 for d in augmented_train_data if d["positive"] == 0)
    pos_count = sum(1 for d in augmented_train_data if d["positive"] == 1)
    # 计算权重
    neg_weight = 1.0
    pos_weight = neg_count / pos_count  # 负样本数量除以正样本数量
    # 设置损失函数
    weights = torch.tensor([neg_weight, pos_weight]).to(device)
    
    print(f"Negative samples: {neg_count}")
    print(f"Positive samples: {pos_count}")
    print(f"Weight ratio: {pos_weight:.2f}")
    # BCELossWithLogitsLoss
    pos_weight = torch.tensor([pos_weight], device=device)  # pos_weight已定义
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # 记录训练历史
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # 记录最佳模型
    best_val_loss = float('inf')
    best_model = None
    if resume_training:
        best_model = model
        best_epoch_id = start_epoch
        best_val_loss = 9999
    patience = 10
    patience_counter = 0
    
    # 训练循环
    for epoch in range(start_epoch, total_epochs):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # 显示进度
        train_pbar = tqdm(range(0, len(augmented_train_data), batch_size), desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for i in train_pbar:
            batch_data = augmented_train_data[i:i + batch_size]
            
            # 准备批次数据
            X_batch = torch.stack([torch.from_numpy(d["img"]).unsqueeze(0).float() / 255.0 for d in batch_data]).to(device)
            meta_batch = torch.zeros(len(batch_data), 5)
            for j, d in enumerate(batch_data):
                meta_batch[j, 0] = d.get("gender0", 0)
                meta_batch[j, 1] = d.get("gender1", 0)
                meta_batch[j, 2] = d.get("age0", 0)
                meta_batch[j, 3] = d.get("age1", 0)
                meta_batch[j, 4] = d.get("age2", 0)
            meta_batch = meta_batch.to(device)
            Y_batch = torch.tensor([d["positive"] for d in batch_data], dtype=torch.float32).to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(X_batch, meta_batch)
            loss = criterion(outputs, Y_batch)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * len(batch_data)
            predicted = (outputs > 0).float()
            train_total += Y_batch.size(0)
            train_correct += (predicted == Y_batch).sum().item()
            
            # 更新进度条
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{train_correct/train_total:.4f}'})
        
        avg_train_loss = train_loss / len(augmented_train_data)
        train_accuracy = train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        # 初始化混淆矩阵统计
        confusion_matrix = torch.zeros(2, 2)  # 2x2矩阵，因为是二分类
        tp = fp = tn = fn = 0


        # 为每个属性组合创建混淆矩阵字典
        age_labels = [0, 1, 2]
        gender_labels = [0, 1]
        group_metrics = {}
        # 初始化每个组合的统计数据
        for age in age_labels:
            for gender in gender_labels:
                group_key = f"age{age}_gender{gender}"
                group_metrics[group_key] = {
                    'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
                    'total': 0
                }

        
        # 显示进度
        val_pbar = tqdm(range(0, len(val_data), batch_size), desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for i in val_pbar:
                batch_data = val_data[i:i + batch_size]
                
                # 准备批次数据
                X_batch = torch.stack([torch.from_numpy(d["img"]).unsqueeze(0).float() / 255.0 for d in batch_data]).to(device)
                meta_batch = torch.zeros(len(batch_data), 5)
                for j, d in enumerate(batch_data):
                    meta_batch[j, 0] = d.get("gender0", 0)
                    meta_batch[j, 1] = d.get("gender1", 0)
                    meta_batch[j, 2] = d.get("age0", 0)
                    meta_batch[j, 3] = d.get("age1", 0)
                    meta_batch[j, 4] = d.get("age2", 0)
                meta_batch = meta_batch.to(device)
                Y_batch = torch.tensor([d["positive"] for d in batch_data], dtype=torch.float32).to(device)
                
                # 前向传播
                outputs = model(X_batch,meta_batch)
                loss = criterion(outputs, Y_batch)
                
                # 统计
                val_loss += loss.item() * len(batch_data)
                predicted = (outputs > 0).float()
                val_total += Y_batch.size(0)
                val_correct += (predicted == Y_batch).sum().item()

                # 计算混淆矩阵
                for pred, true in zip(predicted.cpu(), Y_batch.cpu()):
                    pred_label = int(pred.item())
                    true_label = int(true.item())
                    confusion_matrix[pred_label, true_label] += 1
                    if int(true.item()) == 1:
                        if int(pred_label) == 1:
                            tp += 1
                        else:
                            fn += 1
                    else:
                        if int(pred_label) == 1:
                            fp += 1
                        else:
                            tn += 1






                # 对每个样本进行分组统计
                for idx, (pred, true) in enumerate(zip(predicted.cpu(), Y_batch.cpu())):
                    # 获取当前样本的age和gender
                    current_age = None
                    if batch_data[idx]['age0'] == 1:
                        current_age = 0
                    elif batch_data[idx]['age1'] == 1:
                        current_age = 1
                    elif batch_data[idx]['age2'] == 1:
                        current_age = 2
                    current_gender = 1 if batch_data[idx]['gender1'] == 1 else 0
                    
                    # 构建组键
                    group_key = f"age{current_age}_gender{current_gender}"
                    
                    # 更新该组的统计数据
                    group_metrics[group_key]['total'] += 1
                    pred_label = int(pred.item())
                    true_label = int(true.item())
                    if true_label == 1:
                        if pred_label == 1:
                            group_metrics[group_key]['tp'] += 1
                        else:
                            group_metrics[group_key]['fn'] += 1
                    else:
                        if pred_label == 1:
                            group_metrics[group_key]['fp'] += 1
                        else:
                            group_metrics[group_key]['tn'] += 1

                
                # 更新进度条
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{train_correct/train_total:.4f}'})

        avg_val_loss = val_loss / len(val_data)
        val_accuracy = val_correct / val_total

        # 计算其他评估指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 打印详细的评估结果
        print("\nValidation Results:")
        print(f"Average Loss: {avg_val_loss:.4f}")
        print(f"Accuracy: {val_accuracy:.4f}")
        print("\nConfusion Matrix:")
        print(f"TP={tp},FP={fp},TN={tn},FN={fn}")
        print("\nDetailed Metrics:")
        print(f"Prec={precision:.4f},Rec={recall:.4f},Spec={specificity:.4f},F1={f1_score:.4f}")

        report += f"\nValidation Results:\nAverage Loss: {avg_val_loss:.4f}\nAccuracy: {val_accuracy:.4f}\nConfusion Matrix:\nTP={tp},FP={fp},TN={tn},FN={fn}\nDetailed Metrics:\nPrec={precision:.4f},Rec={recall:.4f},Spec={specificity:.4f},F1={f1_score:.4f}\n"



        # 打印每个组的统计结果
        print("\n=== Detailed Group Analysis ===")
        report+="\n=== Detailed Group Analysis ===\n"
        for group_key, metrics in group_metrics.items():
            tp = metrics['tp']
            fp = metrics['fp']
            tn = metrics['tn']
            fn = metrics['fn']
            total = metrics['total']
            
            # 计算指标
            accuracy = (tp + tn) / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\nGroup: {group_key} (Total samples: {total})")
            print("Confusion Matrix:")
            print(f"TP={tp},FP={fp},TN={tn},FN={fn},")
            print(f"Acc={accuracy:.4f},Prec={precision:.4f},Rec={recall:.4f},Spec={specificity:.4f},F1={f1_score:.4f}")

            report += f"\nGroup: {group_key} (Total samples: {total})\nConfusion Matrix:\nTP={tp},FP={fp},TN={tn},FN={fn},Acc={accuracy:.4f},Prec={precision:.4f},Rec={recall:.4f},Spec={specificity:.4f},F1={f1_score:.4f}\n"


        # 记录历史
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)


        
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        report+=f'Epoch [{epoch+1}/{num_epochs}], '
        report+=f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
        report+=f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}\n'



        # 保存当前模型
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        save_model(model, avg_val_loss, timestamp, epoch+1)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(model)
            patience_counter = 0
            best_epoch_id=epoch+1
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= patience:
            print(f'早停: {patience} 个epoch没有改善')
            report += f'早停: {patience} 个epoch没有改善\n'
            break
    
    # 绘制训练历史
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    plot_history(train_losses, val_losses, train_accs, val_accs, timestamp)
    
    # 保存最佳模型
    if best_model is not None:
        
        best_model_path = save_model(best_model,best_val_loss, timestamp, best_epoch_id, end="-best")
        print(f"最佳模型已保存: {best_model_path}, Val Loss: {best_val_loss:.4f}")
        report += f"最佳模型已保存: {best_model_path}, Val Loss: {best_val_loss:.4f}\n"
    
    
    # 保存报告
    report_path = os.path.join('report', f'ResViTM_BCELoss_report-{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    return best_model, best_val_loss


def select_model():
    """选择要使用的模型文件"""
    model_dir = 'model_output'
    if not os.path.exists(model_dir):
        print(f"错误: 模型目录 {model_dir} 不存在!")
        print("0. 从头开始训练")
        choice = int(input("请选择 (输入0从头训练): "))
        return None
    
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.pth') and f.startswith('ResViTM_BCELoss')])
    if not model_files:
        print(f"错误: 在 {model_dir} 中没有找到模型文件!")
        print("0. 从头开始训练")
        choice = int(input("请选择 (输入0从头训练): "))
        return None
    
    print("找到以下模型文件:")
    model_files.sort()
    for i, model_file in enumerate(model_files):
        print(f"{i+1}. {model_file}")
    
    while True:
        try:
            choice = int(input("请选择要使用的模型 (输入序号): "))
            if 1 <= choice <= len(model_files):
                return os.path.join(model_dir, model_files[choice-1])
            elif choice == 0:
                return None
            else:
                print(f"请输入1到{len(model_files)}之间的数字")
        except ValueError:
            print("请输入有效的数字")


def main():
    resume_training = True
    # 选择并加载模型
    model_path = select_model()
    if not model_path:
        resume_training = False
    print("开始训练ResViTM_BCELoss模型")
    import utils.read_data_meta as read_data_meta
    data_lists = read_data_meta.main()

    

    
    # 训练模型
    best_model, best_val_loss = train_ResViTM_BCELoss(
        data_lists=data_lists,
        num_epochs=40,
        batch_size=4,
        learning_rate=1e-4,
        resume_training=resume_training,
        model_path=model_path
    )
    
    print("ResViTM_BCELoss模型训练完成！")


if __name__ == "__main__":
    main()
