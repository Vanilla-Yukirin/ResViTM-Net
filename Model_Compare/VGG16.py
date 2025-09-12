'''
本代码将测试Vgg16模型。输入是1024*1024的图片，输出是0或1，0表示正常，1表示异常。
使用read_data_meta.py中的main函数读取数据集，并使用Vgg16模型进行训练。
模型保存在model_output文件夹下
文件名形如
Vgg16-20250409010540-{loss:.4f}.pth

使用元数据
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
sys.path.append(os.path.join(current_dir, '..'))





# Vgg16

import torchvision.models as models
from torchvision.models import VGG16_Weights
from torchvision.transforms import Normalize

class Vgg16(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 使用预训练的 Vgg16
        vgg_model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        
         
        # 2. 将模型的 'features' 部分单独拿出来
        self.features = vgg_model.features
        
        # 3. 添加自适应平均池化层，将任何尺寸的特征图都转换为 7x7
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # 4. 获取原始的分类器，并修改最后一层
        # VGG16的分类器是一个nn.Sequential
        original_classifier = vgg_model.classifier
        
        # 获取原始分类器倒数第二层的输出维度
        num_features_before_last = original_classifier[-1].in_features # 通常是 4096
        
        # 替换最后一层为你需要的输出
        original_classifier[-1] = nn.Linear(num_features_before_last, num_classes)
        
        self.classifier = original_classifier
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # 展平特征图
        x = self.classifier(x)
        x = x.squeeze(1)
        return torch.sigmoid(x)


def save_model(model, loss, timestamp, epoch, model_path='model_output', end=""):
    """保存模型"""
    os.makedirs(model_path, exist_ok=True)
    loss_str = f"{loss:.4f}"
    model_name = f'Vgg16-{timestamp}-{epoch}-{loss_str}{end}.pth'
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
    plt.savefig(f'model_output/Vgg16-{timestamp}.png')
    # plt.show()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, device='cuda'):
        super().__init__()
        # 计算正负样本比例
        neg_count = 3126  # 根据您之前提供的数据
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




def train_cnn_vit_meta(data_lists, num_epochs, batch_size, learning_rate, resume_training, model_path):
    report=""
    """训练Vgg16模型"""
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
    # model = CNN_ViT_meta(
    #     img_size=1024,
    #     patch_size=16, # best is 16
    #     in_channels=1,  # 灰度图像
    #     num_classes=2,  # 二分类
    #     embed_dim=64,  # 减小嵌入维度以适应内存
    #     depth=2,        # 减少Transformer块数量
    #     num_heads=4,    # 减少注意力头数
    #     mlp_ratio=2.0,
    #     dropout=0.2
    # )

    model = Vgg16(num_classes=1)

    # 如果继续训练，加载已保存的模型并解析文件名获取epoch信息
    if resume_training and model_path:
        print(f"加载模型: {model_path}")
        model.load_state_dict(torch.load(model_path))
        
        # 从文件名中提取epoch信息
        # 格式: Resnet-timestamp-epoch-loss.pth
        try:
            filename = os.path.basename(model_path)
            parts = filename.split('-')
            # epoch是第3个部分(索引2)
            start_epoch = int(parts[2])
            print(f"从文件名解析的epoch: {start_epoch}")
            # 使用原始模型的时间戳
            timestamp = parts[1]
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
    
    # 定义损失函数和优化器

    # CrossEntropyLoss
    # criterion = nn.CrossEntropyLoss()

    # 带权重的CrossEntropyLoss

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

    criterion = nn.CrossEntropyLoss(weight=weights)

    # FocalLoss
    # criterion = FocalLoss(gamma=2, device=device)

    # SmoothL1Loss
    beta_value = 0.6
    print(f"使用 SmoothL1Loss, beta (分段值) = {beta_value}")
    report+=f"使用 SmoothL1Loss, beta (分段值) = {beta_value}\n"
    criterion = nn.SmoothL1Loss(beta=beta_value)

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
        

        # Resnet的输入的归一化转换
        # ... 在循环外部定义好归一化转换
        # ImageNet的均值和标准差
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        normalizer = Normalize(mean=imagenet_mean, std=imagenet_std)

        # 使用tqdm显示进度
        train_pbar = tqdm(range(0, len(augmented_train_data), batch_size), desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for i in train_pbar:
            batch_data = augmented_train_data[i:i + batch_size]
            
            # 准备批次数据
            # X_batch = torch.stack([torch.from_numpy(d["img"]).unsqueeze(0).float() / 255.0 for d in batch_data]).to(device)

            # 黑白变成三通道
            # 1. 准备原始批次数据
            X_batch_raw = torch.stack([torch.from_numpy(d["img"]).unsqueeze(0).float() / 255.0 for d in batch_data]).to(device)
            # 2. 将单通道复制为三通道
            X_batch_rgb = X_batch_raw.repeat(1, 3, 1, 1)  # Shape: [B, 3, H, W]
            # 3. 应用ImageNet归一化
            X_batch = normalizer(X_batch_rgb)

            meta_batch = torch.zeros(len(batch_data), 5)
            for j, d in enumerate(batch_data):
                meta_batch[j, 0] = d.get("gender0", 0)
                meta_batch[j, 1] = d.get("gender1", 0)
                meta_batch[j, 2] = d.get("age0", 0)
                meta_batch[j, 3] = d.get("age1", 0)
                meta_batch[j, 4] = d.get("age2", 0)
            meta_batch = meta_batch.to(device)
            Y_batch = torch.tensor([d["positive"] for d in batch_data]).to(device)
            
            # 前向传播
            optimizer.zero_grad()
            # outputs = model(X_batch, meta_batch)
            outputs = model(X_batch)  # 不传入 meta_batch
            loss = criterion(outputs, Y_batch)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * len(batch_data)
            # _, predicted = torch.max(outputs, 1) # for focal loss
            # predicted = (outputs > 0).float() # for bce loss
            predicted = (outputs > 0.5).float() # for mse loss
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

        
        # 使用tqdm显示进度
        val_pbar = tqdm(range(0, len(val_data), batch_size), desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for i in val_pbar:
                batch_data = val_data[i:i + batch_size]
                
                # 准备批次数据
                # X_batch = torch.stack([torch.from_numpy(d["img"]).unsqueeze(0).float() / 255.0 for d in batch_data]).to(device)
                
                # 黑白变成三通道
                # 1. 准备原始批次数据
                X_batch_raw = torch.stack([torch.from_numpy(d["img"]).unsqueeze(0).float() / 255.0 for d in batch_data]).to(device)
                # 2. 将单通道复制为三通道
                X_batch_rgb = X_batch_raw.repeat(1, 3, 1, 1)  # Shape: [B, 3, H, W]
                # 3. 应用ImageNet归一化
                X_batch = normalizer(X_batch_rgb)
                
                meta_batch = torch.zeros(len(batch_data), 5)
                for j, d in enumerate(batch_data):
                    meta_batch[j, 0] = d.get("gender0", 0)
                    meta_batch[j, 1] = d.get("gender1", 0)
                    meta_batch[j, 2] = d.get("age0", 0)
                    meta_batch[j, 3] = d.get("age1", 0)
                    meta_batch[j, 4] = d.get("age2", 0)
                meta_batch = meta_batch.to(device)
                Y_batch = torch.tensor([d["positive"] for d in batch_data]).to(device)
                
                # 前向传播
                # outputs = model(X_batch,meta_batch)
                outputs = model(X_batch)  # 不传入 meta_batch
                loss = criterion(outputs, Y_batch)
                
                # 统计
                val_loss += loss.item() * len(batch_data)
                # _, predicted = torch.max(outputs, 1) # for focal loss
                # predicted = (outputs > 0).float() # for bce loss
                predicted = (outputs > 0.5).float() # for mse loss
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
    report_path = os.path.join('report', f'Vgg11_report-{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    return best_model, best_val_loss


def select_model():
    """选择要使用的模型文件"""
    model_dir = 'model_output'
    if not os.path.exists(model_dir):
        print(f"错误: 模型目录 {model_dir} 不存在!")
        return None
    
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.pth') and f.startswith('Vgg16')])
    if not model_files:
        print(f"错误: 在 {model_dir} 中没有找到模型文件!")
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
    print("开始训练Vgg16模型")
    import utils.read_data_meta as read_data_meta
    data_lists = read_data_meta.main()

    

    
    # 训练模型
    best_model, best_val_loss = train_cnn_vit_meta(
        data_lists=data_lists,
        num_epochs=40,
        batch_size=2,
        learning_rate=1e-4,
        resume_training=resume_training,
        model_path=model_path
    )
    
    print("Vgg16模型训练完成！")


if __name__ == "__main__":
    main()
