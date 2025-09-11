'''
本代码将测试INV_CNN（CNN反演）模型。输入是1024*1024的图片，输出是0或1，0表示正常，1表示异常。
使用read_date_new.py中的read_data_new函数读取数据集，并使用INV_CNN模型进行训练。
模型保存在INV_CNN_output文件夹下
文件名形如
INV_CNN-20250409010540-{epoch}-{loss:.4f}.pth


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
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))


# 数据增强函数
def rotate_image(image, angle):
    """旋转图像，保持尺寸不变"""
    height, width = image.shape
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated


def adjust_brightness(img):
    """使用gamma变换调整亮度"""
    # 使用对数正态分布
    mu = 0  # 对数正态分布的均值参数
    sigma = 0.3  # 控制分布的扩散程度
    gamma = np.random.lognormal(mu, sigma)

    # 归一化到[0,1]
    img_normalized = img / 255.0
    img_gamma = np.power(img_normalized, gamma)
    img = (img_gamma * 255).astype(np.uint8)
    return img


def resize_img(img, target_size=(1024, 1024)):
    """调整图像大小到目标尺寸"""
    return cv2.resize(img, target_size)


def apply_augmentation(image):
    """对图像应用随机变换（针对分类任务的版本）"""
    # 随机旋转
    angle = random.randint(-30, 30)
    image = rotate_image(image, angle)
    
    # 随机裁剪后resize回1024x1024
    crop_size = random.randint(800, 1000)
    start_x = random.randint(0, 1024 - crop_size)
    start_y = random.randint(0, 1024 - crop_size)
    
    image = image[start_y:start_y+crop_size, start_x:start_x+crop_size]
    image = resize_img(image)
    
    # gamma变换调整亮度
    image = adjust_brightness(image)

    return image


class INV_CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        """
        针对1024x1024输入图像的INV_CNN模型
        Args:
            in_channels: 输入通道数（灰度图为1）
            num_classes: 分类数
        """
        super().__init__()
        
        # INV_CNN特征提取部分
        self.features = nn.Sequential(
            # 第一层卷积块
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),  # 输出: 512x512x16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 256x256x16
            
            # 第二层卷积块
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 输出: 256x256x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 128x128x32
            
            # 第三层卷积块
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 输出: 128x128x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 64x64x64
            
            # 第四层卷积块
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 输出: 64x64x128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 32x32x128
            
            # 第五层卷积块
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 输出: 32x32x256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 16x16x256
        )
        
        # 计算特征图展平后的大小: 16 * 16 * 256 = 65536
        self.feature_size = 16 * 16 * 256
        

        # 共享的特征层
        self.shared_fc=nn.Sequential(
            nn.Linear(self.feature_size, 1024),  # 输入维度必须是65536
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        # 性别分类
        self.gender_classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # 输出维度是2，表示性别分类
        )

        # 年龄分类
        self.age_classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # 三分类: 0-年轻, 1-中年, 2-老年
        )


        # 初始化权重
        self._initialize_weights()
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平特征图，保留批次维度
        x = self.shared_fc(x)
        
        gender_out = self.gender_classifier(x)
        age_out = self.age_classifier(x)
        
        return gender_out, age_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def save_model(model, loss, timestamp, epoch, model_path='INV_CNN_output', end=""):
    """保存模型"""
    os.makedirs(model_path, exist_ok=True)
    loss_str = f"{loss:.4f}"
    model_name = f'INV_CNN-{timestamp}-{epoch}-{loss_str}{end}.pth'
    save_path = os.path.join(model_path, model_name)
    
    torch.save(model.state_dict(), save_path)
    print(f'模型已保存至: {save_path}')
    return save_path


def plot_history(train_losses, val_losses, train_accs, val_accs):
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
    os.makedirs('INV_CNN_output', exist_ok=True)
    plt.savefig('INV_CNN_output/INV_CNN-training_history.png')
    # plt.show()


def save_model_multitask(model, val_loss, timestamp, epoch, end=""):
    """保存多任务模型"""
    if not os.path.exists("INV_CNN_output"):
        os.makedirs("INV_CNN_output")
    model_path = os.path.join("INV_CNN_output", f"INV_CNN-{timestamp}-{epoch}-{val_loss:.4f}{end}.pth")
    torch.save(model.state_dict(), model_path)
    return model_path

def plot_history_multitask(train_losses, val_losses, train_accs, val_accs):
    """绘制多任务训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 绘制总损失
    axes[0, 0].plot(train_losses['total'], label='Train Loss')
    axes[0, 0].plot(val_losses['total'], label='Val Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 绘制任务损失
    axes[0, 1].plot(train_losses['gender'], label='Train Gender Loss')
    axes[0, 1].plot(val_losses['gender'], label='Val Gender Loss')
    axes[0, 1].plot(train_losses['age'], label='Train Age Loss')
    axes[0, 1].plot(val_losses['age'], label='Val Age Loss')
    axes[0, 1].set_title('Task Losses')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 绘制性别准确率
    axes[1, 0].plot(train_accs['gender'], label='Train Gender Acc')
    axes[1, 0].plot(val_accs['gender'], label='Val Gender Acc')
    axes[1, 0].set_title('Gender Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 绘制年龄准确率
    axes[1, 1].plot(train_accs['age'], label='Train Age Acc')
    axes[1, 1].plot(val_accs['age'], label='Val Age Acc')
    axes[1, 1].set_title('Age Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    os.makedirs('INV_CNN_output', exist_ok=True)
    plt.savefig(f"INV_CNN_output/INV_CNN_history_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
    plt.close()


def pad_to_square(image, target_size=1024):
    # 获取原始尺寸
    h, w = image.shape[:2]
    
    # 计算缩放比例
    scale = target_size / max(h, w)
    
    # 等比例缩放
    new_h, new_w = int(h * scale), int(w * scale)
    resized_img = cv2.resize(image, (new_w, new_h))
    
    # 创建目标画布
    padded_img = np.zeros((target_size, target_size), dtype=np.uint8)
    
    # 计算偏移量（居中放置）
    h_offset = (target_size - new_h) // 2
    w_offset = (target_size - new_w) // 2
    
    # 将缩放后的图像放入画布中心
    padded_img[h_offset:h_offset+new_h, w_offset:w_offset+new_w] = resized_img
    
    return padded_img


def train_INV_CNN(data_lists, num_epochs, batch_size, learning_rate, resume_training, model_path):
    """训练INV_CNN模型"""
    # 合并数据列表
    all_data = []
    for data_list in data_lists:
        all_data.extend(data_list)
    
    print(f"合并后的数据总量: {len(all_data)}")
    
    # 缩放尺寸，放在1024*1024的画布内部
    print("开始缩放尺寸")
    for data in tqdm(all_data):
        data['img']=pad_to_square(data['img'], target_size=1024)

    # 划分分类
    print("开始划分分类")
    ages_cnt=[0,0,0]
    genders_cnt=[0,0]
    age_split1=18
    age_split2=30
    for data in tqdm(all_data):
        data['gender']=int(data['gender']) # 0/1
        genders_cnt[data['gender']]+=1

        data['age0']=int(data['age']<=age_split1)
        data['age1']=int(data['age']>age_split1 and data['age']<=age_split2)
        data['age2']=int(data['age']>age_split2)
        if data['age']<=age_split1:
            ages_cnt[0]+=1
            data['age_group'] = 0  # 第一类
        elif data['age']>age_split1 and data['age']<=age_split2:
            ages_cnt[1]+=1
            data['age_group'] = 1
        else:
            ages_cnt[2]+=1
            data['age_group'] = 2
    print(f"性别分布为： 女:男={genders_cnt[0]}:{genders_cnt[1]}")
    print(f"三个年龄段分别为： 0-{age_split1}:{age_split1}~{age_split2}:>{age_split2}+={ages_cnt[0]}:{ages_cnt[1]}:{ages_cnt[2]}")


    # 划分训练集和验证集
    train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)
    print(f"训练集大小: {len(train_data)}, 验证集大小: {len(val_data)}")
    

    # 数据增强
    print("开始增强训练集")
    if not os.path.exists("INV_CNN_aug_debug"):
        os.makedirs("INV_CNN_aug_debug")

    augmented_train_data = []
    for data in tqdm(train_data):
        # 原始数据
        augmented_train_data.append(data)
        # 增强数据
        aug_num=1
        for _ in range(aug_num):  # 每个样本生成若干个增强版本
            img_aug = apply_augmentation(data["img"])
            aug_data = {
                "file_name": data["file_name"] + f"_aug_{_}",
                "img": img_aug,
                "positive": data["positive"],
                "gender": data["gender"],
                "age": data["age"],
                "age0": data["age0"],
                "age1": data["age1"],
                "age2": data["age2"],
                "age_group": data["age_group"],
                "description": data["description"]
            }
            # 保存部分增强图像用于调试
            if random.random() < 0.05:  # 只保存5%的增强图像
                Image.fromarray(img_aug.astype(np.uint8)).save(
                    os.path.join("INV_CNN_aug_debug", f"{os.path.splitext(data['file_name'])[0]}_aug_{_}.png")
                )
            augmented_train_data.append(aug_data)
    # 打乱顺序
    random.shuffle(augmented_train_data)
    print(f"增强后的训练集大小: {len(augmented_train_data)}")
    start_epoch = 0

    # 初始化模型
    model = INV_CNN(in_channels=1)  # 使用新的多任务模型

    # 如果继续训练，加载已保存的模型并解析文件名获取epoch信息
    if resume_training and model_path:
        print(f"加载模型: {model_path}")
        model.load_state_dict(torch.load(model_path))
        
        # 从文件名中提取epoch信息
        # 格式: INV_CNN-timestamp-epoch-loss.pth
        try:
            filename = os.path.basename(model_path)
            parts = filename.split('-')
            # 假设epoch是第3个部分(索引2)
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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    
    # 记录训练历史
    train_losses = {'total': [], 'gender': [], 'age': []}
    val_losses = {'total': [], 'gender': [], 'age': []}
    train_accs = {'gender': [], 'age': []}
    val_accs = {'gender': [], 'age': []}

    # 记录最佳模型
    best_val_loss = float('inf')
    best_model = None
    patience = 10
    patience_counter = 0
    
    # 训练循环
    for epoch in range(start_epoch, total_epochs):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # 训练阶段
        model.train()
        train_loss = 0
        train_gender_loss = 0
        train_age_loss = 0
        train_gender_correct = 0
        train_age_correct = 0
        train_total = 0
        
        # 使用tqdm显示进度
        train_pbar = tqdm(range(0, len(augmented_train_data), batch_size), desc=f'Epoch {epoch+1}/{total_epochs} [Train]')
        
        for i in train_pbar:
            batch_data = augmented_train_data[i:i + batch_size]
            
            # 准备批次数据
            X_batch = torch.stack([torch.from_numpy(d["img"]).unsqueeze(0).float() / 255.0 for d in batch_data]).to(device)
            gender_batch = torch.tensor([d["gender"] for d in batch_data]).to(device)
            age_batch = torch.tensor([d["age_group"] for d in batch_data]).to(device)
            
            # 前向传播
            optimizer.zero_grad()
            gender_outputs, age_outputs = model(X_batch)
            
            # 计算损失
            gender_loss = criterion(gender_outputs, gender_batch)
            age_loss = criterion(age_outputs, age_batch)
            # 总损失 = 性别损失 + 年龄损失（可以调整权重）
            loss = gender_loss + age_loss
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * len(batch_data)
            train_gender_loss += gender_loss.item() * len(batch_data)
            train_age_loss += age_loss.item() * len(batch_data)
            
            # 计算准确率
            _, gender_predicted = torch.max(gender_outputs, 1)
            _, age_predicted = torch.max(age_outputs, 1)
            train_total += gender_batch.size(0)
            train_gender_correct += (gender_predicted == gender_batch).sum().item()
            train_age_correct += (age_predicted == age_batch).sum().item()
            
            # 更新进度条
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}', 
                'g_acc': f'{train_gender_correct/train_total:.4f}',
                'a_acc': f'{train_age_correct/train_total:.4f}'
            })
        
        # 计算平均损失和准确率
        avg_train_loss = train_loss / len(augmented_train_data)
        avg_train_gender_loss = train_gender_loss / len(augmented_train_data)
        avg_train_age_loss = train_age_loss / len(augmented_train_data)
        train_gender_accuracy = train_gender_correct / train_total
        train_age_accuracy = train_age_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_gender_loss = 0
        val_age_loss = 0
        val_gender_correct = 0
        val_age_correct = 0
        val_total = 0
        
        # 使用tqdm显示进度
        val_pbar = tqdm(range(0, len(val_data), batch_size), desc=f'Epoch {epoch+1}/{total_epochs} [Val]')
        
        with torch.no_grad():
            for i in val_pbar:
                batch_data = val_data[i:i + batch_size]
                
                # 准备批次数据
                X_batch = torch.stack([torch.from_numpy(d["img"]).unsqueeze(0).float() / 255.0 for d in batch_data]).to(device)
                gender_batch = torch.tensor([d["gender"] for d in batch_data]).to(device)
                age_batch = torch.tensor([d["age_group"] for d in batch_data]).to(device)
                
                # 前向传播
                gender_outputs, age_outputs = model(X_batch)
                
                # 计算损失
                gender_loss = criterion(gender_outputs, gender_batch)
                age_loss = criterion(age_outputs, age_batch)
                loss = gender_loss + age_loss
                
                # 统计
                val_loss += loss.item() * len(batch_data)
                val_gender_loss += gender_loss.item() * len(batch_data)
                val_age_loss += age_loss.item() * len(batch_data)
                
                # 计算准确率
                _, gender_predicted = torch.max(gender_outputs, 1)
                _, age_predicted = torch.max(age_outputs, 1)
                val_total += gender_batch.size(0)
                val_gender_correct += (gender_predicted == gender_batch).sum().item()
                val_age_correct += (age_predicted == age_batch).sum().item()
        
        # 计算平均损失和准确率
        avg_val_loss = val_loss / len(val_data)
        avg_val_gender_loss = val_gender_loss / len(val_data)
        avg_val_age_loss = val_age_loss / len(val_data)
        val_gender_accuracy = val_gender_correct / val_total
        val_age_accuracy = val_age_correct / val_total
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        # 记录历史
        train_losses['total'].append(avg_train_loss)
        train_losses['gender'].append(avg_train_gender_loss)
        train_losses['age'].append(avg_train_age_loss)
        val_losses['total'].append(avg_val_loss)
        val_losses['gender'].append(avg_val_gender_loss)
        val_losses['age'].append(avg_val_age_loss)
        train_accs['gender'].append(train_gender_accuracy)
        train_accs['age'].append(train_age_accuracy)
        val_accs['gender'].append(val_gender_accuracy)
        val_accs['age'].append(val_age_accuracy)
        
        print(f'Epoch [{epoch+1}/{total_epochs}], '
              f'Train Loss: {avg_train_loss:.4f} (G:{avg_train_gender_loss:.4f}, A:{avg_train_age_loss:.4f}), '
              f'Train Acc: (G:{train_gender_accuracy:.4f}, A:{train_age_accuracy:.4f}), '
              f'Val Loss: {avg_val_loss:.4f} (G:{avg_val_gender_loss:.4f}, A:{avg_val_age_loss:.4f}), '
              f'Val Acc: (G:{val_gender_accuracy:.4f}, A:{val_age_accuracy:.4f})')

        # 保存当前模型
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        save_model(model, avg_val_loss, timestamp, epoch+1)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(model)
            patience_counter = 0
            best_epoch_id = epoch+1
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= patience:
            print(f'早停: {patience} 个epoch没有改善')
            break
    
    # 绘制训练历史
    plot_history_multitask(train_losses, val_losses, train_accs, val_accs)
    
    # 保存最佳模型
    if best_model is not None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        best_model_path = save_model_multitask(best_model, best_val_loss, timestamp, best_epoch_id, end="-best")
        print(f"最佳模型已保存: {best_model_path}, Val Loss: {best_val_loss:.4f}")
    
    return best_model, best_val_loss



def select_model():
    """选择要使用的模型文件"""
    model_dir = 'INV_CNN_output'
    if not os.path.exists(model_dir):
        print(f"错误: 模型目录 {model_dir} 不存在!")
        return None
    
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.pth') and f.startswith('INV_CNN-2')])
    if not model_files:
        print(f"错误: 在 {model_dir} 中没有找到模型文件!")
        return None
    
    print("找到以下模型文件:")
    model_files.sort()
    for i, model_file in enumerate(model_files):
        print(f"{i+1}. {model_file}")
    
    while True:
        try:
            choice = int(input("请选择要使用的模型 (输入序号，输入0创建新模型): "))
            if 1 <= choice <= len(model_files):
                return os.path.join(model_dir, model_files[choice-1])
            elif choice == 0:
                return None
            else:
                print(f"请输入0到{len(model_files)}之间的数字")
        except ValueError:
            print("请输入有效的数字")


def main():
    resume_training = True
    # 选择并加载模型
    model_path = select_model()
    if not model_path:
        resume_training = False
    print("开始训练INV_CNN模型")
    from utils import read_data
    data_lists = read_data.read_data(order="0110")

    
    # 训练模型
    best_model, best_val_loss = train_INV_CNN(
        data_lists=data_lists,
        num_epochs=50,
        batch_size=16,  # 增加批次大小，因为CNN模型比CNN-ViT小
        learning_rate=1e-3,  # 稍微增加学习率
        resume_training=resume_training,
        model_path=model_path
    )
    
    print("INV_CNN模型训练完成！")


if __name__ == "__main__":
    main()
