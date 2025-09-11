import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from datetime import datetime
from PIL import Image
import copy
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
from utils.read_data import read_data
from utils.read_mask import read_mask



SIZE=1024
# SIZE=2048*2048时估算需要23.8G显存
# SIZE=1024*1024实测占用8.3G显存

# UNet模型定义
class DoubleConv(nn.Module):
    """双重卷积模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """U-Net模型"""
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        # 编码器
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        # 解码器
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)  # 1024因为concatenation
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)  # 512因为concatenation
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)  # 256因为concatenation
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)  # 128因为concatenation
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # 编码路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 解码路径与跳跃连接
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up3(x)
        
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up4(x)
        
        x = self.outc(x)
        return x


import torch
import torch.nn.functional as F

def resize_img(img,h=SIZE,w=SIZE):
    # 将numpy数组转换为torch张量
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()  # 假设图像是灰度图
    # 使用双线性插值缩放到h*w
    img_resized = F.interpolate(img_tensor, size=(h, w), mode='bilinear', align_corners=False)
    # 将torch张量转换回numpy数组
    img_resized_np = img_resized.squeeze().numpy()
    return img_resized_np


import cv2
import random

def apply_augmentation(image, mask):
    """
    对图像和掩码应用相同的随机变换
    """
    # 随机旋转
    angle = random.randint(-30, 30)
    image = rotate_image(image, angle)
    mask = rotate_image(mask, angle)
    
    # 随机裁剪后resize回1024x1024
    crop_size = random.randint(800, 1000)
    start_x = random.randint(0, 1024 - crop_size)
    start_y = random.randint(0, 1024 - crop_size)
    
    image = image[start_y:start_y+crop_size, start_x:start_x+crop_size]
    mask = mask[start_y:start_y+crop_size, start_x:start_x+crop_size]
    
    image = resize_img(image)
    mask = resize_img(mask)
    
    # gamma变换调整亮度
    image = adjust_brightness(image)

    return image, mask

def rotate_image(image, angle):
    """
    旋转图像，保持尺寸不变
    """
    height, width = image.shape
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated

def adjust_brightness(img):
    '''
    期望输入image应该是0~255的uint8或者float类型
    '''
    # 使用对数正态分布
    mu = 0  # 对数正态分布的均值参数
    sigma = 0.3  # 控制分布的扩散程度
    gamma = np.random.lognormal(mu, sigma)

    # 归一化到[0,1]
    img_normalized = img / 255.0
    img_gamma = np.power(img_normalized, gamma)
    img= (img_gamma * 255).astype(np.uint8)
    return img

    
    


def train_model(data_list, num_epochs=50, batch_size=1, learning_rate=0.0001):
    # 划分训练集和验证集
    train_size = int(0.8 * len(data_list))
    train_data = data_list[:train_size]
    val_data = data_list[train_size:]
    print(f"训练集大小: {len(train_data)}, 验证集大小: {len(val_data)}")

    # 数据增强
    if not os.path.exists("unet_aug_debug"):
        os.makedirs("unet_aug_debug")

    augmented_train_data = []
    for data in train_data:
        # 原始数据
        augmented_train_data.append(data)
        Image.fromarray(data["X"].astype(np.uint8)).save(os.path.join("unet_aug_debug", f"{os.path.splitext(data['file_name'])[0]}.png"))
        Image.fromarray(data["Y"].astype(np.uint8)).save(os.path.join("unet_aug_debug", f"{os.path.splitext(data['file_name'])[0]}_mask.png"))
        # 增强数据
        for _ in range(4):  # 每个样本生成多个增强版本
            X_aug, Y_aug = apply_augmentation(data["X"], data["Y"])
            aug_data = {
                "file_name": data["file_name"] + f"_aug_{_}",
                "X": X_aug,
                "Y": Y_aug
            }
            # 保存图片看看内容
            Image.fromarray(X_aug.astype(np.uint8)).save(os.path.join("unet_aug_debug", f"{os.path.splitext(data['file_name'])[0]}_aug_{_}.png"))
            Image.fromarray(Y_aug.astype(np.uint8)).save(os.path.join("unet_aug_debug", f"{os.path.splitext(data['file_name'])[0]}_aug_{_}_mask.png"))
            augmented_train_data.append(aug_data)
    

    augmented_val_data = []
    for data in val_data:
        # 原始数据
        augmented_val_data.append(data)
        # 增强数据
        for _ in range(4):  # 每个样本生成多个增强版本
            X_aug, Y_aug = apply_augmentation(data["X"], data["Y"])
            aug_data = {
                "file_name": data["file_name"] + f"_aug_{_}",
                "X": X_aug,
                "Y": Y_aug
            }
            Image.fromarray(X_aug.astype(np.uint8)).save(os.path.join("unet_aug_debug", f"{os.path.splitext(data['file_name'])[0]}_aug_{_}.png"))
            augmented_val_data.append(aug_data)



    print("增强后的数据集已保存在unet_aug_debug文件夹下")

    # 使用增强后的训练数据
    train_data = augmented_train_data
    val_data = augmented_val_data
    print(f"增强后的训练集大小: {len(train_data)}")
    print(f"增强后的验证集大小: {len(val_data)}")


    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=1, n_classes=1).to(device)
    print(f"使用设备: {device}")

    # 定义Dice Loss
    class DiceLoss(nn.Module):
        def __init__(self, smooth=1e-5):
            super(DiceLoss, self).__init__()
            self.smooth = smooth
            
        def forward(self, pred, target):
            pred = torch.sigmoid(pred)
            pred = pred.view(-1)
            target = target.view(-1)
            intersection = (pred * target).sum()
            return 1 - ((2. * intersection + self.smooth) / 
                       (pred.sum() + target.sum() + self.smooth))

    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 记录最佳模型
    best_val_loss = float('inf')
    best_epoch_id=0
    best_model=None
    patience = 10
    patience_counter = 0
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        
        # 处理训练数据
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i + batch_size]
            
            # 准备批次数据
            X_batch = torch.stack([torch.from_numpy(d["X"]).unsqueeze(0).float() / 255.0 for d in batch_data]).to(device)
            Y_batch = torch.stack([torch.from_numpy(d["Y"]).unsqueeze(0).float() / 255.0 for d in batch_data]).to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / (len(train_data) // batch_size)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch_data = val_data[i:i + batch_size]
                
                # 准备批次数据
                X_batch = torch.stack([torch.from_numpy(d["X"]).unsqueeze(0).float() / 255.0 for d in batch_data]).to(device)
                Y_batch = torch.stack([torch.from_numpy(d["Y"]).unsqueeze(0).float() / 255.0 for d in batch_data]).to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                val_loss += loss.item()
                
                # 计算正确率
                # 使用 sigmoid 激活函数将输出转换为概率
                predicted = torch.sigmoid(outputs).view(-1)
                predicted = (predicted > 0.5).float()  # 二值化处理
                # 统计 TP, TN, FP, FN
                correct += (predicted == Y_batch.view(-1)).sum().item()  # TP
                total += Y_batch.numel()  # 总样本数
        
        avg_val_loss = val_loss / (len(val_data) // batch_size)
        accuracy = (correct / total) * 100  # 转换为百分比

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f},'
              f'Val Accuracy: {accuracy:.4f}%')
        
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_epoch_id=epoch

            best_model = copy.deepcopy(model)
        else:
            patience_counter += 1
        save_model(model, avg_val_loss, epoch+1)
        # 早停
        if patience_counter >= patience:
            print(f'早停: {patience} 个epoch没有改善')
            break
    print("最佳模型epoch=",best_epoch_id,"val loss=",best_val_loss)
    return best_model, best_val_loss




def save_model(model, loss, epoch, model_path='unet_model_output'):
    # 保存模型
    from datetime import datetime
    import os
    save_dir = model_path
    os.makedirs(save_dir, exist_ok=True)
    
    
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    loss_str=f"{loss:.4f}"
    model_name = f'Unet-{timestamp}-{epoch}-{loss_str}.pth'
    save_path = os.path.join(save_dir, model_name)
    
    torch.save(model.state_dict(), save_path)
    print(f'模型已保存至: {save_path}')


def main():
    print("读取数据集")
    img_lists=read_data(order="0010")
    img_list=img_lists[1]
    mask_list=read_mask()
    print("数据集读取完毕")
    # 输出数据类型
    print("img的数据类型为", img_list[0]["img"].dtype)  # 输出具体数据类型 
    print("mask的数据类型为", mask_list[0]["combined_mask"].dtype)  # 输出具体数据类型
    # 由于原来的数据集删了一个，所以需要重新定义一个数据集
    '''
    data={
        "file_name":"",
        "img"=[],
        "mask"=[],
        "X"=[],
        "Y"=[]    
    }
    '''
    data_list=[]
    
    print("开始组合数据")

    for i in img_list:
        # 在mask_list中寻找对应的mask
        succ_find=0
        for j in mask_list:
            if i["file_name"]==j["file_name"]:
                h=j["combined_mask"].shape[0]
                w=j["combined_mask"].shape[1]
                # 缩放到2048*2048

                img_resized_np=resize_img(i["img"]).astype(np.uint8)
                mask_resized_np=resize_img(j["combined_mask"]).astype(np.uint8)
                
                data={
                    "file_name":i["file_name"],
                    "img":i["img"],
                    "mask":j["combined_mask"],
                    "X":img_resized_np,
                    "Y":mask_resized_np
                }
                data_list.append(data)
                succ_find=1
                break

        if succ_find==0:
            print("警告: 找不到对应的遮罩文件")

    print("数据集处理完毕")
    print("得到了",len(data_list),"对数据")
    # 展示第一个 保存在本地
    print("展示第一个")
    
    if data_list:
        # 获取第一个数据
        first_data = data_list[0]
        img=first_data["img"]
        mask=first_data["mask"]
        img_resized_np = first_data["X"]
        mask_resized_np = first_data["Y"]
        # 输出类型
        print("img的数据类型为", img.dtype)
        print("mask的数据类型为", mask.dtype)
        print("img_resized_np的数据类型为", img_resized_np.dtype)
        print("mask_resized_np的数据类型为", mask_resized_np.dtype)
        import os
        # 创建保存目录（如果不存在）
        save_dir = "temp"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 保存图像
        img_path = os.path.join(save_dir, "first_image.png")
        mask_path = os.path.join(save_dir, "first_mask.png")
        X_path = os.path.join(save_dir, "first_X.png")
        Y_path = os.path.join(save_dir, "first_Y.png")
        # 将numpy数组转换为PIL图像
        img_pil = Image.fromarray((img).astype(np.uint8))
        mask_pil = Image.fromarray((mask).astype(np.uint8))
        X_pil = Image.fromarray((img_resized_np).astype(np.uint8))
        Y_pil = Image.fromarray((mask_resized_np).astype(np.uint8))
        # 保存图像
        img_pil.save(img_path)
        mask_pil.save(mask_path)
        X_pil.save(X_path)
        Y_pil.save(Y_path)

        print(f"图像已保存到 {img_path}")
        print(f"掩码已保存到 {mask_path}")
        print(f"缩放后的图像已保存到 {X_path}")
        print(f"缩放后的掩码已保存到 {Y_path}")
    else:
        print("数据列表为空，无法保存图像")

    #return
    # 训练
    model,loss=train_model(data_list, num_epochs=50, batch_size=1, learning_rate=0.0001)

    save_model(model,loss,50)









if __name__ == "__main__":
    main()