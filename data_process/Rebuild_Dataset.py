'''
这个代码将会利用unet_model_output中的unet模型，
将read_data得到的所有图片数据进行肺叶切割：
    将所有图片使用双线性插值转化到SIZE=1024的尺寸
    将所有图片使用unet，得到遮罩矩阵
    将得到的遮罩图片再次使用双线性插值，将这个遮罩矩阵变换回原图的大小
    将遮罩矩阵二值化
    将二值化遮罩矩阵与原图进行合成，得到只包含肺叶区域的图片
    获取肺叶区域的rmin rmax cmin cmax
    根据rmax rmin cmax，将图片进行裁剪，得到最小有效正方形区域。
    现在，要将这个最小有效正方形区域根据“等比例缩放”的原则，
    使用双线性插值变成TARGET_SIZE*TARGET_SIZE的.具体等比例缩放的方法如下：
        假如得到的肺部遮罩数据为0-1矩阵X，其中1表示该像素属于肺叶，0表示不属于肺叶
        将宽边拉伸为1024像素，同时宽边同比例拉伸并且补充黑色区域，以保持肺部比例不变。
        具体的，如果图像为$h\times w(h<w)$，则拉伸比例为$\frac{1024}{w}$，原本长度为$h$的边将被拉伸至$h\times\frac{1024}{w}$
        并在两侧分别填充$\frac{1024-h\times\frac{1024}{w}}{2}$的空白区域（使用全0填充）。
    并且用类似于原本的数据集分类，保存在本地。
    预计保存在：data_new_{timestamp}下面的data1，data2，data3：
    其中每个文件夹下包含着：
        若干图片
        txt文本
        txt文本的格式，每行为几个信息：
            1.图片名
            2.样本性别
            3.样本年龄
            *4.样本病情的描述

'''


import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import time
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))


# 导入数据读取函数和模型
from utils.read_data import read_data
from data_process.train_unet import UNet

SIZE = 1024
TARGET_SIZE = 1024

def select_model():
    """选择要使用的模型文件"""
    model_dir = 'unet_model_output'
    if not os.path.exists(model_dir):
        print(f"错误: 模型目录 {model_dir} 不存在!")
        return None
    
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.pth')])
    if not model_files:
        print(f"错误: 在 {model_dir} 中没有找到模型文件!")
        return None
    
    print("找到以下模型文件:")
    for i, model_file in enumerate(model_files):
        print(f"{i+1}. {model_file}")
    
    while True:
        try:
            choice = int(input("请选择要使用的模型 (输入序号): "))
            if 1 <= choice <= len(model_files):
                return os.path.join(model_dir, model_files[choice-1])
            else:
                print(f"请输入1到{len(model_files)}之间的数字")
        except ValueError:
            print("请输入有效的数字")

def load_model(model_path):
    """加载UNet模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=1, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"模型已从 {model_path} 加载")
    return model, device

def resize_img(img, target_h, target_w):
    """使用双线性插值调整图像大小"""
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    resized_img = F.interpolate(img_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)
    return resized_img.squeeze().numpy().astype(np.uint8)


from skimage.filters import threshold_otsu
def process_image(img, model, device, size=SIZE, file_name="file_name"):
    """处理单张图像，生成遮罩并应用"""
    # 保存原始图像尺寸
    orig_h, orig_w = img.shape
    
    # 调整图像大小为模型输入尺寸
    img_resized = resize_img(img, size, size)
    
    # 转换为张量并归一化
    img_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)
    
    # 使用模型预测
    with torch.no_grad():
        output = model(img_tensor)
        mask_pred = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # 保存遮罩图（用于debug）
    mask_output = (mask_pred*255).astype(np.uint8)

    # 二值化遮罩
    # mask_binary = (mask_pred >= 0.5).astype(np.uint8) * 255

    # 自适应二值化
    threshold = threshold_otsu(mask_pred)
    mask_binary = (mask_pred >= threshold).astype(np.uint8) * 255

    # 面积下限
    # print("当前图像面积为",np.sum(mask_binary))
    if np.sum(mask_binary) < 255*1024*1024*0.10:
        print("当前阈值为",threshold)
        while np.sum(mask_binary) < 255*1024*1024*0.10 and threshold > 0.05:
            threshold -= 0.04
            mask_binary = (mask_pred >= threshold).astype(np.uint8) * 255
        print(f"{file_name} : 二值化遮罩面积不足，故将阈值调整为",threshold)
    if np.sum(mask_binary) < 255*1024*1024*0.10:
        threshold = -1
        mask_binary = (mask_pred >= threshold).astype(np.uint8) * 255
        print("放弃治疗了")
        # 放弃治疗了
    # 将掩码调整回原始图像尺寸
    mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).unsqueeze(0).float()
    mask_orig_size = F.interpolate(mask_tensor, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
    mask_orig_size = mask_orig_size.squeeze().numpy().astype(np.uint8)
    
    return mask_orig_size, img_resized, mask_binary, mask_output

def find_lung_region(mask, file_name):
    """找到肺部区域的边界"""
    # 找到所有非零像素的位置
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    # 获取边界
    if np.any(rows) and np.any(cols):
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax
    else:
        print(f"{file_name} : 无法找到肺部区域，将保存整个图像")
        return -1, -1, -1, -1

def crop_and_resize_with_padding(img, target_size=TARGET_SIZE):
    """根据遮罩裁剪图像，并调整为目标尺寸，保持比例"""
    try:
        # 裁剪图像
        cropped_img = img
        
        # 获取裁剪后的尺寸
        h, w = cropped_img.shape
        
        # 计算缩放比例以保持纵横比
        if w > h:  # 宽边是宽度
            scale = target_size / w
            new_w = target_size
            new_h = int(h * scale)
        else:  # 宽边是高度
            scale = target_size / h
            new_h = target_size
            new_w = int(w * scale)
        
        # 使用双线性插值调整大小
        img_tensor = torch.from_numpy(cropped_img).unsqueeze(0).unsqueeze(0).float()
        resized_img = F.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
        resized_img = resized_img.squeeze().numpy()
        
        # 创建目标大小的空白图像
        result_img = np.zeros((target_size, target_size), dtype=np.uint8)
        
        # 计算填充的起始位置，使图像居中
        start_h = (target_size - new_h) // 2
        start_w = (target_size - new_w) // 2
        
        # 将调整大小后的图像放入目标图像
        result_img[start_h:start_h+new_h, start_w:start_w+new_w] = resized_img
        
        return result_img
        
    except Exception as e:
        print(f"裁剪和调整大小时出错: {str(e)}")
        # 如果出错，返回原图调整到目标大小
        return resize_img(img, target_size, target_size)

def create_dataset_structure():
    """创建新数据集的目录结构"""
    # base_dir = 'data_new'
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base_dir = os.path.join("save",f"data_new-{timestamp}")
    dirs = ['data1', 'data2', 'data3']
    
    for dir_name in dirs:
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    return base_dir, dirs

def save_metadata(base_dir, dir_name, data_list):
    """保存元数据信息到文本文件"""
    metadata_path = os.path.join(base_dir, dir_name, 'metadata.txt')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for data in data_list:
            # 构建元数据行
            file_name = os.path.basename(data.get('processed_img_path', ''))
            gender = data.get('gender', -1)
            age = data.get('age', -1)
            description = data.get('description', 'No description')
            positive = data["positive"]
            # 写入文件
            f.write(f"{file_name}	{positive}	{gender}	{age}	{description}\n")

def apply_mask_to_image(img, mask):
    """将遮罩应用到图像上，只保留肺叶区域"""
    # 确保mask是二值的
    binary_mask = mask.astype(bool)
    # 创建与原图相同大小的全黑图像
    masked_img = np.zeros_like(img)
    # 只保留遮罩区域内的像素
    masked_img[binary_mask] = img[binary_mask]
    return masked_img

def debug_first_image(data_lists, model, device):
    """调试第一张图片的处理过程"""
    temp_dir = "temp_debug"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    if data_lists and data_lists[2]:
        debug_data = data_lists[2][0]
        debug_img = debug_data['img']
        print(f"调试图片: {debug_data['file_name']}, 形状: {debug_img.shape}")
        
        # 保存原始图片
        Image.fromarray(debug_img).save(os.path.join(temp_dir, "01_original.png"))
        
        # 调整大小到SIZE用于模型输入
        debug_img_resized = resize_img(debug_img, SIZE, SIZE)
        Image.fromarray(debug_img_resized).save(os.path.join(temp_dir, "02_resized.png"))
        
        # 生成遮罩
        mask_orig_size, img_resized, mask_1024, mask_output = process_image(debug_img, model, device, file_name=debug_data['file_name'])
        Image.fromarray(mask_output).save(os.path.join(temp_dir, "03_mask_output.png"))
        # 保存遮罩
        Image.fromarray(mask_orig_size * 255).save(os.path.join(temp_dir, "03a_mask_orig_size.png"))
        Image.fromarray(mask_1024).save(os.path.join(temp_dir, "03b_mask_1024.png"))
        
        # 应用遮罩到原图
        masked_img = apply_mask_to_image(debug_img, mask_orig_size)
        Image.fromarray(masked_img).save(os.path.join(temp_dir, "04_masked_image.png"))
        
        try:
            # 找到肺部区域
            rmin, rmax, cmin, cmax = find_lung_region(mask_orig_size,debug_data['file_name'])
            print(f"肺部区域边界: 行({rmin}, {rmax}), 列({cmin}, {cmax})")
            if rmin == -1:
                print("无法找到肺部区域，将保存整个图像")
                rmin, rmax, cmin, cmax = 0, masked_img.shape[0]-1, 0, masked_img.shape[1]-1
                print("边界已设置为整个图像")
        
            # 在原图上标记边界
            marked_border_img = masked_img.copy()
            line_width = 5
            marked_border_img[rmin:rmin+line_width, :] = 255
            marked_border_img[rmax-line_width+1:rmax+1, :] = 255
            marked_border_img[:, cmin:cmin+line_width] = 255
            marked_border_img[:, cmax-line_width+1:cmax+1] = 255
            Image.fromarray(marked_border_img).save(os.path.join(temp_dir, "05_marked_boundaries.png"))
        
            # 裁剪图像
            cropped_img = masked_img[rmin:rmax+1, cmin:cmax+1]
            Image.fromarray(cropped_img).save(os.path.join(temp_dir, "06_cropped.png"))
            
            # 等比例缩放并填充
            final_img = crop_and_resize_with_padding(cropped_img)
            Image.fromarray(final_img).save(os.path.join(temp_dir, "07_final_result.png"))
            
        except Exception as e:
            print(f"调试过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"调试图像已保存到 {temp_dir} 目录")

def main():
    print("开始重塑数据集...")
    
    # 选择并加载模型
    model_path = select_model()
    if not model_path:
        return
    
    model, device = load_model(model_path)
    
    # 读取数据集
    print("读取原始数据集...")
    data_lists = read_data(order="0111")
    
    # 调试第一张图片
    print("开始调试第一张图片...")
    debug_first_image(data_lists, model, device)
    
    # 创建新数据集目录结构
    base_dir, dirs = create_dataset_structure()
    
    # 处理每个数据集
    for i, data_list in enumerate(data_lists):
        if not data_list or i >= len(dirs):
            continue
        
        dir_name = dirs[i]
        print(f"处理数据集 {i+1} 到 {dir_name}...")
        
        # 创建保存处理后图像的目录
        images_dir = os.path.join(base_dir, dir_name)
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        
        # 处理每张图像
        processed_data = []
        for data in tqdm(data_list, desc=f"处理数据集 {i+1}"):
            try:
                # 获取原始图像
                img = data['img']
                
                # 处理图像，生成遮罩
                mask_orig_size, _, _, mask_output = process_image(img, model, device, file_name=data["file_name"])
                
                # 保存遮罩
                if not os.path.exists(os.path.join(base_dir, "debug")):
                    os.makedirs(os.path.join(base_dir, "debug"))
                mask_path = os.path.join(os.path.join(base_dir, "debug"), f"{os.path.splitext(data['file_name'])[0]}_mask.png")
                Image.fromarray(mask_output).save(mask_path)

                # 应用遮罩到原图，只保留肺部区域
                masked_img = apply_mask_to_image(img, mask_orig_size)
                
                # 找到肺部区域
                rmin, rmax, cmin, cmax = find_lung_region(mask_orig_size, data["file_name"])
                if rmin == -1:
                    print("无法找到肺部区域，将保存整个图像")
                    rmin, rmax, cmin, cmax = 0, masked_img.shape[0]-1, 0, masked_img.shape[1]-1
                    print("边界已设置为整个图像")
                # 裁剪出肺部区域
                cropped_img = masked_img[rmin:rmax+1, cmin:cmax+1]
                
                # 等比例缩放并填充
                processed_img = crop_and_resize_with_padding(cropped_img)
                
                # 保存处理后的图像
                img_filename = f"{os.path.splitext(data['file_name'])[0]}.png"
                img_path = os.path.join(images_dir, img_filename)
                Image.fromarray(processed_img).save(img_path)
                
                # 添加到处理后的数据列表
                data_copy = data.copy()
                data_copy['processed_img_path'] = img_path
                processed_data.append(data_copy)
                
            except Exception as e:
                print(f"处理图像 {data['file_name']} 时出错: {str(e)}")
        
        # 保存元数据
        save_metadata(base_dir, dir_name, processed_data)
        
        print(f"数据集 {i+1} 处理完成，共处理 {len(processed_data)} 张图像")
    
    print("数据集重塑完成!")

if __name__ == "__main__":
    main()
