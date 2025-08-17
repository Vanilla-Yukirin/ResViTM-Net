
import os
import numpy as np
from PIL import Image

def read_mask():
    left_mask_path = 'data/MontgomeryCXR/MontgomerySet/ManualMask/leftMask'
    right_mask_path = 'data/MontgomeryCXR/MontgomerySet/ManualMask/rightMask'
    

    mask_list = []  # 用于存储所有遮罩数据的列表

    # 读取左侧遮罩
    left_mask_files = os.listdir(left_mask_path)
    for file_name in left_mask_files:
        if not file_name.endswith('.png'):
            continue  # 仅处理 PNG 文件
        
        # 获取图像的文件名和索引
        prefix, idx, positive = file_name.split("_")
        positive = int(positive.strip(".png"))  # 获取阳性状态
        
        # 读取左侧遮罩图像
        left_image_path = os.path.join(left_mask_path, file_name)
        left_mask = Image.open(left_image_path)
        left_mask_array = np.array(left_mask)

        # 读取相应的右侧遮罩
        right_image_name = f"{prefix}_{idx}_{positive}.png"
        right_image_path = os.path.join(right_mask_path, right_image_name)
        if os.path.exists(right_image_path):
            right_mask = Image.open(right_image_path)
            right_mask_array = np.array(right_mask)
           
            # 合并左右遮罩，使用逻辑或操作
            combined_mask = np.logical_or(left_mask_array, right_mask_array).astype(np.uint8)*255

            # 构建数据结构
            data = {
                "file_name": file_name,
                "prefix": prefix,
                "idx": idx,
                "positive": positive,
                # 不要保存左右部分，以节省内存
                # "left_mask": left_mask_array,
                # "right_mask": right_mask_array,
                "combined_mask": combined_mask
            }
            mask_list.append(data)
        else:
            print(f"警告: 找不到对应的右侧遮罩文件 {right_image_name}")


    print(f"总共读取到 {len(mask_list)} 个遮罩文件.")
    print("遮罩数据结构:", mask_list[0])
    mask_area = np.sum(mask_list[0]["combined_mask"].astype(np.float32)/255)
    print("common_mask_list sum=",mask_area)
    total_pixels = mask_list[0]["combined_mask"].shape[0] * mask_list[0]["combined_mask"].shape[1]
    area_percentage = (mask_area / total_pixels) * 100
    print(f"遮罩面积：{mask_area}")
    print(f"总像素数：{total_pixels}")
    print(f"面积占比：{area_percentage:.2f}%")
    return mask_list


def main():
    read_mask()
if __name__ == "__main__":
    main()
