'''
查询INV_CNN_output下面的INV_CNN.pth文件，并返回一个包含所有文件的列表，排序
询问要用哪个
模型构建文件来自 train_INV_CNN.py
read_data.read_data(order="0001")，在数据集3上应用这个模型
将预测结果保存到save/meta_data_new_3-20250409xxxxxx.txt中
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
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))


from train_INV_CNN import INV_CNN, select_model

def main():
    model_path=select_model()
    if not model_path:
        print("未选择模型，退出程序")
        return
    print(f"使用模型: {model_path}")

    # 加载模型
    model = INV_CNN()
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"模型已加载到 {device}")

    print("开始读取数据集3")
    from utils import read_data
    data_lists=read_data.read_data(order="0001")
    data_list=data_lists[2]
    print(f"数据集3包含 {len(data_list)} 个样本")

    print("开始缩放尺寸")
    from train_INV_CNN import pad_to_square
    for data in tqdm(data_list):
        data['img']=pad_to_square(data['img'], target_size=1024)

    # 创建保存目录
    save_dir = 'save'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 创建结果文件
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    result_file = os.path.join(save_dir, f"meta_data_new_3-{timestamp}.txt")

    # 准备结果列表
    results = []

    print("开始推理")
    with torch.no_grad():
        for data in tqdm(data_list):
            # 准备输入数据
            img = torch.from_numpy(data["img"]).unsqueeze(0).unsqueeze(0).float() / 255.0
            img = img.to(device)
            
            # 模型推理
            gender_outputs, age_outputs = model(img)
            
            # 获取预测结果
            _, gender_pred = torch.max(gender_outputs, 1)
            _, age_pred = torch.max(age_outputs, 1)
            
            gender_pred = gender_pred.item()
            age_pred = age_pred.item()
            
            
            # 保存结果
            result = {
                "file_name": data["file_name"],
                "gender": gender_pred,
                "age": age_pred
            }
            results.append(result)
    
    # 保存结果到文件
    with open(result_file, 'w') as f:
        for result in results:
            f.write(f"{result['file_name']}	{result['gender']}	{result['age']}\n")
    
    # 统计分布：性别(0女/1男) × 年龄段(0<=18,1:18~30,2:>30)
    stats = {
        (0, 0): 0,  # female, <18
        (1, 0): 0,  # male,   <18
        (0, 1): 0,  # female, 18~30
        (1, 1): 0,  # male,   18~30
        (0, 2): 0,  # female, >30
        (1, 2): 0,  # male,   >30
    }
    for r in results:
        key = (r['gender'], r['age'])
        if key in stats:
            stats[key] += 1

    print("统计信息（gender 0女/1男, age 0:<18, 1:18~30, 2:>30）:")
    print(f"female,<18 : {stats[(0, 0)]}")
    print(f"male,<18   : {stats[(1, 0)]}")
    print(f"female,18~30: {stats[(0, 1)]}")
    print(f"male,18~30  : {stats[(1, 1)]}")
    print(f"female,>30  : {stats[(0, 2)]}")
    print(f"male,>30    : {stats[(1, 2)]}")

    print(f"推理完成，结果已保存至: {result_file}")
    print(f"共处理了 {len(results)} 个样本")







if __name__ == "__main__":
    main()