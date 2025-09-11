'''
main函数：
读取save文件夹下面的文件夹，提取出所有以data_new开头的
按照文件名排序
像之前选择unet模型那样，询问用户要选择哪个文件夹里面的数据集

调用read_data_new.py中的read_data_new函数
'''
import os
from PIL import Image
import numpy as np
from tqdm import tqdm



def read_data_new(choose_path):
    print(f"准备读取数据集 from {choose_path}")
    data_lists=[]
    for num in range(1, 4):
        data_list=[]
        sub_path=os.path.join(choose_path, f'data{num}')
        print(f"读取子数据集 from {sub_path}")
        # 先读取metadata.txt，格式为一行一个，用\t分割
        # 文件名    positive    性别    年龄    描述
        meta_datas=[]
        # 先读取文件内容
        with open(os.path.join(sub_path, 'metadata.txt'), 'r') as f:
            lines = f.readlines()  # 读取所有行
        # 使用 tqdm 包装行列表
        for line in tqdm(lines, desc=f"读取子数据集 {num}", unit="条"):
            data={}
            line=line.strip()
            if line:
                meta_data=line.split('\t')
                meta_datas.append(meta_data)
                data['file_path']=os.path.join(sub_path, meta_data[0])
                data['file_name']=meta_data[0]
                data['positive']=int(meta_data[1])
                data['gender']=int(meta_data[2])
                data['age']=float(meta_data[3])
                data['description']=meta_data[4]
                # 开始读图
                img = Image.open(data['file_path'])
                img_array = np.array(img)
                data['img']=img_array

                data_list.append(data)
        print(f"读取子数据集 {sub_path} 完成，共读取 {len(data_list)} 张图像")
        data_lists.append(data_list)
    # 数据集三合一
    print("数据读取完毕，数据量为",sum([len(data_list) for data_list in data_lists]))
    return data_lists



def main():
    # 读取save文件夹下面的文件夹，提取出所有以data_new开头的
    dir_list = os.listdir(os.path.join(os.getcwd(), 'save'))
    dir_list = [dir for dir in dir_list if dir.startswith('data_new')]
    dir_list.sort()
    for i, dir_name in enumerate(dir_list):
        print(f'{i+1}. {dir_name}')
    # 询问用户要选择哪个文件夹里面的数据集
    choice = int(input('请选择要处理的数据集：')) - 1
    if choice < 0 or choice >= len(dir_list):
        print('输入错误，请重新输入！')
        return
    choose_path=os.path.join(os.getcwd(), 'save', dir_list[choice])
    return read_data_new(choose_path)

if __name__ == "__main__":
    main()

