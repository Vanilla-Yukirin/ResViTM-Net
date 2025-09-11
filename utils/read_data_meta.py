'''
main函数：先直接调用read_data_new，然后再将第三个数据集读取inv数据


'''
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def select_inv_data(inv_path="save"):
    inv_files = sorted([f for f in os.listdir(inv_path) if f.endswith('.txt') and f.startswith('meta_data_new_3-')])
    if not inv_files:
        print(f"在 {inv_path} 中没有找到模型文件!")
        return None
    print("找到以下模型文件:")
    inv_files.sort()
    for i, inv_file in enumerate(inv_files):
        print(f"{i+1}. {inv_file}")
    while True:
        try:
            choice = int(input("请选择要使用的模型 (输入序号，输入0时退出): "))
            if 1 <= choice <= len(inv_files):
                inv_meta_data_path=os.path.join(inv_path, inv_files[choice-1])
                # 开始读取
                with open(inv_meta_data_path, 'r') as f:
                    lines = f.readlines()
                inv_data_list = []
                for line in tqdm(lines, desc="读取预测结果"):
                    try:
                        parts = line.strip().split('\t')
                        if len(parts) == 3:
                            # 格式: file_name\t性别\t年龄段
                            file_name = parts[0]
                            gender = int(parts[1])  # 0或1
                            age_group = int(parts[2])  # 0、1或2
                            
                            inv_data = {
                                "file_name": file_name,
                                "gender": gender,
                                "age_group": age_group
                            }
                            inv_data_list.append(inv_data)
                        else:
                            print(f"警告: 跳过格式不正确的行: {line}")
                    except Exception as e:
                        print(f"处理行时出错: {line}, 错误: {e}")
                
                print(f"成功读取 {len(inv_data_list)} 条预测结果")
                return inv_data_list
            

            elif choice == 0:
                return None
            else:
                print(f"请输入0到{len(inv_files)}之间的数字")
        except ValueError:
            print("请输入有效的数字")





def main():
    import utils.read_data_new as read_data_new
    data_lists=read_data_new.main()


    age_split1=18
    age_split2=30
    print("读取数据集完毕。现在开始计算数据集1、2的分类")
    for num in range(0,2):
        data_list=data_lists[num]
        for data in tqdm(data_list, desc=f"计算数据集 {num+1}", unit="条"):
            data['gender']=int(data['gender']) # 0/1
            if data['gender']==0:
                data['gender0']=1
                data['gender1']=0
            else:
                data['gender0']=0
                data['gender1']=1
            data['age0']=int(data['age']<=age_split1)
            data['age1']=int(data['age']>age_split1 and data['age']<=age_split2)
            data['age2']=int(data['age']>age_split2)
            if data['age']<=age_split1:
                data['age_group'] = 0  # 第一类
            elif data['age']>age_split1 and data['age']<=age_split2:
                data['age_group'] = 1
            else:
                data['age_group'] = 2
    
    print("数据集12处理完毕。现在开始读取第三个数据集的meta预测数据")
    meta3=select_inv_data()
    if meta3 is None:
        return
    
    print("开始合并数据")
    succ=0
    for meta in tqdm(meta3, desc="合并数据", unit="条"):
        file_name=meta["file_name"]
        for data in data_lists[2]:
            if data["file_name"]==file_name:
                data["gender"]=meta["gender"]
                if data['gender']==0:
                    data['gender0']=1
                    data['gender1']=0
                else:
                    data['gender0']=0
                    data['gender1']=1
                data["age_group"]=meta["age_group"]
                data['age0']=data['age1']=data['age2']=0
                data["age0"]=int(data['age_group']==0)
                data["age1"]=int(data['age_group']==1)
                data['age2']=int(data['age_group']==2)
                succ+=1
                break

    print(f"成功合并{succ}条数据")
    print("第三个数据集结构如下：")
    print(data_lists[2][0])

    # 统计三个数据集的信息
    for num in range(len(data_lists)):
        data_list=data_lists[num]
        print(f"数据集 {num+1} 的数据量为 {len(data_list)} 条")
        age_cnt=[0,0,0]
        gender_cnt=[0,0]
        age_gender_cnt=[[0,0],[0,0],[0,0]]
        for data in data_list:
            age_cnt[data['age_group']]+=1
            gender_cnt[data['gender']]+=1
            age_gender_cnt[data['age_group']][data['gender']]+=1
        print(f"性别分布为： 女:男={gender_cnt[0]}:{gender_cnt[1]}", f"年龄分布为： 0-{age_split1}:{age_split1}~{age_split2}:>{age_split2}={age_cnt[0]}:{age_cnt[1]}:{age_cnt[2]}")
        for i in range(3):
            print(f"在年龄段{i+1}中，女性={age_gender_cnt[i][0]}，男性={age_gender_cnt[i][1]}")

    return data_lists


if __name__ == "__main__":
    main()