'''
期待文件路径树状图
data
│
├── ChinaSet_AllFiles
│   ├── ClinicalReadings
│   └── CXR_png
│
├── MontgomeryCXR
│   └── MontgomerySet
│       ├── ClinicalReadings
│       └── CXR_png
│
└── TB_Chest_Radiography_Database
    ├── Normal
    └── Tuberculosis
'''


import os
from PIL import Image
import numpy as np

# 传入指定路径，即data文件夹
def read_data(data_path=os.path.join(os.getcwd(),'data'),order="0111"):
    print('数据集根目录data_path:',data_path)

     # 定义每个数据集下的路径
    data_paths = {
        "ChinaSet_AllFiles": {
            "clinical_readings": os.path.join(data_path, 'ChinaSet_AllFiles', 'ClinicalReadings'),
            "cxr_png": os.path.join(data_path, 'ChinaSet_AllFiles', 'CXR_png')
        },
        "MontgomeryCXR": {
            "clinical_readings": os.path.join(data_path, 'MontgomeryCXR', 'MontgomerySet', 'ClinicalReadings'),
            "cxr_png": os.path.join(data_path, 'MontgomeryCXR', 'MontgomerySet', 'CXR_png')
        },
        "TB_Chest_Radiography_Database": {
            "path": os.path.join(data_path, 'TB_Chest_Radiography_Database'),
            "Tuberculosis": os.path.join(data_path, 'TB_Chest_Radiography_Database', 'Tuberculosis'),
            "Normal": os.path.join(data_path, 'TB_Chest_Radiography_Database', 'Normal')
        }
    }
    # print("111:",data_paths["ChinaSet_AllFiles"]["clinical_readings"])
    # print("111type:",type(data_paths["ChinaSet_AllFiles"]["clinical_readings"]))
    n1=count_files(data_paths["ChinaSet_AllFiles"]["clinical_readings"])
    print("n1:",n1)
    n2=count_files(data_paths["MontgomeryCXR"]["clinical_readings"])
    print("n2:",n2)
    n30=count_files(data_paths["TB_Chest_Radiography_Database"]["Normal"])
    n31=count_files(data_paths["TB_Chest_Radiography_Database"]["Tuberculosis"])
    print("n30:",n30)
    print("n31:",n31)

    # 读取数据!
   
    if order[1]=='1':
        

        # 处理第一个数据集（ChinaSet_AllFiles）
        data_list1=read_data1(data_paths["ChinaSet_AllFiles"]["cxr_png"],data_paths["ChinaSet_AllFiles"]["clinical_readings"])
        
        # 打印结果
        # for data in data_list:
        #     print(data)
        print(f"数据集n1读取完成，读取到{len(data_list1)}个文件")
        # 计算阳性比例
        positive_count1 = sum(data["positive"] for data in data_list1)
        print(f"数据集n1中阳性比例为{positive_count1/len(data_list1)}")
    else:
        data_list1=[]


    if order[2]=='1':
        # 处理第二个数据集（MontgomeryCXR）
        print("数据集2读取中...")
        data_list2=read_data2(data_paths["MontgomeryCXR"]["cxr_png"],data_paths["MontgomeryCXR"]["clinical_readings"])
        
        # 打印结果
        # for data in data_list:
        #     print(data)
        print(f"数据集n2读取完成，读取到{len(data_list2)}个文件")
        # 计算阳性比例
        positive_count2 = sum(data["positive"] for data in data_list2)
        print(f"数据集n2中阳性比例为{positive_count2/len(data_list2)}")
    else:
        data_list2=[]


    if order[3]=='1':

        # 处理第三个数据集（TB_Chest_Radiography_Database）
        print("数据集3读取中...")
        data_list3=read_data3(data_paths["TB_Chest_Radiography_Database"]["Normal"],data_paths["TB_Chest_Radiography_Database"]["Tuberculosis"])
        
        # 打印结果
        # for data in data_list:
        #     print(data)
        print(f"数据集n3读取完成，读取到{len(data_list3)}个文件")
        # 计算阳性比例
        positive_count3 = sum(data["positive"] for data in data_list3)
        print(f"数据集n3中阳性比例为{positive_count3/len(data_list3)}")
    else:
        data_list3=[]


    # import pickle
    # # 保存数据
    # with open('data_list1.pkl', 'wb') as f1:
    #     pickle.dump(data_list1, f1) # 预计5375.44M
    # with open('data_list2.pkl', 'wb') as f2:
    #     pickle.dump(data_list2, f2) # 预计2622M
    # with open('data_list3.pkl', 'wb') as f3:
    #     pickle.dump(data_list3, f3) # 3.0G




    return data_list1,data_list2,data_list3





    

def count_files(path):
    print("path:",path)
    if os.path.exists(path) and os.path.isdir(path):
        return len(os.listdir(path))
    else:
        return 0


def read_data1(cxr_png_path,clinical_readings_path):
    print("数据集1读取中...")
    
    # 读取 CXR_png 下的所有文件
    files = os.listdir(cxr_png_path)
    data_list1 = []
    for file_name in files:
        # print(f"正在处理文件: {file_name}")
        # 1. 检查文件名是否符合格式
        if not file_name.startswith("CHNCXR_") or not file_name.endswith(".png"):
            print(f"文件名不符合格式: {file_name}")
            continue
        
        # 2. 提取文件名中的关键信息
        try:
            prefix, idx, positive = file_name.split("_")
            # idx = int(idx)  # 序号不转int
            positive = int(positive.strip(".png"))  # 去掉 ".png" 并转换为数字类型
        except ValueError:
            print(f"文件名解析失败: {file_name}")
            continue
        try:
            positive=int(positive)
            if positive!=0 and positive!=1:
                print("患者肺结核阴阳属性的值不是0或1",txt_file_name)
                continue
        except:
            print("患者肺结核阴阳属性读取失败无法转化为数字",txt_file_name)
            continue
        # 3. 检查对应的 txt 文件是否存在
        txt_file_name = f"CHNCXR_{idx}_{positive}.txt"
        txt_file_path = os.path.join(clinical_readings_path, txt_file_name)
        
        if not os.path.exists(txt_file_path):
            print(f"对应的 txt 文件不存在: {txt_file_name}")
            continue
        
        # 4. 读取并解析 txt 文件
        with open(txt_file_path, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                print(f"txt 文件格式错误: {txt_file_name}")
                continue
            
            ## 解析第一行：性别和年龄
            '''
            gender_age = lines[0].strip().split()
            if len(gender_age) != 2:
                print(f"txt 文件第一行格式错误: {txt_file_name}")
                print(f"该文件的第一行的内容是: {lines[0]}")
                continue

            
            gender = gender_age[0].lower()  # 性别
            try:
                # 如果在gender_age[1]中存在"yrs"，则去掉"yrs"并转换为数字类型
                if "yrs" in gender_age[1]:
                    age = int(gender_age[1].strip("yrs"))  # 去掉 "yrs" 并转换为数字类型
                elif "month" in gender_age[1]:
                    age = int(gender_age[1].strip("month"))  # 去掉 "mo" 并转换为数字类型
                else:
                    print(f"txt 文件age格式错误: {txt_file_name}")
                    continue
            except:
                print(f"txt 文件age格式错误: {txt_file_name}")
                break
            '''
            ## 这部分很复杂，重构

            ### 去首尾空格，全部转小写
            gender_age=lines[0].strip()
            gender_age=gender_age.lower()
            ### 前6/4个字符是female还是male
            if len(gender_age)<4:
                print(f"txt 文件格式错误:第一行太短: {txt_file_name}")
                continue
            if gender_age[:4]=="fema": # 不能用前6字符判断，因为有“femal”
                gender=0
                age_str=gender_age[6:]
            elif gender_age[:4]=="male":
                gender=1
                age_str=gender_age[4:]
            else:
                print(f"txt 文件gender格式错误: {txt_file_name}: {lines[0]}")
                continue
            ### 处理剩下的内容
            #### 找出数字部分
            age=""
            for c in age_str:
                if c.isdigit():
                    age+=c
            if age == "":
                print(f"txt 文件age格式错误，找不到数字: {txt_file_name}")
                continue
            #### 判断这是年还是月
            if "yrs" in age_str and "month" in age_str and "days" in age_str:
                print(f"txt 文件有年有月有日，无法确定年月: {txt_file_name}")
                continue
            elif "yrs" in age_str or "yr" in age_str:
                age=int(age)
            elif "month" in age_str:
                age=int(age)/12
            elif "days" in age_str:
                age=int(age)/365
            else:
                # print(f"txt 文件无年无月无日，无法确定年月: {txt_file_name}: {lines[0]}")
                # 经过检查，仅CHNCXR_0018_0.txt: male 42是没有单位的。当作yrs
                if idx=="0018":
                    age=int(age)
                else:
                    print(f"txt 文件无年无月无日，无法确定年月: {txt_file_name}: {lines[0]}")
                    continue

            # 解析第二行：描述
            description = lines[-1].strip() # 部分数据txt存在中间有空一行的情况
            if description == "":
                print(f"txt 文件描述为空: {txt_file_name}")
                print(f"该文件内容为: {lines}")
                continue
        
        # 5. 读取图像文件并转换为 NumPy 数组
        image_path = os.path.join(cxr_png_path, file_name)
        try:
            img = Image.open(image_path).convert('L')
            img_array = np.array(img)
        except Exception as e:
            print(f"无法加载或转换图片: {e}")
            continue
        # 6. 构建数据结构
        data = {
            "file_name": file_name,
            "prefix": prefix,
            "idx": idx,
            "positive": positive, #0/1
            "gender": gender,  # 0: 女, 1: 男
            "age": age, #int or float
            "description": description,
            "img": img_array
        }
        # print(img_array)
        # 输出图像的最大值
        # print(f"图像最大值: {np.max(img_array)}")
        # return
        # print(img)
        data_list1.append(data)

        # 提前结束
        if idx=="0634":
            # break
            pass

        # print(idx)
    return data_list1


def read_data2(cxr_png_path,clinical_readings_path):
    # 读取 CXR_png 下的所有文件
    files = os.listdir(cxr_png_path)
    data_list2 = []
    for file_name in files:
        # print(f"正在处理文件: {file_name}")
        # 1. 检查文件名是否符合格式
        if not file_name.startswith("MCUCXR_") or not file_name.endswith(".png"):
            print(f"文件名不符合格式: {file_name}")
            continue
        
        # 2. 提取文件名中的关键信息
        try:
            prefix, idx, positive = file_name.split("_")
            # idx = int(idx)  # 序号不转int
            positive = int(positive.strip(".png"))  # 去掉 ".png" 并转换为数字类型
        except ValueError:
            print(f"文件名解析失败: {file_name}")
            continue
        try:
            positive=int(positive)
            if positive!=0 and positive!=1:
                print("患者肺结核阴阳属性的值不是0或1",txt_file_name)
                continue
        except:
            print("患者肺结核阴阳属性读取失败无法转化为数字",txt_file_name)
            continue
        # 3. 检查对应的 txt 文件是否存在
        txt_file_name = f"MCUCXR_{idx}_{positive}.txt"
        txt_file_path = os.path.join(clinical_readings_path, txt_file_name)
        
        if not os.path.exists(txt_file_path):
            print(f"对应的 txt 文件不存在: {txt_file_name}")
            continue
        
        # 4. 读取并解析 txt 文件
        with open(txt_file_path, 'r') as f:
            lines = f.readlines()
            # print(lines)
            if len(lines) < 2:
                print(f"txt 文件格式错误: {txt_file_name}")
                continue
            
            ## 解析第一行：性别

            ### 去首尾空格
            gender_age=lines[0].strip()
            # gender_age=gender_age.lower()
            ### 最后一个字符是M还是F
            if len(gender_age)<len("Patient's Sex: "):
                print(f"txt 文件格式错误:第一行太短: {txt_file_name}")
                continue
            if gender_age[-1]=="F":
                gender=0
            elif gender_age[-1]=="M":
                gender=1
            # elif gender_age[-1]=="O":
            #     gender=0.5
            else:
                print(f"txt 文件gender格式错误: {txt_file_name}: {lines[0]}")
                continue
            # 解析第二行：年龄
            age_list=lines[1].split()
            age_str=age_list[-1]
            age=age_str[:-1]
            if age_str[-1]=='Y':
                age=float(age)
            elif age_str[-1]=='M':
                age=float(age)/12
            else:
                print(f"txt 文件age格式错误: {txt_file_name}: {lines[1]}")
                continue

            # 解析第三行：描述
            # description = lines[-1].strip() # 部分数据txt存在中间有空一行的情况
            description = lines[2].strip()
            if description == "":
                print(f"txt 文件描述为空: {txt_file_name}")
                print(f"该文件内容为: {lines}")
                continue
        # print("读取图像文件并转换为 NumPy 数组")
        # 5. 读取图像文件并转换为 NumPy 数组
        image_path = os.path.join(cxr_png_path, file_name)
        try:
            img = Image.open(image_path).convert('L')
            img_array = np.array(img)
        except Exception as e:
            print(f"无法加载或转换图片: {e}")
            continue
        # 6. 构建数据结构
        data = {
            "file_name": file_name,
            "prefix": prefix,
            "idx": idx,
            "positive": positive, #0/1
            "gender": gender,  # 0: 女, 1: 男
            "age": age, #int or float
            "description": description,
            "img": img_array
        }
        # print(img_array)
        # 输出图像的最大值
        # print(f"图像最大值: {np.max(img_array)}")
        # return
        # print(img)
        data_list2.append(data)

        # 提前结束
        if idx=="0144":
            # break
            pass
        # print(idx)

    
    # 找0080
    # for data in data_list2:
    #     if data["idx"]=="0080":
    #         print(data)
    #         print("能够找到0080")
    #         break

    return data_list2




def read_data3(Normal_path,Tuberculosis_path):
    '''
    read_data3
    
    对于Normal，不读取1~406
    对于Tuberculosis，不读取307~700
    '''
    Normal_files = os.listdir(Normal_path)
    data_list3 = []
    for file_name in Normal_files:
        # print(f"正在处理文件: {file_name}")
        # 1. 检查文件名是否符合格式
        if not file_name.startswith("Normal-") or not file_name.endswith(".png"):
            print(f"文件名不符合格式: {file_name}")
            continue
        
        # 2. 提取文件名中的关键信息
        try:
            positive_str, idx= file_name.split("-")
            idx = idx.strip(".png")  # 去掉 ".png"
            positive=0
        except ValueError:
            print(f"文件名解析失败: {file_name}")
            continue

        # 排除1~406
        if int(idx)>=1 and int(idx)<=406:
            continue
        
        # print("读取图像文件并转换为 NumPy 数组")
        # 5. 读取图像文件并转换为 NumPy 数组
        image_path = os.path.join(Normal_path, file_name)
        try:
            img = Image.open(image_path).convert('L')
            img_array = np.array(img)
        except Exception as e:
            print(f"无法加载或转换图片: {e}")
            continue
        # 6. 构建数据结构
        data = {
            "file_name": file_name,
            "prefix": "TB_Chest_Radiography",
            "idx": idx,
            "positive": positive, # 0/1
            "gender": -1,  # 0: 女, 1: 男
            "age": -1, #int or float
            "description": None,
            "img": img_array
        }
        # print(img_array)
        # 输出图像的最大值
        # print(f"图像最大值: {np.max(img_array)}")
        # return
        # print(img)
        data_list3.append(data)

        # 提前结束
        if idx=="0080":
            # break
            pass
        # print(idx)
    
    print("data3 Normal读取完毕，数据量为",len(data_list3))
    # ----------------------------------------------------------

    Tuberculosis_files = os.listdir(Tuberculosis_path)
    for file_name in Tuberculosis_files:
        # print(f"正在处理文件: {file_name}")
        # 1. 检查文件名是否符合格式
        if not file_name.startswith("Tuberculosis-") or not file_name.endswith(".png"):
            print(f"文件名不符合格式: {file_name}")
            continue
        
        # 2. 提取文件名中的关键信息
        try:
            positive_str, idx= file_name.split("-")
            idx = idx.strip(".png")  # 去掉 ".png"
            idx+="0000" # 为了和Normal区分开。这样Tuberculosis的最小idx为10000
            positive=1
        except ValueError:
            print(f"文件名解析失败: {file_name}")
            continue

        # 排除307~700
        if int(idx)>=307 and int(idx)<=700:
            continue
        
        # print("读取图像文件并转换为 NumPy 数组")
        # 5. 读取图像文件并转换为 NumPy 数组
        image_path = os.path.join(Tuberculosis_path, file_name)
        try:
            img = Image.open(image_path).convert('L')
            img_array = np.array(img)
        except Exception as e:
            print(f"无法加载或转换图片: {e}")
            continue
        # 6. 构建数据结构
        data = {
            "file_name": file_name,
            "prefix": "TB_Chest_Radiography",
            "idx": idx,
            "positive": positive, #0/1
            "gender": -1,  # 0: 女, 1: 男
            "age": -1, #int or float
            "description": None,
            "img": img_array
        }
        # print(img_array)
        # 输出图像的最大值
        # print(f"图像最大值: {np.max(img_array)}")
        # return
        # print(img)
        data_list3.append(data)

        # 提前结束
        if idx=="0080":
            # break
            pass
        # print(idx)

    
    # 找0080
    # for data in data_list2:
    #     if data["idx"]=="0080":
    #         print(data)
    #         print("能够找到0080")
    #         break

    return data_list3






def main():
    print("read_data.py 测试模式。将自动调用read_data()并只读取第一个数据集的数据")
    read_data(order="0100")


if __name__=="__main__":
    main()