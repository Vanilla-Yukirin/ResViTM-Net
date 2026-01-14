"""
ROC曲线绘制脚本
1. 让用户输入模型路径
2. 导入模型和数据集
3. 计算模型预测概率和真实标签
4. 绘制ROC曲线，保存在根目录的ROC-output文件夹内
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

# 添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(current_dir, '..'))

# 导入训练脚本中定义的模型
import importlib.util
train_script_path = os.path.join(current_dir, '..', 'train_ResViTM.py')
spec = importlib.util.spec_from_file_location("train_ResViTM", train_script_path)
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)
ResViTM = train_module.ResViTM


def select_model_file():
    """让用户选择或输入模型路径"""
    model_dir = os.path.join('model_output', 'ResViTM')
    
    # 检查模型目录是否存在
    if not os.path.exists(model_dir):
        print(f"模型目录 {model_dir} 不存在")
        print("0. 自定义模型路径")
        choice = int(input("请选择 (输入0自定义): "))
        return None
    
    # 列出所有模型文件
    model_files = sorted([f for f in os.listdir(model_dir) 
                         if f.endswith('.pth') and f.startswith('ResViTM')])
    
    if not model_files:
        print(f"在 {model_dir} 中没有找到模型文件!")
        print("0. 自定义模型路径")
        choice = int(input("请选择 (输入0自定义): "))
        return None
    
    print("\n找到以下模型文件:")
    for i, model_file in enumerate(model_files):
        print(f"{i+1}. {model_file}")
    print("0. 自定义模型路径")
    
    while True:
        try:
            choice = int(input("\n请选择要使用的模型 (输入序号，0为自定义): "))
            if 1 <= choice <= len(model_files):
                return os.path.join(model_dir, model_files[choice-1])
            elif choice == 0:
                return None
            else:
                print(f"请输入0到{len(model_files)}之间的数字")
        except ValueError:
            print("请输入有效的数字")


def load_model(model_path, device):
    """加载模型"""
    print(f"\n加载模型: {model_path}")
    
    # 初始化模型
    model = ResViTM(
        img_size=1024,
        patch_size=16,
        in_channels=1,
        num_classes=2,
        embed_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.2
    )
    
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("模型加载成功!")
    return model


def load_dataset():
    """加载数据集"""
    print("\n加载数据集...")
    import utils.read_data_meta as read_data_meta
    data_lists = read_data_meta.main()
    
    # 合并所有数据
    all_data = []
    for data_list in data_lists:
        all_data.extend(data_list)
    
    print(f"数据集总量: {len(all_data)}")
    return all_data


def calculate_roc_metrics(model, dataset, batch_size=4, device='cuda'):
    """计算ROC曲线所需的指标"""
    print("\n计算预测概率和真实标签...")
    
    all_predictions = []
    all_labels = []
    
    # 按批次处理数据
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    pbar = tqdm(range(num_batches), desc="计算预测")
    
    with torch.no_grad():
        for batch_idx in pbar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            batch_data = dataset[start_idx:end_idx]
            
            # 准备批次数据
            X_batch = torch.stack([torch.from_numpy(d["img"]).unsqueeze(0).float() / 255.0 
                                  for d in batch_data]).to(device)
            
            # 准备元数据
            meta_batch = torch.zeros(len(batch_data), 5)
            for j, d in enumerate(batch_data):
                meta_batch[j, 0] = d.get("gender0", 0)
                meta_batch[j, 1] = d.get("gender1", 0)
                meta_batch[j, 2] = d.get("age0", 0)
                meta_batch[j, 3] = d.get("age1", 0)
                meta_batch[j, 4] = d.get("age2", 0)
            meta_batch = meta_batch.to(device)
            
            # 获取真实标签
            Y_batch = np.array([d["positive"] for d in batch_data])
            
            # 模型预测
            outputs = model(X_batch, meta_batch)
            predictions = outputs.cpu().numpy()
            
            all_predictions.extend(predictions.tolist())
            all_labels.extend(Y_batch.tolist())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    print(f"总样本数: {len(all_labels)}")
    print(f"正常样本(0): {np.sum(all_labels == 0)}")
    print(f"异常样本(1): {np.sum(all_labels == 1)}")
    
    return all_predictions, all_labels


def plot_roc_curve(predictions, labels, output_dir='ROC-output'):
    """绘制和保存ROC曲线"""
    print("\n绘制ROC曲线...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    # 打印ROC-AUC分数
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    
    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    
    # 绘制ROC曲线
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制ROC曲线
    ax.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.4f})')
    # 绘制随机分类器参考线
    ax.plot([0, 1], [0, 1], color='navy', lw=2.5, linestyle='--', label='Random Classifier')
    
    # 严格设置x、y轴限制为0~1
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    
    # 设置标签和标题，增加字号
    ax.set_xlabel('False Positive Rate', fontsize=16, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=16, fontweight='bold')
    ax.set_title('ROC Curve - ResViTM Model', fontsize=18, fontweight='bold', pad=20)
    
    # 增强图例
    ax.legend(loc="lower right", fontsize=14, framealpha=0.95, edgecolor='black')
    
    # 增强网格
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
    
    # 调整刻度标签字体大小
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # 设置脊线宽度
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    # 保存ROC曲线
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = os.path.join(output_dir, f'ROC_ResViTM_{timestamp}.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"ROC曲线已保存至: {output_path}")
    plt.close()
    
    # 保存详细的ROC指标信息
    metrics_path = os.path.join(output_dir, f'ROC_metrics_{timestamp}.txt')
    with open(metrics_path, 'w') as f:
        f.write("ROC Curve Analysis - ResViTM Model\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"ROC-AUC Score: {roc_auc:.4f}\n\n")
        f.write("False Positive Rates:\n")
        f.write(f"  {fpr}\n\n")
        f.write("True Positive Rates:\n")
        f.write(f"  {tpr}\n\n")
        f.write("Thresholds:\n")
        f.write(f"  {thresholds}\n")
    
    print(f"详细指标已保存至: {metrics_path}")
    
    return roc_auc, output_path


def main():
    """主函数"""
    print("=" * 60)
    print("ResViTM ROC曲线绘制工具")
    print("=" * 60)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 1. 选择/输入模型路径
    model_path = select_model_file()
    if model_path is None:
        model_path = input("\n请输入模型文件的完整路径: ")
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在!")
        return
    
    # 2. 加载模型
    model = load_model(model_path, device)
    
    # 3. 加载数据集
    dataset = load_dataset()
    
    # 4. 计算预测概率
    predictions, labels = calculate_roc_metrics(model, dataset, batch_size=4, device=device)
    
    # 5. 绘制ROC曲线
    roc_auc, output_path = plot_roc_curve(predictions, labels, output_dir='ROC-output')
    
    print("\n" + "=" * 60)
    print("ROC曲线绘制完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
