"""
t-SNE特征可视化工具 - 重构版本
简化的设计：每个模型自己提取特征，然后调用通用的t-SNE可视化函数
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 设置为非交互后端，适用于无GUI环境
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from datetime import datetime
import matplotlib.font_manager as fm
import torch.nn.functional as F

# 设置字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 手动添加Times New Roman字体文件
try:
    import matplotlib.font_manager as fm
    import os

    # 重新初始化字体管理器
    fm.fontManager.__init__()

    # 手动添加Times New Roman字体文件
    font_dir = os.path.expanduser('~/.local/share/fonts/times-new-roman')
    if os.path.exists(font_dir):
        font_files = ['TIMES.TTF', 'TIMESBD.TTF', 'TIMESI.TTF', 'TIMESBI.TTF']
        for font_file in font_files:
            font_path = os.path.join(font_dir, font_file)
            if os.path.exists(font_path):
                try:
                    fm.fontManager.addfont(font_path)
                except Exception:
                    pass  # 忽略重复添加等错误

except Exception as e:
    print(f"Font setup warning: {e}")
    # 如果字体设置失败，继续执行


def create_output_directory():
    """创建t-SNE输出目录"""
    output_dir = "t-SNE-output"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def generate_timestamp():
    """生成时间戳"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_tsne_visualization(features, labels, model_name):
    """
    通用t-SNE可视化函数
    Args:
        features: 提取的特征向量 [n_samples, feature_dim]
        labels: 对应的标签 [n_samples] (0=阴性, 1=阳性)
        model_name: 模型名称，用于文件命名
    Returns:
        output_path: 生成的图像文件路径
        features_2d: 2D降维后的特征
    """
    print(f"Starting t-SNE visualization for {model_name}...")
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
    
    # 创建输出目录
    output_dir = create_output_directory()
    
    # 生成时间戳文件名
    timestamp = generate_timestamp()
    filename = f"{model_name}-{timestamp}"
    
    print(f"Performing t-SNE dimensionality reduction...")
    # 执行t-SNE降维
    perplexity = min(30, len(features)-1)
    if perplexity < 5:
        perplexity = 5
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    features_2d = tsne.fit_transform(features)
    
    print(f"Generating visualization plot...")
    # 生成可视化图像
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 分别绘制阳性和阴性样本
    positive_mask = labels == 1
    negative_mask = labels == 0
    
    if np.any(positive_mask):
        ax.scatter(features_2d[positive_mask, 0], features_2d[positive_mask, 1], 
                  c='red', alpha=0.7, s=50, label='Positive')
    
    if np.any(negative_mask):
        ax.scatter(features_2d[negative_mask, 0], features_2d[negative_mask, 1], 
                  c='blue', alpha=0.7, s=50, label='Negative')
    
    # 设置更大的字体大小
    ax.set_xlabel('t-SNE Component 1', fontsize=16, fontweight='bold')
    ax.set_ylabel('t-SNE Component 2', fontsize=16, fontweight='bold')
    ax.set_title(f't-SNE Feature Visualization - {model_name}', fontsize=24, fontweight='bold')
    
    # 设置图例字体大小
    ax.legend(fontsize=14, frameon=True, fancybox=True, shadow=True)
    
    # 设置刻度标签字体大小
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    ax.grid(True, alpha=0.3)
    
    # 保存图像
    output_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存数据
    data_path = os.path.join(output_dir, f"{filename}_data.npz")
    np.savez(data_path, features=features, labels=labels, features_2d=features_2d)
    
    print(f"t-SNE visualization saved: {output_path}")
    return output_path, features_2d


def extract_features_for_tsne(model, val_data, n_samples=200, device='cuda', model_needs_meta=False, needs_3_channels=True, resize_to=None):
    """
    从模型中提取t-SNE用的特征
    Args:
        model: 训练好的模型（必须有extract_features方法）
        val_data: 验证集数据
        n_samples: 要使用的样本数量
        device: 计算设备
        model_needs_meta: 模型是否需要元数据（如CViTM）
        needs_3_channels: 模型是否需要3通道输入（预训练模型通常需要）
    Returns:
        features: 提取的特征向量 [n_samples, feature_dim]
        labels: 对应的标签 [n_samples]
    """
    model.eval()
    model = model.to(device)  # 确保模型在正确的设备上
    
    # 取前n个样本（-1表示全选）
    if n_samples == -1 or n_samples >= len(val_data):
        selected_data = val_data
        print(f"使用全部{len(val_data)}个样本。")
    else:
        selected_data = val_data[:n_samples]
    
    features_list = []
    labels_list = []
    
    # 批处理
    batch_size = 8
    with torch.no_grad():
        for i in range(0, len(selected_data), batch_size):
            batch_data = selected_data[i:i+batch_size]
            
            # 准备图像数据
            X_batch = torch.stack([
                torch.from_numpy(d["img"]).unsqueeze(0).float() / 255.0 
                for d in batch_data
            ]).to(device)
            
            # 如果模型需要3通道输入，复制单通道到3通道
            if needs_3_channels and X_batch.shape[1] == 1:
                X_batch = X_batch.repeat(1, 3, 1, 1)  # [B, 1, H, W] -> [B, 3, H, W]
            
            # 可选：按需resize到固定尺寸（如ViT 224x224），不做归一化/增强
            if resize_to is not None and (X_batch.shape[-1] != resize_to or X_batch.shape[-2] != resize_to):
                X_batch = F.interpolate(X_batch, size=(resize_to, resize_to), mode='bilinear', align_corners=False)
            
            # 根据模型类型调用特征提取
            if model_needs_meta:
                # CViTM类型模型需要元数据
                meta_batch = torch.zeros(len(batch_data), 5).to(device)
                for j, d in enumerate(batch_data):
                    meta_batch[j, 0] = d.get("gender0", 0)
                    meta_batch[j, 1] = d.get("gender1", 0)
                    meta_batch[j, 2] = d.get("age0", 0)
                    meta_batch[j, 3] = d.get("age1", 0)
                    meta_batch[j, 4] = d.get("age2", 0)
                
                features = model.extract_features(X_batch, meta_batch)
            else:
                # 标准模型
                features = model.extract_features(X_batch)
            
            # 收集特征和标签
            features_list.append(features.cpu().numpy())
            labels = [d["positive"] for d in batch_data]
            labels_list.extend(labels)
    
    # 合并所有特征
    all_features = np.vstack(features_list)
    all_labels = np.array(labels_list)
    
    return all_features, all_labels
