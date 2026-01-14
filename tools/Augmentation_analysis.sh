#!/bin/bash
# 循环测试不同的数据增强策略组合

# 增强策略组合数组
# 格式: "XYZ" 其中 X/Y/Z 为 0 或 1 表示 Rotation/Cropping/Brightness
strategies=(
    "000"
    "001"
    "010"
    "011"
    "100"
    "110"
)

# 对应的实验编号
exp_ids=(
    "1"
    "2"
    "3"
    "4"
    "5"
    "7"
)

for i in "${!strategies[@]}"
do
    strategy="${strategies[$i]}"
    exp_id="${exp_ids[$i]}"
    
    echo "========================================="
    echo "开始实验 $exp_id：增强策略 = $strategy"
    echo "Rotation=$(echo $strategy | cut -c1), Cropping=$(echo $strategy | cut -c2), Brightness=$(echo $strategy | cut -c3)"
    echo "========================================="
    
    # 传递增强策略到脚本
    # 0: 从无模型开始训练
    # 1: 选择第一个数据集
    # 1: inv数据选择
    echo -e "0\n1\n1\n$strategy" | python ./ResViTM-Net/Augmentation_analysis/Augmentation_analysis.py
    
    if [ $? -eq 0 ]; then
        echo "实验 $exp_id 训练完成"
    else
        echo "实验 $exp_id 训练失败，退出"
        exit 1
    fi
    echo ""
done

echo "所有数据增强实验训练完成！"
