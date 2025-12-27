#!/bin/bash
# 循环测试Adjust_Loss目录下的所有模型

models=(
    "CELoss.py"
    "MAELoss.py"
    "FocalLoss.py"
    "BCELoss.py"
    "MSELoss.py"
)

for model in "${models[@]}"
do
    echo "========================================="
    echo "开始训练模型：$model"
    echo "========================================="
    # 传递三个输入：0（模型）、1（数据集）、1（inv数据）
    echo -e "0\n1\n1" | python ./ResViTM-Net/Adjust_Loss/$model
    
    if [ $? -eq 0 ]; then
        echo "$model 训练完成"
    else
        echo "$model 训练失败，退出"
        exit 1
    fi
    echo ""
done

echo "所有模型训练完成！"
