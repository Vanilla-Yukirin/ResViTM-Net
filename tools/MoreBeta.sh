#!/bin/bash
# 循环测试不同的beta值，[0.0:0.1:1.0]
for beta in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    echo "========================================="
    echo "开始训练，beta = $beta"
    echo "========================================="
    # 传递三个输入：0（模型）、1（数据集）、1（inv数据）
    echo -e "0\n1\n1" | python ./ResViTM-Net/MoreBeta/MoreBeta.py -b $beta
    
    if [ $? -eq 0 ]; then
        echo "beta = $beta 训练完成"
    else
        echo "beta = $beta 训练失败，退出"
        exit 1
    fi
    echo ""
done

echo "所有beta值训练完成！"