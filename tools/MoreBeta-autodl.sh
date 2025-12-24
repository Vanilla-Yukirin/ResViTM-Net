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
        
        # 将model_output同步到autodl持久化目录
        echo "正在同步模型文件到 /root/autodl-tmp ..."
        if command -v rsync &> /dev/null; then
            # 使用rsync支持增量同步和合并
            rsync -av --progress ./model_output/ /root/autodl-tmp/model_output/
        else
            # 如果没有rsync，使用cp
            mkdir -p /root/autodl-tmp/model_output
            cp -r ./model_output/* /root/autodl-tmp/model_output/
        fi
        
        if [ $? -eq 0 ]; then
            echo "模型文件同步成功"
        else
            echo "警告: 模型文件同步失败，但继续训练下一个beta值"
        fi
    else
        echo "beta = $beta 训练失败，退出"
        exit 1
    fi
    echo ""
done

echo "所有beta值训练完成！"
echo "所有模型已同步到 /root/autodl-tmp/model_output/"