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
        
        # 将model_output移动到autodl持久化目录
        echo "正在移动模型文件到 /root/autodl-tmp ..."
        mkdir -p /root/autodl-tmp/model_output
        if [ $? -ne 0 ]; then
            echo "错误: 无法创建目标目录 /root/autodl-tmp/model_output"
            exit 1
        fi
        
        move_success=0
        if command -v rsync &> /dev/null; then
            # 使用rsync支持增量同步，--remove-source-files会在同步后删除源文件
            rsync -av --progress --remove-source-files ./model_output/ /root/autodl-tmp/model_output/
            if [ $? -eq 0 ]; then
                move_success=1
            fi
        else
            # 如果没有rsync，先尝试mv，失败后降级到cp+rm
            mv ./model_output/* /root/autodl-tmp/model_output/ 2>/dev/null
            if [ $? -eq 0 ]; then
                move_success=1
            else
                # mv失败，使用cp+rm方案
                cp -r ./model_output/* /root/autodl-tmp/model_output/ 2>/dev/null
                if [ $? -eq 0 ]; then
                    rm -rf ./model_output/*
                    if [ $? -eq 0 ]; then
                        move_success=1
                    fi
                fi
            fi
        fi
        
        if [ $move_success -eq 1 ]; then
            echo "模型文件移动成功"
        else
            echo "警告: 模型文件移动失败，但继续训练下一个beta值"
        fi
    else
        echo "beta = $beta 训练失败，退出"
        exit 1
    fi
    echo ""
done

echo "所有beta值训练完成！"
echo "所有模型已移动到 /root/autodl-tmp/model_output/"