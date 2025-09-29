#!/bin/bash

# set -e

# # --- 在这里设置公共命令 ---
# # 将 GPU 设备 ID 和 python 命令设置为一个变量
# export CUDA_VISIBLE_DEVICES=4,5,6,7

# echo "开始执行实验..."
# start_time=$(date +%s)
# echo "开始时间: $(date)"

# # 实验1：原始方案（不使用低秩）
# echo "------------------------------------"
# echo "正在运行实验1:原始方案 (original_demo)"
# echo "------------------------------------"
# python train.py model.target.model.use_low_rank=False output=original_demo gpus=4

# # # 实验2：低秩分解 rank=8
# # # echo "------------------------------------"
# # # echo "正在运行实验2：低秩分解 rank=8 (lowrank_r8)"
# # # echo "------------------------------------"
# # # python train.py model.target.model.low_rank_dim=8 output=lowrank_r8

# # # # 实验3：低秩分解 rank=16（默认）
# # # echo "------------------------------------"
# # # echo "正在运行实验3：低秩分解 rank=16 (lowrank_r16)"
# # # echo "------------------------------------"
# # # python train.py model.target.model.low_rank_dim=16 output=lowrank_r16

# # 实验4：低秩分解 rank=32
# echo "------------------------------------"
# echo "正在运行实验4:低秩分解 rank=32 (lowrank_r32)"
# echo "------------------------------------"
# python train.py model.target.model.low_rank_dim=32 output=lowrank_r32 gpus=4

# # # 实验5：低秩分解 rank=48
# # echo "------------------------------------"
# # echo "正在运行实验5：低秩分解 rank=48 (lowrank_r48)"
# # echo "------------------------------------"
# # python train.py model.target.model.low_rank_dim=48 output=lowrank_r48

# # # 实验6：低秩分解 rank=64
# # echo "------------------------------------"
# # echo "正在运行实验6：低秩分解 rank=64 (lowrank_r64)"
# # echo "------------------------------------"
# # python train.py model.target.model.low_rank_dim=64 output=lowrank_r64

# echo "------------------------------------"
# echo "所有实验均已完成！"
# echo "完成时间: $(date)"
# echo "总耗时: $(( ($(date +%s) - start_time) / 60 )) 分钟"
# echo "------------------------------------"




set -e 

# --- 在这里设置公共命令 --- 
# 将 GPU 设备 ID 和 python 命令设置为一个变量 
export CUDA_VISIBLE_DEVICES=3,4,5,7 

echo "开始执行实验..." 
total_start_time=$(date +%s) 
echo "总开始时间: $(date)" 

# 定义seed范围
SEEDS=(42 43 44 45 46 47 48 49 50)

# 循环执行10次实验
for seed in "${SEEDS[@]}"
do
    echo "===================================="
    echo "正在运行实验: 低秩分解 rank=32 - Seed: $seed"
    echo "===================================="
    
    # 记录单次实验开始时间
    exp_start_time=$(date +%s)
    echo "实验开始时间: $(date)"
    
    # # 执行训练命令，输出目录包含seed信息
    # python train.py \
    #     model.model.use_low_rank=False \
    #     output=original_demo_seed${seed} \
    #     gpus=4 \
    #     seed=$seed

    python train.py model.target.model.low_rank_dim=32 output=lowrank_r32_seed${seed} gpus=4 seed=$seed
    
    # 记录单次实验结束时间
    exp_end_time=$(date +%s)
    exp_duration=$((exp_end_time - exp_start_time))
    echo "Seed $seed 实验完成，用时: $((exp_duration / 60)) 分钟 $((exp_duration % 60)) 秒"
    echo "------------------------------------"
    echo ""
done

# 计算总时间
total_end_time=$(date +%s)
total_duration=$((total_end_time - total_start_time))

echo "===================================="
echo "所有实验完成!"
echo "总用时: $((total_duration / 3600)) 小时 $((total_duration % 3600 / 60)) 分钟 $((total_duration % 60)) 秒"
echo "结束时间: $(date)"
echo "===================================="
