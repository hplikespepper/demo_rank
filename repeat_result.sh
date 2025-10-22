#!/bin/bash
set -e
# 批量评估original_demo_seed下不同seed目录中的模型性能
# 该脚本会评估seed42到seed50的9个模型

echo "开始批量评估original_demo_seed下的模型..."

# 设置GPU
export CUDA_VISIBLE_DEVICES=4

# 定义seed范围
# seeds=(42 43 44 45 46 47 48 49 50)
seeds=(43 44 47 49 50)

# seeds=(42)

# 基础路径
base_path="outputs/original_demo_seed"
# output_base="original_seed_result"
output_base="original_seed_result_class_3"

# base_path="outputs/lowrank_r32_seed"
# # output_base="lowrank_seed_result"
# output_base="lowrank_seed_result_class_3"

# 遍历所有seeds
for seed in "${seeds[@]}"; do
    echo ""
    echo "=================================================="
    echo "Processing seed${seed}"
    echo "=================================================="
    
    # 查找checkpoint路径
    seed_dir="${base_path}/seed${seed}"               #记得换
    # seed_dir="${base_path}/lowrank_r32_seed${seed}"
    if [ ! -d "$seed_dir" ]; then
        echo "Warning: $seed_dir does not exist, skipping..."
        continue
    fi
    
    # 查找timestamp目录（假设只有一个）
    timestamp_dir=$(find "$seed_dir" -maxdepth 1 -type d -name "20*" | sort | tail -1)
    if [ -z "$timestamp_dir" ]; then
        echo "Warning: No timestamp directory found in $seed_dir, skipping..."
        continue
    fi
    
    # 构建checkpoint路径
    checkpoint_path="${timestamp_dir}/checkpoints/epoch=59.ckpt"
    if [ ! -f "$checkpoint_path" ]; then
        echo "Warning: $checkpoint_path does not exist, skipping..."
        continue
    fi
    
    # 设置输出目录
    output_dir="${output_base}/seed${seed}"
    
    echo "Checkpoint: $checkpoint_path"
    echo "Output: $output_dir"
    
    # 运行评估命令
    echo "Running evaluation..."
    echo "Command: python eval.py model.target.model.use_low_rank=False checkpoint='$checkpoint_path' output='$output_dir' gpus=1"
    
    #
    # python eval.py \
    #     model.target.model.use_low_rank=False \
    #     "checkpoint='$checkpoint_path'" \
    #     "output='$output_dir'" \
    #     gpus=1

    # # 普通评估
    # python eval.py \
    #     model.target.model.low_rank_dim=32 \
    #     "checkpoint='$checkpoint_path'" \
    #     "output='$output_dir'" \
    #     gpus=1
    
    # # 按agent类型评估
    # python eval_type2.py \
    #     model.target.model.low_rank_dim=32 \
    #     "checkpoint='$checkpoint_path'" \
    #     "output='$output_dir'" \
    #     gpus=1

    # # 按agent类型评估
    python eval_type2.py \
        model.target.model.use_low_rank=False \
        "checkpoint='$checkpoint_path'" \
        "output='$output_dir'" \
        gpus=1
    
    if [ $? -eq 0 ]; then
        echo "Successfully evaluated seed${seed}"
    else
        echo "Error: Failed to evaluate seed${seed}"
    fi
done

echo ""
echo "=================================================="
echo "批量评估完成！"
echo "=================================================="
echo "请运行 python repeat_results_analysis.py 来分析结果"