# 模型评估与统计分析流程

本文档介绍如何对original_demo_seed下不同seed模型进行批量评估和统计分析。

## 文件说明

### 主要脚本
- `repeat_result.sh` - 批量评估脚本（bash）
- `repeat_results_analysis.py` - 结果分析脚本（Python）

### 目录结构
```
outputs/
├── original_demo_seed/           # 训练好的模型
│   ├── seed42/
│   │   └── 20250929-211317/
│   │       └── checkpoints/
│   │           └── epoch=59.ckpt
│   ├── seed43/ ... seed50/       # 其他8个seed
└── original_seed_result/         # 评估结果
    ├── seed42/ ... seed50/       # 各seed评估结果
    ├── evaluation_statistics.txt # 完整统计结果
    ├── key_metrics_statistics.txt # 关键指标统计
    ├── detailed_results.txt      # 详细结果
    └── results_summary.csv       # CSV格式汇总
```

## 使用流程

### 第一步：批量评估
运行bash脚本进行批量模型评估：

```bash
./repeat_result.sh
```

**脚本功能：**
- 自动查找seed42-50的9个模型checkpoint文件
- 使用GPU 4进行评估
- 为每个seed生成独立的评估结果目录
- 每个结果目录包含`summary_metrics.txt`文件

**评估命令示例：**
```bash
CUDA_VISIBLE_DEVICES=4 python eval.py \
    model.target.model.use_low_rank=False \
    checkpoint="outputs/original_demo_seed/seed42/20250929-211317/checkpoints/epoch=59.ckpt" \
    output="original_seed_result/seed42" \
    gpus=1
```

### 第二步：统计分析
运行Python脚本分析评估结果：

```bash
python repeat_results_analysis.py
```

**脚本功能：**
- 自动读取9个seed的`summary_metrics.txt`文件
- 计算各指标的平均值、标准差、最小值、最大值
- 生成多种格式的统计报告

## 输出结果

### 控制台输出
显示12个关键指标的统计结果：
- `val_MR`, `val_minADE1`, `val_minADE6`, `val_minFDE1`, `val_minFDE6`, `val_b-minFDE6`
- `val_new_MR`, `val_new_minADE1`, `val_new_minADE6`, `val_new_minFDE1`, `val_new_minFDE6`, `val_new_b-minFDE6`

### 生成文件

1. **`evaluation_statistics.txt`** - 完整统计信息
   - 包含所有21个指标的详细统计
   - 平均值、标准差、最小值、最大值、原始值列表

2. **`key_metrics_statistics.txt`** - 关键指标统计
   - 只包含12个关键指标的统计信息
   - 格式与完整统计相同

3. **`detailed_results.txt`** - 各seed详细结果
   - 列出每个seed的所有指标值
   - 便于查看单个seed的具体表现

4. **`results_summary.csv`** - CSV格式汇总
   - 便于导入Excel或其他工具进行进一步分析
   - 包含指标名、平均值、标准差、最小值、最大值、样本数

## 统计指标说明

### 主要评估指标
- **MR (Miss Rate)**: 失误率，越低越好
- **ADE (Average Displacement Error)**: 平均位移误差，越低越好
- **FDE (Final Displacement Error)**: 最终位移误差，越低越好
- **b-minFDE**: 最佳最小最终位移误差

### 指标类别
- `val_*`: 标准验证指标
- `val_new_*`: 新的验证指标（可能使用了不同的评估方法）
- 数字后缀（1/6）表示使用的预测轨迹数量

## 注意事项

1. **GPU资源**: 脚本默认使用GPU 4，确保该GPU可用
2. **路径依赖**: 脚本假设在DeMo目录下运行
3. **文件完整性**: 确保所有seed目录都有对应的checkpoint文件
4. **评估时间**: 9个模型的完整评估需要较长时间，建议在后台运行

## 故障排除

### 常见问题
1. **找不到checkpoint文件**: 检查`outputs/original_demo_seed/`目录结构
2. **GPU内存不足**: 调整batch size或使用其他GPU
3. **参数解析错误**: 确保checkpoint路径使用引号包围

### 检查命令
```bash
# 检查所有checkpoint文件是否存在
find outputs/original_demo_seed -name "epoch=59.ckpt" | wc -l  # 应该输出9

# 检查评估结果
find outputs/original_seed_result -name "summary_metrics.txt" | wc -l  # 应该输出9
```