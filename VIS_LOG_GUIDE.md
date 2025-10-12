# TensorBoard 日志可视化工具使用指南

`vis_log.py` 是一个用于读取和可视化 TensorBoard 日志文件的命令行工具，可以快速绘制训练过程中的各类指标曲线。

## 📋 目录

- [功能特性](#功能特性)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [使用示例](#使用示例)
- [参数说明](#参数说明)
- [常见场景](#常见场景)
- [FAQ](#faq)

---

## 功能特性

✅ **自动读取 TensorBoard 事件文件**（支持 `logs/` 和 `logs/version_X/` 结构）  
✅ **智能合并多个事件文件**（自动处理训练中断重启的情况）  
✅ **支持多种指标选择方式**（指定标签、模式匹配）  
✅ **自适应子图布局**（1-6 个指标自动排版）  
✅ **数据平滑**（移动平均滤波）  
✅ **灵活输出**（保存图片或直接显示）  
✅ **标签列表功能**（快速查看所有可用指标）  

---

## 环境要求

```bash
# 已包含在 DeMo 环境中的依赖
pip install tensorboard matplotlib numpy
```

---

## 快速开始

### 1. 列出所有可用指标

在开始绘图前，先查看日志文件中有哪些指标：

```bash
python vis_log.py --log_dir outputs/lowrank_r32/20250925-002248/logs --list
```

**输出示例：**
```
[train]
  - train/loss
  - train/loss_epoch
  - train/cls_loss
  - train/cls_loss_epoch
  - train/laplace_loss
  - train/laplace_loss_epoch
  - train/best_mode_switch
  - train/best_mode_switch_epoch
  - train/best_mode_ratio_0
  ...

[val]
  - val/minADE6
  - val/minFDE6
  - val/best_mode_switch_epoch
  ...
```

### 2. 绘制单个指标

```bash
python vis_log.py \
    --log_dir outputs/lowrank_r32/20250925-002248/logs \
    --tags train/loss_epoch
```

这会弹出一个窗口显示曲线图。

### 3. 保存到文件

```bash
python vis_log.py \
    --log_dir outputs/lowrank_r32/20250925-002248/logs \
    --tags train/loss_epoch \
    --output loss_curve.png
```

---

## 使用示例

### 示例 1: 绘制主要损失曲线

绘制总损失、分类损失和 Laplace 损失的 epoch 曲线：

```bash
python vis_log.py \
    --log_dir outputs/lowrank_r32/20250925-002248/logs \
    --tags train/loss_epoch train/cls_loss_epoch train/laplace_loss_epoch \
    --output training_losses.png \
    --smooth 5 \
    --title "Training Loss Curves"
```

**效果：**
- 3 个子图并排显示
- 每条曲线应用 5 步移动平均平滑
- 自动添加标题
- 保存为高清 PNG 图片

---

### 示例 2: 监控 best_mode 切换情况

使用模式匹配绘制所有与 best_mode 相关的指标：

```bash
python vis_log.py \
    --log_dir outputs/lowrank_r32/20250925-002248/logs \
    --pattern best_mode \
    --output best_mode_analysis.png \
    --figsize 16 10
```

**包含指标：**
- `train/best_mode_switch_epoch`
- `train/best_mode_ratio_0` ~ `train/best_mode_ratio_5`
- `train/best_mode_unique_epoch`
- `train/pi_entropy_epoch`

---

### 示例 3: 对比训练和验证指标

```bash
python vis_log.py \
    --log_dir outputs/lowrank_r32/20250925-002248/logs \
    --tags train/loss_epoch val/minADE6 val/minFDE6 \
    --output train_val_metrics.png \
    --smooth 3
```

---

### 示例 4: 查看模态概率分布

绘制每个模态被选为 best_mode 的比例：

```bash
python vis_log.py \
    --log_dir outputs/lowrank_r32/20250925-002248/logs \
    --tags train/best_mode_ratio_0 train/best_mode_ratio_1 \
           train/best_mode_ratio_2 train/best_mode_ratio_3 \
           train/best_mode_ratio_4 train/best_mode_ratio_5 \
    --output mode_distribution.png \
    --title "Best Mode Distribution Across Training"
```

---

### 示例 5: 分析模式塌缩问题

同时查看模态多样性和概率熵：

```bash
python vis_log.py \
    --log_dir outputs/lowrank_r32/20250925-002248/logs \
    --tags train/best_mode_unique_epoch train/pi_entropy_epoch \
           train/best_mode_switch_epoch \
    --output diversity_analysis.png \
    --smooth 10
```

**解读：**
- `best_mode_unique`: 应接近 6（表示所有模态都被激活）
- `pi_entropy`: 较高表示概率分布均匀，较低说明趋向 one-hot
- `best_mode_switch`: 高值表示最优模态频繁切换

---

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--log_dir` | str | `outputs/lowrank_r32/20250925-002248/logs` | TensorBoard 日志目录路径 |
| `--tags` | list | None | 要绘制的标签列表（空格分隔） |
| `--pattern` | str | None | 标签匹配模式，绘制所有包含该字符串的标签 |
| `--output` | str | None | 输出图片路径（`.png`/`.pdf`/`.svg`），不指定则显示窗口 |
| `--title` | str | None | 图表总标题 |
| `--smooth` | int | 0 | 平滑窗口大小（移动平均），0 表示不平滑 |
| `--figsize` | int int | 12 6 | 图片尺寸（宽度 高度，单位：英寸） |
| `--list` | flag | False | 列出所有可用标签（不绘图） |

---

## 常见场景

### 🔍 场景 1: 诊断损失波动

如果发现训练损失出现尖刺，可以这样排查：

```bash
# 1. 查看总损失和各分量的关系
python vis_log.py \
    --log_dir outputs/xxx/logs \
    --tags train/loss_epoch train/cls_loss_epoch train/laplace_loss_epoch \
           train/reg_loss_epoch train/others_reg_loss_epoch \
    --output loss_breakdown.png \
    --smooth 5

# 2. 检查 best_mode 切换是否剧烈
python vis_log.py \
    --log_dir outputs/xxx/logs \
    --tags train/best_mode_switch_epoch train/loss_epoch \
    --output switch_vs_loss.png
```

---

### 📊 场景 2: 论文作图

生成高质量、多子图的训练曲线：

```bash
python vis_log.py \
    --log_dir outputs/xxx/logs \
    --tags train/loss_epoch val/minADE6 val/minFDE6 val/MR \
    --output paper_figure.png \
    --smooth 10 \
    --figsize 14 8 \
    --title "Training and Validation Metrics"
```

**建议：**
- 使用 `--smooth 10` 让曲线更平滑
- 设置 `--figsize 14 8` 获得合适的宽高比
- 保存为 `.pdf` 格式以获得矢量图

---

### 🚀 场景 3: 快速验证新指标

添加新的监控指标后，快速验证是否正常记录：

```bash
# 使用 pattern 匹配所有新增指标
python vis_log.py \
    --log_dir outputs/xxx/logs \
    --pattern pi_entropy
```

---

### 🔬 场景 4: 多实验对比

如果需要对比多个实验，可以写一个简单的脚本：

```bash
#!/bin/bash
# compare_experiments.sh

experiments=(
    "outputs/lowrank_r16/xxx/logs"
    "outputs/lowrank_r32/xxx/logs"
    "outputs/lowrank_r64/xxx/logs"
)

for exp in "${experiments[@]}"; do
    name=$(basename $(dirname $(dirname $exp)))
    python vis_log.py \
        --log_dir $exp \
        --tags train/loss_epoch val/minADE6 \
        --output "${name}_metrics.png" \
        --smooth 5
done
```

---

## FAQ

### Q1: 提示 "未找到事件文件" 怎么办？

**A:** 检查以下几点：
1. 日志目录路径是否正确
2. 目录结构是否为 `logs/` 或 `logs/version_X/`
3. 目录下是否存在 `events.out.tfevents.*` 文件
4. 使用绝对路径或相对于当前工作目录的正确路径

```bash
# 查看日志目录内容
ls -la outputs/lowrank_r32/20250925-002248/logs/
ls -la outputs/lowrank_r32/20250925-002248/logs/version_0/
```

**注意**：工具会自动查找 `version_X` 子目录，并使用最新的版本。

---

### Q2: 如何绘制 step 级别的曲线而不是 epoch？

**A:** 直接使用不带 `_epoch` 后缀的标签：

```bash
python vis_log.py \
    --log_dir outputs/xxx/logs \
    --tags train/loss train/cls_loss
```

---

### Q3: 标签太多，如何筛选？

**A:** 使用三种方法：

1. **精确指定**：`--tags train/loss_epoch val/minADE6`
2. **模式匹配**：`--pattern epoch`（绘制所有 epoch 指标）
3. **先列出再选择**：`--list` 查看所有标签，复制需要的

---

### Q4: 如何调整子图布局？

**A:** 工具会自动根据指标数量调整布局：
- 1 个：1×1
- 2 个：1×2
- 3-4 个：2×2
- 5-6 个：2×3
- 更多：自动计算

如果需要自定义，可以修改 `--figsize` 参数。

---

### Q5: 平滑参数 `--smooth` 选多大合适？

**A:** 根据数据点密度选择：
- **不平滑（0）**：数据点少（<100）或需要看原始波动
- **轻度平滑（3-5）**：常规训练曲线
- **中度平滑（10-20）**：step 级别的密集数据
- **重度平滑（50+）**：论文作图，需要非常平滑的曲线

---

### Q6: 支持哪些图片格式？

**A:** 支持所有 Matplotlib 支持的格式：
- **PNG**（默认推荐）：`.png`
- **PDF**（矢量图，论文首选）：`.pdf`
- **SVG**（矢量图，可编辑）：`.svg`
- **JPG**：`.jpg` 或 `.jpeg`

```bash
# 保存为 PDF 矢量图
python vis_log.py --tags train/loss_epoch --output loss.pdf
```

---

### Q7: 如何在远程服务器上使用（没有显示器）？

**A:** 两种方法：

1. **保存到文件**（推荐）：
```bash
python vis_log.py --tags train/loss_epoch --output loss.png
```

2. **使用 X11 转发**：
```bash
ssh -X user@server
python vis_log.py --tags train/loss_epoch
```

---

## 高级技巧

### 技巧 1: 批量生成所有指标图

```bash
# get_all_metrics.sh
python vis_log.py --log_dir $1 --list | grep "train/" | grep "epoch" | \
    while read -r tag; do
        tag_clean=$(echo $tag | tr '/' '_' | sed 's/^[ \t-]*//')
        python vis_log.py --log_dir $1 --tags $tag \
            --output "metrics_${tag_clean}.png" --smooth 5
    done
```

使用方法：
```bash
bash get_all_metrics.sh outputs/lowrank_r32/20250925-002248/logs
```

---

### 技巧 2: 创建训练报告

结合其他工具生成完整的训练报告：

```bash
# training_report.sh
LOG_DIR=$1
OUTPUT_DIR="report_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# 1. 生成损失曲线
python vis_log.py --log_dir $LOG_DIR \
    --pattern loss_epoch --output $OUTPUT_DIR/losses.png

# 2. 生成验证指标
python vis_log.py --log_dir $LOG_DIR \
    --tags val/minADE6 val/minFDE6 val/MR \
    --output $OUTPUT_DIR/validation.png

# 3. 生成 best_mode 分析
python vis_log.py --log_dir $LOG_DIR \
    --pattern best_mode --output $OUTPUT_DIR/best_mode.png

echo "报告已生成到: $OUTPUT_DIR"
```

---

## 故障排查

### 问题：ImportError: No module named 'tensorboard'

**解决方法：**
```bash
conda activate DeMo
pip install tensorboard
```

---

### 问题：图片显示不全或重叠

**解决方法：**
调整 `--figsize` 参数，增大尺寸：
```bash
python vis_log.py --tags ... --figsize 16 12
```

---

### 问题：曲线太抖动

**解决方法：**
增大平滑窗口：
```bash
python vis_log.py --tags train/loss_epoch --smooth 20
```

---

### 问题：多个事件文件如何处理？

**A:** 工具会自动处理多个事件文件：

1. **自动检测**：在 `logs/version_X/` 目录下可能有多个 `events.out.tfevents.*` 文件
   - 第一个文件：主要的训练日志
   - 后续文件：训练中断重启或其他写入流

2. **智能合并**：
   - 读取所有事件文件
   - 按 `step` 排序合并数据
   - 自动去重（相同 step 保留最后一个值）

3. **无需手动操作**：工具会自动处理，确保数据完整性

**查看合并情况：**
```bash
# 查看有多少个事件文件
ls -la outputs/xxx/logs/version_0/events.out.tfevents.*

# 工具会显示读取的文件数量
python vis_log.py --log_dir outputs/xxx/logs --list
# 输出: 找到 2 个事件文件
#       读取事件文件: events.out.tfevents.xxx.0
#       读取事件文件: events.out.tfevents.xxx.1
```

---

## 贡献与反馈

如有问题或建议，欢迎提 Issue 或 Pull Request！

---

## 更新日志

### v1.0 (2025-10-01)
- ✅ 初始版本
- ✅ 支持 TensorBoard 事件文件读取
- ✅ 自动布局和平滑功能
- ✅ 模式匹配和标签列表
- ✅ 完整的命令行接口

---

**最后更新**: 2025-10-01  
**维护者**: DeMo Project Team
