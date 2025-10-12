#!/usr/bin/env python
"""
可视化 TensorBoard 日志文件中的指标
支持从指定目录读取事件文件并绘制指标曲线
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_tensorboard_logs(log_dir, tags=None):
    """
    从 TensorBoard 日志目录读取指标数据
    
    Args:
        log_dir: 日志目录路径
        tags: 要读取的标签列表，None表示读取所有
        
    Returns:
        dict: {tag: {'steps': [...], 'values': [...]}}
    """
    log_path = Path(log_dir)
    
    # 查找事件文件，支持 version_X 子目录结构
    event_files = list(log_path.glob("events.out.tfevents.*"))
    
    # 如果当前目录没有事件文件，查找 version_X 子目录
    if not event_files:
        version_dirs = sorted(log_path.glob("version_*"))
        if version_dirs:
            # 使用最新的 version 目录
            latest_version = version_dirs[-1]
            event_files = list(latest_version.glob("events.out.tfevents.*"))
            print(f"使用日志目录: {latest_version}")
        
    if not event_files:
        raise ValueError(f"未找到事件文件: {log_dir}\n请检查目录结构是否为 logs/ 或 logs/version_X/")
    
    print(f"找到 {len(event_files)} 个事件文件")
    
    # 读取所有事件文件并合并数据
    all_data = defaultdict(lambda: {'steps': [], 'values': [], 'wall_times': []})
    all_available_tags = set()
    
    for event_file in sorted(event_files):
        print(f"读取事件文件: {event_file.name}")
        
        # 加载事件数据
        ea = EventAccumulator(str(event_file))
        ea.Reload()
        
        # 获取所有可用的标签
        file_tags = ea.Tags()['scalars']
        all_available_tags.update(file_tags)
        
        # 确定要读取的标签
        if tags is None:
            read_tags = file_tags
        else:
            read_tags = [t for t in tags if t in file_tags]
        
        # 读取数据并累积
        for tag in read_tags:
            events = ea.Scalars(tag)
            all_data[tag]['steps'].extend([e.step for e in events])
            all_data[tag]['values'].extend([e.value for e in events])
            all_data[tag]['wall_times'].extend([e.wall_time for e in events])
    
    available_tags = sorted(list(all_available_tags))
    print(f"可用的标签 ({len(available_tags)} 个)")
    
    # 确定要读取的标签
    if tags is None:
        tags = available_tags
    else:
        # 验证标签是否存在
        invalid_tags = [t for t in tags if t not in available_tags]
        if invalid_tags:
            print(f"警告: 以下标签不存在: {invalid_tags}")
        tags = [t for t in tags if t in available_tags]
    
    # 转换为 numpy 数组并排序（按 step）
    data = {}
    for tag in tags:
        if tag in all_data:
            steps = np.array(all_data[tag]['steps'])
            values = np.array(all_data[tag]['values'])
            wall_times = np.array(all_data[tag]['wall_times'])
            
            # 按 step 排序
            sort_idx = np.argsort(steps)
            steps = steps[sort_idx]
            values = values[sort_idx]
            wall_times = wall_times[sort_idx]
            
            # 去重（如果有重复的 step，保留最后一个）
            unique_steps, unique_idx = np.unique(steps, return_index=True)
            if len(unique_steps) < len(steps):
                # 有重复，需要保留每个 step 的最后一个值
                step_to_idx = {}
                for i, step in enumerate(steps):
                    step_to_idx[step] = i
                unique_idx = [step_to_idx[s] for s in unique_steps]
                steps = steps[unique_idx]
                values = values[unique_idx]
                wall_times = wall_times[unique_idx]
            
            data[tag] = {
                'steps': steps,
                'values': values,
                'wall_times': wall_times
            }
            print(f"已读取 {tag}: {len(steps)} 个数据点")
    
    return data, available_tags


def plot_metrics(data, output_path=None, title=None, smooth=0, figsize=(12, 6)):
    """
    绘制指标曲线
    
    Args:
        data: 指标数据字典
        output_path: 输出图片路径，None表示显示
        title: 图表标题
        smooth: 平滑窗口大小，0表示不平滑
        figsize: 图片尺寸
    """
    num_metrics = len(data)
    if num_metrics == 0:
        print("没有数据可绘制")
        return
    
    # 自动调整子图布局
    if num_metrics == 1:
        nrows, ncols = 1, 1
    elif num_metrics == 2:
        nrows, ncols = 1, 2
    elif num_metrics <= 4:
        nrows, ncols = 2, 2
    elif num_metrics <= 6:
        nrows, ncols = 2, 3
    else:
        ncols = 3
        nrows = (num_metrics + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if num_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if nrows * ncols > 1 else [axes]
    
    for idx, (tag, metric_data) in enumerate(data.items()):
        ax = axes[idx]
        steps = metric_data['steps']
        values = metric_data['values']
        
        # 平滑处理
        if smooth > 1 and len(values) >= smooth:
            smoothed = np.convolve(values, np.ones(smooth)/smooth, mode='valid')
            smoothed_steps = steps[smooth-1:]
            ax.plot(smoothed_steps, smoothed, label='Smoothed', linewidth=2, alpha=0.8)
            ax.plot(steps, values, label='Raw', alpha=0.3, linewidth=1)
            ax.legend()
        else:
            ax.plot(steps, values, linewidth=2)
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.set_title(tag, fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for idx in range(num_metrics, len(axes)):
        axes[idx].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()


def list_available_tags(log_dir):
    """列出日志目录中所有可用的标签"""
    _, available_tags = read_tensorboard_logs(log_dir, tags=[])
    print("\n所有可用的标签:")
    print("=" * 60)
    
    # 按前缀分组
    grouped = defaultdict(list)
    for tag in sorted(available_tags):
        prefix = tag.split('/')[0] if '/' in tag else 'other'
        grouped[prefix].append(tag)
    
    for prefix, tags in sorted(grouped.items()):
        print(f"\n[{prefix}]")
        for tag in tags:
            print(f"  - {tag}")


def main():
    parser = argparse.ArgumentParser(
        description="可视化 TensorBoard 日志文件中的指标",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 列出所有可用的标签
  python vis_log.py --log_dir outputs/lowrank_r32/20250925-002248/logs --list
  
  # 绘制单个指标
  python vis_log.py --log_dir outputs/lowrank_r32/20250925-002248/logs --tags train/loss_epoch
  
  # 绘制多个指标
  python vis_log.py --log_dir outputs/lowrank_r32/20250925-002248/logs --tags train/loss_epoch train/cls_loss_epoch train/laplace_loss_epoch
  
  # 绘制所有 best_mode 相关指标
  python vis_log.py --log_dir outputs/lowrank_r32/20250925-002248/logs --pattern best_mode
  
  # 保存到文件并使用平滑
  python vis_log.py --log_dir outputs/lowrank_r32/20250925-002248/logs --tags train/loss_epoch --output loss_curve.png --smooth 10
        """
    )
    
    parser.add_argument(
        '--log_dir',
        type=str,
        default='outputs/lowrank_r32/20250925-002248/logs',
        help='TensorBoard 日志目录路径'
    )
    
    parser.add_argument(
        '--tags',
        type=str,
        nargs='+',
        default=None,
        help='要绘制的标签名称列表'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default=None,
        help='标签匹配模式（包含该字符串的标签都会被绘制）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出图片路径，不指定则直接显示'
    )
    
    parser.add_argument(
        '--title',
        type=str,
        default=None,
        help='图表标题'
    )
    
    parser.add_argument(
        '--smooth',
        type=int,
        default=0,
        help='平滑窗口大小，0表示不平滑'
    )
    
    parser.add_argument(
        '--figsize',
        type=int,
        nargs=2,
        default=[12, 6],
        help='图片尺寸 (宽度 高度)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='列出所有可用的标签'
    )
    
    args = parser.parse_args()
    
    # 检查日志目录是否存在
    if not os.path.exists(args.log_dir):
        print(f"错误: 日志目录不存在: {args.log_dir}")
        return
    
    # 列出标签
    if args.list:
        list_available_tags(args.log_dir)
        return
    
    # 确定要读取的标签
    tags = args.tags
    if args.pattern:
        # 先读取所有标签，然后过滤
        _, available_tags = read_tensorboard_logs(args.log_dir, tags=[])
        tags = [t for t in available_tags if args.pattern in t]
        if not tags:
            print(f"警告: 没有找到匹配 '{args.pattern}' 的标签")
            return
        print(f"找到 {len(tags)} 个匹配的标签:")
        for tag in tags:
            print(f"  - {tag}")
    
    # 读取数据
    data, _ = read_tensorboard_logs(args.log_dir, tags=tags)
    
    if not data:
        print("没有数据可绘制")
        return
    
    # 绘制图表
    plot_metrics(
        data,
        output_path=args.output,
        title=args.title,
        smooth=args.smooth,
        figsize=tuple(args.figsize)
    )


if __name__ == '__main__':
    main()
