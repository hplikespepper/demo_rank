#!/usr/bin/env python3
"""
分析批量评估结果脚本
读取original_seed_result目录下各个seed的summary_metrics.txt文件
计算各指标的平均值、标准差等统计信息
"""

import os
import glob
import statistics
from collections import defaultdict


def find_summary_metrics_files(base_dir="outputs/original_seed_result", target_seeds=[42, 43, 46, 47, 50]):    # lowrank_seed_result, original_seed_result
    """查找指定seed目录下的summary_metrics.txt文件"""
    summary_files = {}
    
    if not os.path.exists(base_dir):
        print(f"错误: 目录 {base_dir} 不存在")
        return summary_files
    
    print(f"指定要统计的seeds: {target_seeds}")
    
    # 只查找指定的seed目录
    for seed_num in target_seeds:
        seed_dir = os.path.join(base_dir, f"seed{seed_num}")
        
        if not os.path.isdir(seed_dir):
            print(f"Warning: 目录不存在 {seed_dir}")
            continue
        
        # 查找timestamp目录
        timestamp_dirs = glob.glob(os.path.join(seed_dir, "20*"))
        timestamp_dirs = [d for d in timestamp_dirs if os.path.isdir(d)]
        
        if not timestamp_dirs:
            print(f"Warning: 在 {seed_dir} 中没有找到timestamp目录")
            continue
        
        # 取最新的timestamp目录
        latest_timestamp = sorted(timestamp_dirs)[-1]
        summary_file = os.path.join(latest_timestamp, "summary_metrics.txt")
        
        if os.path.exists(summary_file):
            summary_files[seed_num] = summary_file
            print(f"找到 seed{seed_num} 的结果文件: {summary_file}")
        else:
            print(f"Warning: 文件不存在 {summary_file}")
    
    return summary_files


def parse_summary_metrics(summary_path):
    """解析summary_metrics.txt文件"""
    metrics = {}
    
    try:
        with open(summary_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line and ':' in line and not line.startswith('---'):
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                try:
                    metrics[key] = float(value)
                except ValueError:
                    print(f"Warning: 无法转换 '{value}' 为数字，指标 '{key}'")
    
    except Exception as e:
        print(f"错误: 读取文件 {summary_path} 失败: {e}")
    
    return metrics


def calculate_statistics(all_metrics):
    """计算所有指标的统计信息"""
    stats = {}
    
    # 获取所有指标名称
    all_keys = set()
    for metrics in all_metrics.values():
        all_keys.update(metrics.keys())
    
    for key in sorted(all_keys):
        values = []
        for seed, metrics in all_metrics.items():
            if key in metrics:
                values.append(metrics[key])
        
        if values:
            stats[key] = {
                'mean': statistics.mean(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                'min': min(values),
                'max': max(values),
                'count': len(values),
                'values': values
            }
    
    return stats


def main():
    """主函数"""
    print("开始分析批量评估结果...")
    
    # 指定要分析的seeds
    target_seeds = [43, 44, 47, 49, 50]
    print(f"指定分析的seeds: {target_seeds}")
    
    # 查找指定seeds的结果文件
    summary_files = find_summary_metrics_files(target_seeds=target_seeds)
    
    if not summary_files:
        print("错误: 没有找到任何summary_metrics.txt文件")
        print("请确保已经运行了 ./repeat_result.sh 并且评估成功完成")
        return
    
    print(f"\n找到 {len(summary_files)} 个seed的结果文件")
    print(f"Seeds: {sorted(summary_files.keys())}")
    
    # 解析所有metrics
    all_metrics = {}
    for seed, file_path in summary_files.items():
        metrics = parse_summary_metrics(file_path)
        if metrics:
            all_metrics[seed] = metrics
            print(f"成功解析 seed{seed} 的指标: {len(metrics)} 个")
        else:
            print(f"Warning: seed{seed} 没有有效指标")
    
    if not all_metrics:
        print("错误: 没有解析到任何有效的指标数据")
        return
    
    # 计算统计信息
    stats = calculate_statistics(all_metrics)
    
    print(f"\n{'='*60}")
    print("统计结果汇总")
    print(f"{'='*60}")
    
    # 创建输出目录
    output_dir = "outputs/original_seed_result"
    # output_dir = "outputs/lowrank_seed_result"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存统计结果到文件
    stats_file = os.path.join(output_dir, "evaluation_statistics.txt")
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=== 指定Seeds模型评估统计结果 ===\n")
        f.write(f"评估的seeds: {sorted(all_metrics.keys())}\n")
        f.write(f"成功评估数量: {len(all_metrics)}\n\n")
        
        for key, stat in sorted(stats.items()):
            f.write(f"{key}:\n")
            f.write(f"  平均值: {stat['mean']:.8f}\n")
            f.write(f"  标准差: {stat['std']:.8f}\n")
            f.write(f"  最小值: {stat['min']:.8f}\n")
            f.write(f"  最大值: {stat['max']:.8f}\n")
            f.write(f"  样本数: {stat['count']}\n")
            f.write(f"  原始值: {[f'{v:.6f}' for v in stat['values']]}\n\n")
    
    print(f"统计结果已保存到: {stats_file}")
    
    # 在控制台输出主要指标（后面12个关键指标）
    key_metrics = [
        'val_MR', 'val_minADE1', 'val_minADE6', 'val_minFDE1', 'val_minFDE6', 'val_b-minFDE6',
        'val_new_MR', 'val_new_minADE1', 'val_new_minADE6', 'val_new_minFDE1', 'val_new_minFDE6', 'val_new_b-minFDE6'
    ]
    
    print(f"\n{'='*60}")
    print("主要指标统计结果（指定seeds的关键指标）")
    print(f"{'='*60}")
    print(f"{'指标名称':<25} {'平均值':<12} {'标准差':<12} {'样本数':<8}")
    print("-" * 60)
    
    for key in key_metrics:
        if key in stats:
            stat = stats[key]
            print(f"{key:<25} {stat['mean']:<12.6f} {stat['std']:<12.6f} {stat['count']:<8}")
    
    # 保存主要指标的专门统计文件
    key_stats_file = os.path.join(output_dir, "key_metrics_statistics.txt")
    with open(key_stats_file, 'w', encoding='utf-8') as f:
        f.write("=== 主要指标统计结果（指定seeds的关键指标） ===\n")
        f.write(f"评估的seeds: {sorted(all_metrics.keys())}\n")
        f.write(f"成功评估数量: {len(all_metrics)}\n\n")
        
        for key in key_metrics:
            if key in stats:
                stat = stats[key]
                f.write(f"{key}:\n")
                f.write(f"  平均值: {stat['mean']:.8f}\n")
                f.write(f"  标准差: {stat['std']:.8f}\n")
                f.write(f"  最小值: {stat['min']:.8f}\n")
                f.write(f"  最大值: {stat['max']:.8f}\n")
                f.write(f"  样本数: {stat['count']}\n")
                f.write(f"  原始值: {[f'{v:.6f}' for v in stat['values']]}\n\n")
    
    print(f"\n主要指标统计结果已保存到: {key_stats_file}")
    
    # 保存详细结果
    detailed_file = os.path.join(output_dir, "detailed_results.txt")
    with open(detailed_file, 'w', encoding='utf-8') as f:
        f.write("=== 各个seed的详细结果 ===\n\n")
        for seed in sorted(all_metrics.keys()):
            f.write(f"Seed {seed}:\n")
            metrics = all_metrics[seed]
            for key in sorted(metrics.keys()):
                f.write(f"  {key}: {metrics[key]:.8f}\n")
            f.write("\n")
    
    print(f"\n详细结果已保存到: {detailed_file}")
    
    # 保存CSV格式便于进一步分析
    csv_file = os.path.join(output_dir, "results_summary.csv")
    with open(csv_file, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write("metric,mean,std,min,max,count\n")
        for key, stat in sorted(stats.items()):
            f.write(f"{key},{stat['mean']:.8f},{stat['std']:.8f},{stat['min']:.8f},{stat['max']:.8f},{stat['count']}\n")
    
    print(f"CSV格式结果已保存到: {csv_file}")
    
    print(f"\n{'='*60}")
    print("分析完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()