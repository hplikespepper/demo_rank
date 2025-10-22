"""
分析验证集中各类别 agent 的统计信息

该脚本用于统计 data/DeMo_processed/val 中：
1. 各类别 agent 的总数量
2. 每个文件中各类别 agent 的分布
3. focal agent 和 scored agent 的类别分布
"""

import os
from pathlib import Path
import torch
from tqdm import tqdm
from collections import defaultdict
import numpy as np


def inspect_single_file(file_path):
    """检查单个文件的数据结构"""
    print(f"检查文件: {file_path.name}")
    data = torch.load(file_path)
    print(f"\n数据字典的键: {data.keys()}")
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, (list, tuple)):
            print(f"  {key}: type={type(value)}, len={len(value)}")
        else:
            print(f"  {key}: type={type(value)}, value={value}")
    print("\n")
    return data


def analyze_validation_dataset(data_root='data/DeMo_processed/val'):
    """
    分析验证集中的 agent 类别分布
    
    Args:
        data_root: 验证集数据路径
    """
    data_folder = Path(data_root)
    file_list = sorted(list(data_folder.glob('*.pt')))
    
    print(f"数据路径: {data_root}")
    print(f"文件总数: {len(file_list)}\n")
    
    # 先检查第一个文件的结构
    if len(file_list) > 0:
        print("="*70)
        print("首先检查数据结构...")
        print("="*70)
        sample_data = inspect_single_file(file_list[0])
        print("\n按任意键继续完整分析...")
        # input()  # 暂停以便查看结构
    
    # 统计变量
    total_agents = 0
    type_counts = defaultdict(int)  # 各类别总数
    focal_type_counts = defaultdict(int)  # focal agent 类别分布
    scored_type_counts = defaultdict(int)  # scored agent 类别分布
    
    # 详细统计
    file_details = []
    
    # 类别名称映射
    type_names = {
        0: 'Vehicle (车辆)',
        1: 'Pedestrian (行人)',
        2: 'Cyclist (骑行者)'
    }
    
    print("开始分析数据...")
    for file_path in tqdm(file_list, desc="Processing files"):
        try:
            data = torch.load(file_path)
            
            # 检查 x_attr 的维度并相应处理
            x_attr = data['x_attr']
            
            # 判断 x_attr 是 2D 还是 3D
            if x_attr.dim() == 2:
                # 2D: [num_agents, features], 类型在第3列 (index=2)
                agent_types = x_attr[:, 2].long()
            elif x_attr.dim() == 3:
                # 3D: [num_agents, num_steps, features], 取第一个时间步
                agent_types = x_attr[:, 0, 2].long()
            else:
                print(f"\n警告：文件 {file_path.name} 的 x_attr 维度异常: {x_attr.shape}")
                continue
            
            num_agents = len(agent_types)
            
            # 统计各类别数量
            total_agents += num_agents
            file_type_count = defaultdict(int)
            
            for agent_type in agent_types:
                agent_type = agent_type.item()
                type_counts[agent_type] += 1
                file_type_count[agent_type] += 1
            
            # 统计 focal agent 类型
            focal_idx = data['focal_idx']
            focal_type = agent_types[focal_idx].item()
            focal_type_counts[focal_type] += 1
            
            # 统计 scored agents 类型
            if 'scored_idx' in data and len(data['scored_idx']) > 0:
                for scored_idx in data['scored_idx']:
                    scored_type = agent_types[scored_idx].item()
                    scored_type_counts[scored_type] += 1
            
            # 记录文件详细信息
            file_details.append({
                'file': file_path.name,
                'total_agents': num_agents,
                'focal_type': focal_type,
                'type_distribution': dict(file_type_count),
                'num_scored': len(data.get('scored_idx', []))
            })
            
        except Exception as e:
            print(f"\n错误：无法处理文件 {file_path.name}: {e}")
            continue
    
    # 打印统计结果
    print("\n" + "="*70)
    print("验证集统计结果")
    print("="*70)
    
    print(f"\n总文件数: {len(file_list)}")
    print(f"总 agent 数: {total_agents}")
    
    print("\n【各类别 Agent 总体分布】")
    print("-" * 70)
    for agent_type in sorted(type_counts.keys()):
        count = type_counts[agent_type]
        percentage = (count / total_agents * 100) if total_agents > 0 else 0
        print(f"类别 {agent_type} - {type_names.get(agent_type, 'Unknown'):20s}: "
              f"{count:6d} ({percentage:5.2f}%)")
    
    print("\n【Focal Agent 类别分布】")
    print("-" * 70)
    print(f"Focal agent 总数: {len(file_list)} (每个文件一个 focal agent)")
    for agent_type in sorted(focal_type_counts.keys()):
        count = focal_type_counts[agent_type]
        percentage = (count / len(file_list) * 100) if len(file_list) > 0 else 0
        print(f"类别 {agent_type} - {type_names.get(agent_type, 'Unknown'):20s}: "
              f"{count:6d} ({percentage:5.2f}%)")
    
    print("\n【Scored Agent 类别分布】")
    print("-" * 70)
    total_scored = sum(scored_type_counts.values())
    print(f"Scored agent 总数: {total_scored}")
    for agent_type in sorted(scored_type_counts.keys()):
        count = scored_type_counts[agent_type]
        percentage = (count / total_scored * 100) if total_scored > 0 else 0
        print(f"类别 {agent_type} - {type_names.get(agent_type, 'Unknown'):20s}: "
              f"{count:6d} ({percentage:5.2f}%)")
    
    # 统计每个文件中 agent 数量的分布
    agents_per_file = [detail['total_agents'] for detail in file_details]
    print("\n【每个文件中 Agent 数量统计】")
    print("-" * 70)
    if len(agents_per_file) > 0:
        print(f"最小值: {min(agents_per_file)}")
        print(f"最大值: {max(agents_per_file)}")
        print(f"平均值: {np.mean(agents_per_file):.2f}")
        print(f"中位数: {np.median(agents_per_file):.2f}")
        print(f"标准差: {np.std(agents_per_file):.2f}")
    else:
        print("没有成功处理的文件")
    
    # 关键信息：用于预测的 agent 数量
    print("\n" + "="*70)
    print("【关键：用于评估的 Agent 数量】")
    print("="*70)
    print("\n根据训练模式的不同，用于评估的 agent 数量会有所不同：")
    print("\n1. only_focal 模式（单 agent 预测）:")
    print(f"   - 每个文件预测 1 个 focal agent")
    print(f"   - 验证集总预测数: {len(file_list)}")
    print("   - 各类别预测数量:")
    for agent_type in sorted(focal_type_counts.keys()):
        count = focal_type_counts[agent_type]
        print(f"     * 类别 {agent_type} - {type_names.get(agent_type, 'Unknown'):20s}: {count:6d}")
    
    print("\n2. focal_and_scored 模式（多 agent 预测）:")
    total_predictions = len(file_list) + total_scored
    print(f"   - 每个文件预测 focal + scored agents")
    print(f"   - 验证集总预测数: {total_predictions} (focal: {len(file_list)}, scored: {total_scored})")
    print("   - 各类别预测数量:")
    combined_counts = defaultdict(int)
    for agent_type in focal_type_counts.keys():
        combined_counts[agent_type] += focal_type_counts[agent_type]
    for agent_type in scored_type_counts.keys():
        combined_counts[agent_type] += scored_type_counts[agent_type]
    for agent_type in sorted(combined_counts.keys()):
        count = combined_counts[agent_type]
        percentage = (count / total_predictions * 100) if total_predictions > 0 else 0
        print(f"     * 类别 {agent_type} - {type_names.get(agent_type, 'Unknown'):20s}: "
              f"{count:6d} ({percentage:5.2f}%)")
    
    # 保存详细结果到文件
    output_file = 'validation_dataset_analysis.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("验证集详细分析结果\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"数据路径: {data_root}\n")
        f.write(f"总文件数: {len(file_list)}\n")
        f.write(f"总 agent 数: {total_agents}\n\n")
        
        f.write("【各类别 Agent 总体分布】\n")
        f.write("-" * 70 + "\n")
        for agent_type in sorted(type_counts.keys()):
            count = type_counts[agent_type]
            percentage = (count / total_agents * 100) if total_agents > 0 else 0
            f.write(f"类别 {agent_type} - {type_names.get(agent_type, 'Unknown'):20s}: "
                   f"{count:6d} ({percentage:5.2f}%)\n")
        
        f.write("\n【Focal Agent 类别分布】\n")
        f.write("-" * 70 + "\n")
        for agent_type in sorted(focal_type_counts.keys()):
            count = focal_type_counts[agent_type]
            percentage = (count / len(file_list) * 100) if len(file_list) > 0 else 0
            f.write(f"类别 {agent_type} - {type_names.get(agent_type, 'Unknown'):20s}: "
                   f"{count:6d} ({percentage:5.2f}%)\n")
        
        f.write("\n【用于预测的 Agent 数量（only_focal 模式）】\n")
        f.write("-" * 70 + "\n")
        for agent_type in sorted(focal_type_counts.keys()):
            f.write(f"类别 {agent_type} - {type_names.get(agent_type, 'Unknown'):20s}: "
                   f"{focal_type_counts[agent_type]:6d}\n")
        
        f.write("\n\n【前20个文件的详细信息示例】\n")
        f.write("-" * 70 + "\n")
        for i, detail in enumerate(file_details[:20]):
            f.write(f"\n文件 {i+1}: {detail['file']}\n")
            f.write(f"  总 agents: {detail['total_agents']}\n")
            f.write(f"  Focal agent 类型: {detail['focal_type']} - {type_names.get(detail['focal_type'], 'Unknown')}\n")
            f.write(f"  Scored agents 数量: {detail['num_scored']}\n")
            f.write(f"  类别分布: ")
            for t, c in sorted(detail['type_distribution'].items()):
                f.write(f"{type_names.get(t, 'Unknown')}: {c}  ")
            f.write("\n")
    
    print(f"\n详细结果已保存到: {output_file}")
    print("="*70)
    
    return {
        'total_files': len(file_list),
        'total_agents': total_agents,
        'type_counts': dict(type_counts),
        'focal_type_counts': dict(focal_type_counts),
        'scored_type_counts': dict(scored_type_counts),
        'file_details': file_details
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='分析验证集中的 agent 类别分布')
    parser.add_argument('--data_root', type=str, default='data/DeMo_processed/val',
                        help='验证集数据路径 (默认: data/DeMo_processed/val)')
    
    args = parser.parse_args()
    
    # 检查路径是否存在
    if not os.path.exists(args.data_root):
        print(f"错误：数据路径不存在: {args.data_root}")
        print("请检查路径是否正确")
        exit(1)
    
    # 执行分析
    results = analyze_validation_dataset(args.data_root)
