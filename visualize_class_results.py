# import argparse
# import pickle
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import os

# def visualize_results(results_file, save_dir):
#     """
#     加载评估结果并为 minADE6, minFDE6, 和 b-minFDE6 指标绘制箱型图。

#     参数:
#     results_file (str): class_wise_detailed.pkl 文件的路径。
#     save_dir (str): 保存生成图表的目录。
#     """
#     # 确保保存目录存在
#     os.makedirs(save_dir, exist_ok=True)

#     # 加载数据
#     with open(results_file, 'rb') as f:
#         data = pickle.load(f)

#     # 定义类别名称映射
#     class_map = {0: 'Vehicle', 1: 'Pedestrian', 2: 'Cyclist'}
    
#     # 将数据转换为 pandas DataFrame 以便绘图
#     plot_data = []
#     for class_id, metrics in data.items():
#         class_name = class_map.get(class_id, f'Unknown_{class_id}')
#         for metric_name, values in metrics.items():
#             for value in values:
#                 plot_data.append({
#                     'Class': class_name,
#                     'Metric': metric_name,
#                     'Value': value
#                 })
    
#     df = pd.DataFrame(plot_data)
    
#     # --- 绘制所有指标的箱型图 ---
#     metrics_to_plot = ['minADE6', 'minFDE6', 'b-minFDE6']
#     for metric in metrics_to_plot:
#         plt.figure(figsize=(10, 7))
#         sns.boxplot(x='Class', y='Value', data=df[df['Metric'] == metric], order=['Vehicle', 'Pedestrian', 'Cyclist'])
        
#         # 设置图表的标题和标签
#         title = f'{metric} Distribution by Agent Class'
#         ylabel = f'{metric} Value'
#         plt.title(title, fontsize=16)
#         plt.xlabel('Agent Class', fontsize=12)
#         plt.ylabel(ylabel, fontsize=12)
#         plt.grid(axis='y', linestyle='--', alpha=0.7)
        
#         # 保存图像
#         save_path = os.path.join(save_dir, f'{metric}_boxplot.png')
#         plt.savefig(save_path)
#         print(f"{metric} 箱型图已保存至: {save_path}")
#         plt.close()


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="为不同智能体类别的评估结果绘制箱型图。")
#     parser.add_argument('--results_file', type=str, required=True,
#                         help='包含分类别评估结果的 .pkl 文件路径。')
#     parser.add_argument('--save_dir', type=str, default='./plots',
#                         help='用于保存生成图表的目录。')
    
#     args = parser.parse_args()
#     visualize_results(args.results_file, args.save_dir)

import argparse
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def visualize_results(results_file, save_dir):
    """
    加载评估结果，并为每个关键指标绘制并排的箱型图和小提琴图。

    参数:
    results_file (str): class_wise_detailed.pkl 文件的路径。
    save_dir (str): 保存生成图表的目录。
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 加载数据
    with open(results_file, 'rb') as f:
        data = pickle.load(f)

    # 定义类别名称映射
    class_map = {0: 'Vehicle', 1: 'Pedestrian', 2: 'Cyclist'}
    
    # 将数据转换为 pandas DataFrame 以便绘图
    plot_data = []
    for class_id, metrics in data.items():
        class_name = class_map.get(class_id, f'Unknown_{class_id}')
        for metric_name, values in metrics.items():
            for value in values:
                plot_data.append({
                    'Class': class_name,
                    'Metric': metric_name,
                    'Value': value
                })
    
    df = pd.DataFrame(plot_data)
    
    # --- 为所有指标绘制组合图表 ---
    # 检查 pkl 文件中存在哪些指标并绘制它们
    metrics_to_plot = df['Metric'].unique()
    
    for metric in metrics_to_plot:
        # 创建一个包含两个子图的图表
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
        
        # --- 子图1: 箱型图 (Box Plot) ---
        sns.boxplot(ax=axes[0], x='Class', y='Value', data=df[df['Metric'] == metric], order=['Vehicle', 'Pedestrian', 'Cyclist'])
        axes[0].set_title(f'Box Plot of {metric}', fontsize=14)
        axes[0].set_xlabel('Agent Class', fontsize=12)
        axes[0].set_ylabel(f'{metric} Value', fontsize=12)
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)

        # --- 子图2: 小提琴图 (Violin Plot) ---
        sns.violinplot(ax=axes[1], x='Class', y='Value', data=df[df['Metric'] == metric], order=['Vehicle', 'Pedestrian', 'Cyclist'])
        axes[1].set_title(f'Violin Plot of {metric}', fontsize=14)
        axes[1].set_xlabel('Agent Class', fontsize=12)
        axes[1].set_ylabel('')  # 共享Y轴，所以不重复显示标签
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)

        # 设置整个图表的总标题
        fig.suptitle(f'{metric} Distribution by Agent Class', fontsize=18)
        
        # 调整布局以避免标题重叠
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 保存图像
        save_path = os.path.join(save_dir, f'combined_{metric}_plot.png')
        plt.savefig(save_path)
        print(f"{metric} 组合图表已保存至: {save_path}")
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="为不同智能体类别的评估结果绘制组合图表（箱型图和小提琴图）。")
    parser.add_argument('--results_file', type=str, required=True,
                        help='包含分类别评估结果的 .pkl 文件路径。')
    parser.add_argument('--save_dir', type=str, default='./plots',
                        help='用于保存生成图表的目录。')
    
    args = parser.parse_args()
    visualize_results(args.results_file, args.save_dir)