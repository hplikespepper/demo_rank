# 此程序用于评估轨迹预测模型并保存每个实例的详细指标。
# 它继承自基础的 Trainer 类，在验证步骤中保存每个样本的所有信息和指标。
# 最终结果会被保存为一个 pickle 文件，包含场景ID、agent类型、各项指标等，
# 便于后续针对表现好/差的实例进行可视化分析。

import os
import sys
import hydra
import pickle
import torch
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from omegaconf import OmegaConf

# 从现有代码库中导入必要的模块
from src.model.trainer_forecast import Trainer
from src.metrics import minADE, minFDE, brier_minFDE

class DetailedEvalTrainer(Trainer):
    """
    一个专门用于保存每个实例详细评估结果的Trainer。
    保存的信息包括:
    - scenario_id: 场景ID
    - agent_type: agent类型 (0: 车辆, 1: 行人, 2: 骑行者)
    - minADE1, minADE6, minFDE1, minFDE6, b-minFDE6: 各项指标值
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 初始化用于存储每个实例详细结果的列表
        self.instance_results = []

    def validation_step(self, data, batch_idx):
        if isinstance(data, list):
            data = data[-1]
            
        out = self(data)
        
        # 关键修复：只使用 refine 后的结果（如果存在）
        if out.get('new_y_hat') is not None:
            # 创建新的输出字典，只包含 refine 后的预测
            refined_out = {
                'y_hat': out['new_y_hat'],
                'pi': out['new_pi']
            }
        else:
            # 如果没有 refine 结果，使用原始预测
            refined_out = {
                'y_hat': out['y_hat'],
                'pi': out['pi']
            }

        # 数据结构说明:
        # - data['target']: [batch_size, 1, 60, 2] (每个样本只有1个focal agent)
        # - data['x_attr']: [batch_size, num_agents, 3] (包含场景中所有agents的属性)
        # - data['scenario_id']: [batch_size] 或 list - 场景ID
        # - refined_out['y_hat']: [batch_size, num_modes, 60, 2]
        # - refined_out['pi']: [batch_size, num_modes]
        
        y_true = data['target'][:, 0]  # [batch_size, 60, 2]
        batch_size = y_true.shape[0]
        
        # 获取每个batch样本的focal agent类型
        agent_types = data['x_attr'][:, 0, 2]  # [batch_size] - 每个样本的第0个agent类型
        
        # 获取场景ID
        scenario_ids = data.get('scenario_id', None)
        
        device = y_true.device

        # 对batch中的每个样本（每个样本对应一个focal agent）
        for i in range(batch_size):
            agent_type = agent_types[i].item()
            
            # 获取场景ID
            if scenario_ids is not None:
                if isinstance(scenario_ids, list):
                    scenario_id = scenario_ids[i]
                else:
                    scenario_id = scenario_ids[i].item() if hasattr(scenario_ids[i], 'item') else scenario_ids[i]
            else:
                scenario_id = f"batch{batch_idx}_sample{i}"

            # 提取单个样本的预测
            single_out = {
                'y_hat': refined_out['y_hat'][i:i+1],  # [1, num_modes, 60, 2]
                'pi': refined_out['pi'][i:i+1]          # [1, num_modes]
            }
            single_y_true = y_true[i:i+1]  # [1, 60, 2]

            # 计算此样本的核心指标值
            ade1 = minADE(k=1).to(device)(single_out, single_y_true).item()
            ade6 = minADE(k=6).to(device)(single_out, single_y_true).item()
            fde1 = minFDE(k=1).to(device)(single_out, single_y_true).item()
            fde6 = minFDE(k=6).to(device)(single_out, single_y_true).item()
            b_fde6 = brier_minFDE(k=6).to(device)(single_out, single_y_true).item()

            # 保存该实例的所有信息
            instance_info = {
                'scenario_id': scenario_id,
                'agent_type': int(agent_type),
                'minADE1': ade1,
                'minADE6': ade6,
                'minFDE1': fde1,
                'minFDE6': fde6,
                'b-minFDE6': b_fde6,
                'batch_idx': batch_idx,
                'sample_idx': i
            }
            
            self.instance_results.append(instance_info)

    def on_validation_epoch_end(self):
        # 保存所有实例的详细结果
        output_dir = HydraConfig.get().runtime.output_dir
        save_path = os.path.join(output_dir, "instance_detailed_results.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(self.instance_results, f)
        print(f"实例详细评估结果已保存至: {save_path}")
        print(f"总共评估了 {len(self.instance_results)} 个实例")
        
        # 手动计算整体平均指标
        if len(self.instance_results) > 0:
            metrics_to_log = {}
            
            # 计算所有实例的平均指标
            avg_ade1 = sum(item['minADE1'] for item in self.instance_results) / len(self.instance_results)
            avg_ade6 = sum(item['minADE6'] for item in self.instance_results) / len(self.instance_results)
            avg_fde1 = sum(item['minFDE1'] for item in self.instance_results) / len(self.instance_results)
            avg_fde6 = sum(item['minFDE6'] for item in self.instance_results) / len(self.instance_results)
            avg_b_fde6 = sum(item['b-minFDE6'] for item in self.instance_results) / len(self.instance_results)
            
            # 计算 MR (Miss Rate)
            mr = sum(1 for item in self.instance_results if item['minFDE6'] > 2.0) / len(self.instance_results)
            
            metrics_to_log['val_minADE1'] = avg_ade1
            metrics_to_log['val_minADE6'] = avg_ade6
            metrics_to_log['val_minFDE1'] = avg_fde1
            metrics_to_log['val_minFDE6'] = avg_fde6
            metrics_to_log['val_MR'] = mr
            metrics_to_log['val_b-minFDE6'] = avg_b_fde6
            
            # 按类别统计
            print("\n" + "="*70)
            print("整体统计信息")
            print("="*70)
            print(f"总实例数: {len(self.instance_results)}")
            print(f"minADE1 平均: {avg_ade1:.6f}")
            print(f"minADE6 平均: {avg_ade6:.6f}")
            print(f"minFDE1 平均: {avg_fde1:.6f}")
            print(f"minFDE6 平均: {avg_fde6:.6f}")
            print(f"MR: {mr:.6f}")
            print(f"b-minFDE6 平均: {avg_b_fde6:.6f}")
            
            # 按类别统计
            class_names = {0: 'Vehicle', 1: 'Pedestrian', 2: 'Cyclist'}
            for class_id, class_name in class_names.items():
                class_instances = [item for item in self.instance_results if item['agent_type'] == class_id]
                if len(class_instances) > 0:
                    class_avg_fde6 = sum(item['minFDE6'] for item in class_instances) / len(class_instances)
                    class_mr = sum(1 for item in class_instances if item['minFDE6'] > 2.0) / len(class_instances)
                    print(f"\n{class_name} (类别 {class_id}):")
                    print(f"  实例数: {len(class_instances)}")
                    print(f"  minFDE6 平均: {class_avg_fde6:.6f}")
                    print(f"  MR: {class_mr:.6f}")
            
            # 找出表现最好和最差的实例
            print("\n" + "="*70)
            print("表现最好的5个实例 (minFDE6最小)")
            print("="*70)
            sorted_instances = sorted(self.instance_results, key=lambda x: x['minFDE6'])
            for i, item in enumerate(sorted_instances[:5]):
                print(f"{i+1}. scenario_id: {item['scenario_id']}, agent_type: {item['agent_type']}, "
                      f"minFDE6: {item['minFDE6']:.6f}, minADE6: {item['minADE6']:.6f}")
            
            print("\n" + "="*70)
            print("表现最差的5个实例 (minFDE6最大)")
            print("="*70)
            for i, item in enumerate(sorted_instances[-5:][::-1]):
                print(f"{i+1}. scenario_id: {item['scenario_id']}, agent_type: {item['agent_type']}, "
                      f"minFDE6: {item['minFDE6']:.6f}, minADE6: {item['minADE6']:.6f}")
            
            print("="*70)
            
            # 记录到 PyTorch Lightning
            self.log_dict(metrics_to_log, sync_dist=True)


@hydra.main(version_base=None, config_path="./conf/", config_name="config")
def main(conf):
    pl.seed_everything(conf.seed)
    output_dir = HydraConfig.get().runtime.output_dir
    
    # 从配置中获取checkpoint路径，支持命令行指定
    if conf.checkpoint:
        checkpoint = to_absolute_path(conf.checkpoint)
    else:
        raise ValueError("请通过 checkpoint=<路径> 参数指定模型权重文件路径")
    
    assert os.path.exists(checkpoint), f"指定的模型权重文件不存在: {checkpoint}"

    trainer = pl.Trainer(
        logger=False,
        accelerator="gpu",
        devices=conf.gpus,
        max_epochs=1,
        limit_val_batches=conf.limit_val_batches,
    )

    datamodule: pl.LightningDataModule = instantiate(conf.datamodule.target, test=conf.test)
    
    model_cfg = OmegaConf.to_container(conf.model.target.model, resolve=True)

    model = DetailedEvalTrainer(
        model=model_cfg, 
        lr=conf.model.target.lr,
        warmup_epochs=conf.model.target.warmup_epochs,
        epochs=conf.model.target.epochs,
        weight_decay=conf.model.target.weight_decay
    )

    if trainer.local_rank == 0:
        with open(os.path.join(output_dir, "model.log"), "w") as f:
            print(model.net, file=f)
    
    # 执行验证流程并捕获返回的结果
    results = trainer.validate(model, datamodule, ckpt_path=checkpoint)

    # 将汇总的指标结果保存到文本文件中 (仅在主进程中执行)
    if trainer.is_global_zero:
        summary_path = os.path.join(output_dir, "summary_metrics_detailed.txt")
        with open(summary_path, "w") as f:
            f.write("--- Final Validation Metrics ---\n")
            if results:
                # results 是一个列表，其中包含一个字典
                metrics = results[0]
                for key, value in sorted(metrics.items()):
                    f.write(f"{key}: {value}\n")
        print(f"汇总指标已保存至: {summary_path}")


if __name__ == "__main__":
    main()
