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

class TypewiseTrainer(Trainer):
    """
    一个专门用于按类别评估的Trainer。
    它会为每个指定的类别计算并记录独立的指标，
    并保存所有样本的详细结果以供后续分析。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 不使用 MetricCollection，因为它在逐样本更新时存在状态管理问题
        # 改为在 validation_step 中手动计算每个样本的 metrics，
        # 然后在 on_validation_epoch_end 中计算平均值

        # 初始化用于存储每个样本结果的字典
        self.results_detailed = {
            0: {'minADE1': [], 'minADE6': [], 'minFDE1': [], 'minFDE6': [], 'b-minFDE6': []},  # 类别0: 车辆
            1: {'minADE1': [], 'minADE6': [], 'minFDE1': [], 'minFDE6': [], 'b-minFDE6': []},  # 类别1: 行人
            2: {'minADE1': [], 'minADE6': [], 'minFDE1': [], 'minFDE6': [], 'b-minFDE6': []}   # 类别2: 骑行者
        }

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
        # - refined_out['y_hat']: [batch_size, num_modes, 60, 2]
        # - refined_out['pi']: [batch_size, num_modes]
        
        y_true = data['target'][:, 0]  # [batch_size, 60, 2]
        batch_size = y_true.shape[0]
        
        # 关键：获取每个batch样本的focal agent类型
        # x_attr shape: [batch_size, num_agents, 3]
        # 我们需要找到focal agent的类型（通常是第0个agent）
        agent_types = data['x_attr'][:, 0, 2]  # [batch_size] - 每个样本的第0个agent类型
        device = y_true.device

        # 对batch中的每个样本（每个样本对应一个focal agent）
        for i in range(batch_size):
            agent_type = agent_types[i].item()

            # 提取单个样本的预测
            single_out = {
                'y_hat': refined_out['y_hat'][i:i+1],  # [1, num_modes, 60, 2]
                'pi': refined_out['pi'][i:i+1]          # [1, num_modes]
            }
            single_y_true = y_true[i:i+1]  # [1, 60, 2]

            # 计算此样本的核心指标值并保存
            ade1 = minADE(k=1).to(device)(single_out, single_y_true).item()
            ade6 = minADE(k=6).to(device)(single_out, single_y_true).item()
            fde1 = minFDE(k=1).to(device)(single_out, single_y_true).item()
            fde6 = minFDE(k=6).to(device)(single_out, single_y_true).item()
            b_fde6 = brier_minFDE(k=6).to(device)(single_out, single_y_true).item()

            # 根据 agent 类型保存结果
            if agent_type == 0:
                self.results_detailed[0]['minADE1'].append(ade1)
                self.results_detailed[0]['minADE6'].append(ade6)
                self.results_detailed[0]['minFDE1'].append(fde1)
                self.results_detailed[0]['minFDE6'].append(fde6)
                self.results_detailed[0]['b-minFDE6'].append(b_fde6)
            elif agent_type == 1:
                self.results_detailed[1]['minADE1'].append(ade1)
                self.results_detailed[1]['minADE6'].append(ade6)
                self.results_detailed[1]['minFDE1'].append(fde1)
                self.results_detailed[1]['minFDE6'].append(fde6)
                self.results_detailed[1]['b-minFDE6'].append(b_fde6)
            elif agent_type == 2:
                self.results_detailed[2]['minADE1'].append(ade1)
                self.results_detailed[2]['minADE6'].append(ade6)
                self.results_detailed[2]['minFDE1'].append(fde1)
                self.results_detailed[2]['minFDE6'].append(fde6)
                self.results_detailed[2]['b-minFDE6'].append(b_fde6)

    def on_validation_epoch_end(self):
        # 不再使用 MetricCollection.compute()，因为它有状态共享的 bug
        # 改为使用手动计算的平均值
        
        output_dir = HydraConfig.get().runtime.output_dir
        save_path = os.path.join(output_dir, "class_wise_detailed.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(self.results_detailed, f)
        print(f"分类别详细评估结果已保存至: {save_path}")
        
        # 手动计算并记录每个类别的指标
        metrics_to_log = {}
        
        # 添加诊断信息
        print("\n" + "="*70)
        print("分类别统计信息")
        print("="*70)
        
        for class_id, class_name, prefix in [
            (0, 'Vehicle', 'val_vehicle_'),
            (1, 'Pedestrian', 'val_pedestrian_'),
            (2, 'Cyclist', 'val_cyclist_')
        ]:
            if len(self.results_detailed[class_id]['minFDE6']) > 0:
                ade1_vals = self.results_detailed[class_id]['minADE1']
                ade6_vals = self.results_detailed[class_id]['minADE6']
                fde1_vals = self.results_detailed[class_id]['minFDE1']
                fde6_vals = self.results_detailed[class_id]['minFDE6']
                b_fde6_vals = self.results_detailed[class_id]['b-minFDE6']
                
                # 计算平均值
                num_samples = len(fde6_vals)
                avg_ade1 = sum(ade1_vals) / num_samples
                avg_ade6 = sum(ade6_vals) / num_samples
                avg_fde1 = sum(fde1_vals) / num_samples
                avg_fde6 = sum(fde6_vals) / num_samples
                avg_b_fde6 = sum(b_fde6_vals) / num_samples
                
                # 计算 MR (Miss Rate)
                # MR 定义为 minFDE6 > 2.0 的样本比例
                mr = sum(1 for fde in fde6_vals if fde > 2.0) / num_samples
                
                # 记录到 metrics_to_log（用于 trainer.validate() 返回）
                metrics_to_log[f"{prefix}minADE1"] = avg_ade1
                metrics_to_log[f"{prefix}minADE6"] = avg_ade6
                metrics_to_log[f"{prefix}minFDE1"] = avg_fde1
                metrics_to_log[f"{prefix}minFDE6"] = avg_fde6
                metrics_to_log[f"{prefix}MR"] = mr
                metrics_to_log[f"{prefix}b-minFDE6"] = avg_b_fde6
                
                # 打印诊断信息
                print(f"\n{class_name} (类别 {class_id}):")
                print(f"  样本数: {num_samples}")
                print(f"  minADE1 平均: {avg_ade1:.6f}")
                print(f"  minADE6 平均: {avg_ade6:.6f}")
                print(f"  minFDE1 平均: {avg_fde1:.6f}")
                print(f"  minFDE6 平均: {avg_fde6:.6f}")
                print(f"  MR: {mr:.6f}")
                print(f"  b-minFDE6 平均: {avg_b_fde6:.6f}")
                
                # 检查是否有很多相同的值
                equal_count = sum(1 for v1, v6 in zip(fde1_vals, fde6_vals) if abs(v1 - v6) < 1e-6)
                print(f"  minFDE1==minFDE6 的样本数: {equal_count} ({equal_count/num_samples*100:.2f}%)")
                
                if equal_count > 0:
                    print(f"  警告: 存在 {equal_count} 个样本的 minFDE1 和 minFDE6 完全相同！")
                    print(f"        这通常意味着模型对这些样本输出的所有模式预测相同")
        
        print("="*70)
        
        # 使用正确的手动计算值记录到 PyTorch Lightning
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

    model = TypewiseTrainer(
        model=model_cfg, 
        lr=conf.model.target.lr,
        warmup_epochs=conf.model.target.warmup_epochs,
        epochs=conf.model.target.epochs,
        weight_decay=conf.model.target.weight_decay
    )

    if trainer.local_rank == 0:
        with open(os.path.join(output_dir, "model.log"), "w") as f:
            print(model.net, file=f)
            
    # trainer.validate(model, datamodule, ckpt_path=checkpoint)
    
    # --- 新增保存结果 ---
    # 执行验证流程并捕获返回的结果
    results = trainer.validate(model, datamodule, ckpt_path=checkpoint)

    # 将汇总的指标结果保存到文本文件中 (仅在主进程中执行)
    if trainer.is_global_zero:
        summary_path = os.path.join(output_dir, "summary_metrics_type.txt")
        with open(summary_path, "w") as f:
            f.write("--- Final Validation Metrics ---\n")
            if results:
                # results 是一个列表，其中包含一个字典
                metrics = results[0]
                for key, value in sorted(metrics.items()):
                    f.write(f"{key}: {value}\n")
        print(f"汇总指标已保存至: {summary_path}")
    # ------------------


if __name__ == "__main__":
    main()