import os
import sys
import hydra
import pickle
import torch
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from torchmetrics import MetricCollection
from omegaconf import OmegaConf

# 从现有代码库中导入必要的模块
from src.model.trainer_forecast import Trainer
from src.metrics import MR, minADE, minFDE, brier_minFDE

class TypewiseTrainer(Trainer):
    """
    一个专门用于按类别评估的Trainer。
    它会为每个指定的类别计算并记录独立的指标，
    并保存所有样本的详细结果以供后续分析。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 为每个类别定义一套独立的评估指标 (新增了 b-minFDE6)
        metrics = MetricCollection({
            "minADE1": minADE(k=1),
            "minADE6": minADE(k=6),
            "minFDE1": minFDE(k=1),
            "minFDE6": minFDE(k=6),
            "MR": MR(),
            "b-minFDE6": brier_minFDE(k=6), # <-- 新增指标
        })
        self.val_metrics_vehicle = metrics.clone(prefix="val_vehicle_")
        self.val_metrics_pedestrian = metrics.clone(prefix="val_pedestrian_")
        self.val_metrics_cyclist = metrics.clone(prefix="val_cyclist_")

        # 初始化用于存储每个样本结果的字典 (为 b-minFDE6 新增了键)
        self.results_detailed = {
            0: {'minADE6': [], 'minFDE6': [], 'b-minFDE6': []},  # 类别0: 车辆
            1: {'minADE6': [], 'minFDE6': [], 'b-minFDE6': []},  # 类别1: 行人
            2: {'minADE6': [], 'minFDE6': [], 'b-minFDE6': []}   # 类别2: 骑行者
        }

    def validation_step(self, data, batch_idx):
        if isinstance(data, list):
            data = data[-1]
            
        out = self(data)
        
        if out.get('new_y_hat') is not None:
            out['y_hat'] = out['new_y_hat']
            out['pi'] = out['new_pi']

        y_true = data['target'][:, 0]
        agent_types = data['x_attr'][:, 0, 2]
        device = y_true.device

        for i in range(len(agent_types)):
            agent_type = agent_types[i].item()

            single_out = {k: v[i:i+1] for k, v in out.items() if isinstance(v, torch.Tensor) and v.dim() > 1 and v.shape[0] == len(agent_types)}
            single_y_true = y_true[i:i+1]

            # 计算此样本的核心指标值并保存 (新增了 b_fde6 的计算)
            ade6 = minADE(k=6).to(device)(single_out, single_y_true).item()
            fde6 = minFDE(k=6).to(device)(single_out, single_y_true).item()
            b_fde6 = brier_minFDE(k=6).to(device)(single_out, single_y_true).item() # <-- 新增计算

            if agent_type == 0:
                self.val_metrics_vehicle.update(single_out, single_y_true)
                self.results_detailed[0]['minADE6'].append(ade6)
                self.results_detailed[0]['minFDE6'].append(fde6)
                self.results_detailed[0]['b-minFDE6'].append(b_fde6) # <-- 新增保存
            elif agent_type == 1:
                self.val_metrics_pedestrian.update(single_out, single_y_true)
                self.results_detailed[1]['minADE6'].append(ade6)
                self.results_detailed[1]['minFDE6'].append(fde6)
                self.results_detailed[1]['b-minFDE6'].append(b_fde6) # <-- 新增保存
            elif agent_type == 2:
                self.val_metrics_cyclist.update(single_out, single_y_true)
                self.results_detailed[2]['minADE6'].append(ade6)
                self.results_detailed[2]['minFDE6'].append(fde6)
                self.results_detailed[2]['b-minFDE6'].append(b_fde6) # <-- 新增保存

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics_vehicle.compute(), sync_dist=True)
        self.log_dict(self.val_metrics_pedestrian.compute(), sync_dist=True)
        self.log_dict(self.val_metrics_cyclist.compute(), sync_dist=True)

        output_dir = HydraConfig.get().runtime.output_dir
        save_path = os.path.join(output_dir, "class_wise_detailed.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(self.results_detailed, f)
        print(f"分类别详细评估结果已保存至: {save_path}")


@hydra.main(version_base=None, config_path="./conf/", config_name="config")
def main(conf):
    pl.seed_everything(conf.seed)
    output_dir = HydraConfig.get().runtime.output_dir
    checkpoint = to_absolute_path("outputs/lowrank_r32/20250925-002248/checkpoints/epoch=59.ckpt")
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
        summary_path = os.path.join(output_dir, "summary_metrics.txt")
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