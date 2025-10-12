import os
import sys
import hydra
from hydra.core.hydra_config import HydraConfig
import pytorch_lightning as pl
from hydra.utils import instantiate, to_absolute_path
from importlib import import_module


@hydra.main(version_base=None, config_path="./conf/", config_name="config")
def main(conf):
    pl.seed_everything(conf.seed)
    output_dir = HydraConfig.get().runtime.output_dir
    checkpoint = to_absolute_path(conf.checkpoint)
    assert os.path.exists(checkpoint), f"Checkpoint {checkpoint} does not exist"

    trainer = pl.Trainer(
        logger=False,
        accelerator="gpu",
        devices=conf.gpus,
        max_epochs=1,
        limit_val_batches=conf.limit_val_batches,
        limit_test_batches=conf.limit_test_batches,
    )

    datamodule: pl.LightningDataModule = instantiate(conf.datamodule.target, test=conf.test)
    model = instantiate(conf.model.target)
    os.system('cp -a %s %s' % ('conf', output_dir))
    # os.system('cp -a %s %s' % ('src', output_dir))
    if trainer.local_rank == 0:
        with open(os.path.join(HydraConfig.get().runtime.output_dir, "model.log"), "x") as f:
            print(model.net, file=f)
    if not conf.test:
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
    else:
        trainer.test(model, datamodule, ckpt_path=checkpoint)


if __name__ == "__main__":
    main()
