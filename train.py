import os
import shutil
import sys
import time
from datetime import datetime, timedelta

import hydra
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.model_summary import ModelSummary


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(conf):
    pl.seed_everything(conf.seed, workers=True)
    output_dir = HydraConfig.get().runtime.output_dir

    logger = TensorBoardLogger(save_dir=output_dir, name="logs")

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="{epoch}",
        monitor=f"{conf.monitor}",
        mode="min",
        save_top_k=conf.save_top_k,
        # Keep the latest complete epoch while fitting so an interrupted run
        # can restore the optimizer, scheduler, and loop state.
        save_last=True,
    )
    callbacks = [
        checkpoint_callback,
        RichModelSummary(max_depth=1),
        RichProgressBar(),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        logger=logger,
        gradient_clip_val=conf.gradient_clip_val,
        gradient_clip_algorithm=conf.gradient_clip_algorithm,
        max_epochs=conf.epochs,
        accelerator="gpu",
        devices=conf.gpus,
        strategy="ddp_find_unused_parameters_false",
        callbacks=callbacks,
        limit_train_batches=conf.limit_train_batches,
        limit_val_batches=conf.limit_val_batches,
        sync_batchnorm=conf.sync_bn,
    )

    model = instantiate(conf.model.target)
    
    # Save model parameters count using ModelSummary
    summary = ModelSummary(model, max_depth=1)
    with open(f'{output_dir}/parameters.txt', 'w') as f:
        f.write(f"Total parameters: {summary.total_parameters:,}\n")
        f.write(f"Trainable parameters: {summary.trainable_parameters:,}\n")
        f.write(f"Non-trainable parameters: {summary.total_parameters - summary.trainable_parameters:,}\n")
    
    os.system('cp -a %s %s' % ('conf', output_dir))
    os.system('cp -a %s %s' % ('src', output_dir))
    with open(f'{output_dir}/model.txt', 'w') as f:
        original_stdout = sys.stdout  
        sys.stdout = f  
        print(model)  
        sys.stdout = original_stdout  
    datamodule = instantiate(conf.datamodule.target)
    checkpoint = conf.checkpoint
    last_checkpoint = os.path.join(output_dir, "checkpoints", "last.ckpt")
    if not checkpoint and os.path.isfile(last_checkpoint):
        checkpoint = last_checkpoint
    if checkpoint:
        print(f"Resuming training from {checkpoint}", flush=True)

    train_start_time = datetime.now().astimezone()
    train_start = time.perf_counter()
    fit_completed = False
    try:
        trainer.fit(model, datamodule, ckpt_path=checkpoint)
        fit_completed = True
    finally:
        train_end = time.perf_counter()
        train_end_time = datetime.now().astimezone()
        if trainer.is_global_zero:
            duration_seconds = train_end - train_start
            with open(os.path.join(output_dir, "train_time.txt"), "w") as f:
                f.write(f"seed: {conf.seed}\n")
                f.write(f"status: {'completed' if fit_completed else 'incomplete'}\n")
                f.write(f"resume_checkpoint: {checkpoint or 'none'}\n")
                f.write(f"start_time: {train_start_time.isoformat(timespec='seconds')}\n")
                f.write(f"end_time: {train_end_time.isoformat(timespec='seconds')}\n")
                f.write(f"duration_seconds: {duration_seconds:.3f}\n")
                f.write(f"duration: {str(timedelta(seconds=round(duration_seconds)))}\n")
    if trainer.is_global_zero and checkpoint_callback.best_model_path:
        best_checkpoint_name = "best-" + os.path.basename(checkpoint_callback.best_model_path)
        shutil.copy2(
            checkpoint_callback.best_model_path,
            os.path.join(output_dir, "checkpoints", best_checkpoint_name),
        )
    trainer.validate(model, datamodule.val_dataloader())
    if trainer.is_global_zero and os.path.isfile(last_checkpoint):
        os.remove(last_checkpoint)


if __name__ == "__main__":
    main()
