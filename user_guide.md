# IntentG User Guide

IntentG is a motion-forecasting model for Argoverse 2 (AV2), built on the DeMo codebase.
It decodes six trajectory modes and adds a low-rank, mode-specific offset (`low_rank_dim`, default 32)
plus a refinement stage (cross-attention, mode self-attention and a bidirectional Mamba block).
The model reports metrics both before refinement (`val_*`) and after refinement (`val_new_*`).

## 1. Repository layout

```text
IntentG/
├── conf/                      # Hydra configs
│   ├── config.yaml            # base config (used by eval.py)
│   ├── train.yaml             # training config (config.yaml + stdout logging)
│   ├── datamodule/av2.yaml    # data paths, batch size, train_mode
│   └── model/model_forecast.yaml  # model hyper-parameters
├── src/
│   ├── datamodule/            # AV2 dataset, extractor (preprocessing) and datamodule
│   ├── model/
│   │   ├── model_forecast.py  # IntentG network
│   │   ├── trainer_forecast.py# LightningModule (losses, metrics, optimizer)
│   │   └── layers/            # encoders, time_decoder.py (low-rank offset + refinement), Mamba
│   ├── metrics/               # minADE / minFDE / MR
│   └── utils/                 # loss, optimizer, AV2 submission helper
├── preprocess.py              # raw AV2 parquet -> processed tensors
├── train.py                   # training entry point
├── eval.py                    # validation / test entry point
├── vis_paper.py               # fixed-size static figures used in the paper
├── vis_gif_combine.py         # multi-scenario video / gif of predictions
└── experiment_revise/         # revision experiments E3–E5 (see its README.md)
```

`src/model/layers/time_decoder_origin.py` (original DeMo decoder) and
`time_decoder_norefinemodule.py` (decoder without the refinement module) are kept for reference.
`model_forecast.py` imports `time_decoder.py`, which is the IntentG decoder.

## 2. Environment

Tested with Python 3.10, PyTorch 2.1.1 (CUDA 11.8) and PyTorch Lightning 2.x.

```bash
conda create -n intentg python=3.10 -y
conda activate intentg
pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install av2 mamba-ssm causal-conv1d   # AV2 API and Mamba kernels
pip install click joblib thop tensorboard # visualization and experiment_revise
```

Run every command below from the `IntentG/` directory.

## 3. Data preparation

1. Download the Argoverse 2 Motion Forecasting dataset so that it looks like
   `<AV2_ROOT>/{train,val,test}/<scenario_id>/*.parquet`.
2. Preprocess it. The output is written to `data/DeMo_processed/{train,val,test}`, which is the
   default `data_root` in `conf/datamodule/av2.yaml`.

```bash
python preprocess.py --data_root <AV2_ROOT> --parallel
```

## 4. Training

`train.py` uses `conf/train.yaml`. Any config value can be overridden from the command line (Hydra syntax).

```bash
# IntentG (low-rank offset, r=32), 8 GPUs, defaults from conf/
python train.py output=intentg_r32 seed=42

# fewer GPUs / smaller batch
python train.py gpus=4 batch_size=16 output=intentg_r32 seed=42

# original DeMo-style global offset (baseline)
python train.py model.target.model.use_low_rank=false output=baseline seed=42
```

Main options:

| Override | Default | Meaning |
|---|---|---|
| `model.target.model.low_rank_dim` | `32` | rank of the mode-specific offset |
| `model.target.model.embed_dim` | `128` | hidden dimension D |
| `model.target.model.offset_mode` | `null` | `lowrank` / `fullrank` / `global`; `null` follows `use_low_rank` |
| `model.target.model.refine_mhca` / `refine_mhsa` / `refine_mamba` | `true` | enable each refinement block |
| `model.target.model.refine_mamba_bidirectional` | `true` | bidirectional Mamba in refinement |
| `datamodule.target.dataset.train_mode` | `only_focal` | `focal_and_scored` for multi-agent training |
| `gpus`, `batch_size`, `epochs`, `lr`, `seed` | `8`, `16`, `60`, `0.003`, `2024` | trainer settings |

Outputs go to `outputs/<output>/<timestamp>/`: `checkpoints/` (top-10 by `val_minFDE6`, plus a
`best-epoch=*.ckpt` copy), TensorBoard `logs/`, `parameters.txt` and `train_time.txt`.

To resume an interrupted run, rerun with the same `hydra.run.dir=<run_dir>`; `train.py` picks up
`checkpoints/last.ckpt` automatically. You can also pass `checkpoint=/path/to/ckpt`.

## 5. Evaluation

Use the same model overrides that were used for training.

```bash
# validation set -> outputs/<output>/<timestamp>/summary_metrics.txt
CUDA_VISIBLE_DEVICES=0 python eval.py gpus=1 \
    model.target.model.low_rank_dim=32 \
    checkpoint='outputs/intentg_r32/<timestamp>/checkpoints/epoch\=59.ckpt' \
    output=intentg_r32_eval

# test set -> AV2 submission parquet for the leaderboard
CUDA_VISIBLE_DEVICES=0 python eval.py gpus=1 test=true \
    model.target.model.low_rank_dim=32 \
    checkpoint='outputs/intentg_r32/<timestamp>/checkpoints/epoch\=59.ckpt'
```

Escape `=` in checkpoint file names as `\=`, otherwise Hydra parses it as an override.
Models trained with `train_mode=focal_and_scored` must be evaluated with the same setting.

## 6. Visualization

`vis_paper.py` draws one scenario with a fixed field of view (110 m × 70 m by default):
history, ground truth and the six predicted trajectories.

```bash
python vis_paper.py \
    --checkpoint outputs/intentg_r32/<timestamp>/checkpoints/epoch=59.ckpt \
    --scene_id 6a28ba10-4ba7-4664-b85f-9adb89729225 \
    --argo2-data-dir <AV2_ROOT>/val \
    --data-root data/DeMo_processed \
    --save-dir viz_paper
# --idx N selects the N-th validation scene instead of --scene_id
# --only_truth draws history and ground truth only
```

`vis_gif_combine.py` renders several scenarios side by side into one mp4 (or gif with `--gif`),
with predicted trajectories overlaid when a checkpoint is given.

```bash
python vis_gif_combine.py \
    --argoverse-scenario-dir <AV2_ROOT>/val \
    --checkpoint outputs/intentg_r32/<timestamp>/checkpoints/epoch=59.ckpt \
    --viz-output-path viz/examples.gif --gif
# --scene-ids id1,id2,id3 chooses the scenarios (three paper examples by default)
```

## 7. Revision experiments (E3–E5)

`experiment_revise/` contains the scripts for the revision experiments:
E4 (rank and hidden-dimension sensitivity), E3 (low-rank vs. full-rank vs. global offset, SVD
analysis and complexity) and E5 (refinement-module ablation). Each runner trains and evaluates
seeds 42–46 and writes summaries to `experiment_revise/results/`. Usage, run order and output
layout are in [experiment_revise/README.md](experiment_revise/README.md).

```bash
python experiment_revise/run_e4.py --gpus 4,5,6,7
python experiment_revise/run_e3.py --gpus 4,5,6,7 --rank 16 --dim 128
python experiment_revise/run_e5.py --gpus 4,5,6,7 --rank 16 --dim 128
```
