# E3–E5 experiments

Run these commands from `IntentG` with the DeMo conda environment active. The Python runners show normal training progress in the terminal and also save it to each run's `console.txt`. Full training is never started by running `--help`.

## Order

1. Run E4: `python experiment_revise/run_e4.py --gpus 4,5,6,7`. This runs both the rank sweep (`r=8,16,32,48,64` at `D=128`) and dimension sweep (`D=64,128,256` at `r=16`) for seeds 42–46. Completed groups are skipped, so rerunning after the rank sweep starts with `r16_d064`. It imports the old `r=32,D=128` checkpoints without retraining, re-evaluates their refined-best epochs, and writes both E4 tables. To import only the old baseline first, use `--only baseline`. To run one setting, use `--only r08_d128` (see `--help`).
2. Inspect `experiment_revise/results/E4/summary/rank_sensitivity.csv` and `dimension_sensitivity.csv`. **Choose** an E4-tested `(rank, dim)` pair yourself; the scripts do not choose one automatically.
3. Run E3 and E5 separately with that pair, for example `python experiment_revise/run_e3.py --gpus 4,5,6,7` and `python experiment_revise/run_e5.py --gpus 4,5,6,7` (both default to `r=16,D=128`). Use `--rank` and `--dim` to select another E4-tested pair. These commands require all five E4 evaluations for the selected pair. The matching E4 model is referenced instead of trained or copied again.

For one-variant runs use `--only NAME`. `--summary-only` recomputes reports from five complete seed results without starting training. Re-running the same command skips completed runs. If a training run was interrupted, it resumes from `train/checkpoints/last.ckpt` (the latest complete epoch) with optimizer and scheduler state; if that file is absent, it tries the latest retained epoch checkpoint. When no complete epoch exists, the incomplete attempt is archived under `train_attempts/` and the seed starts again. An interrupted evaluation is archived under `eval_attempts/` and repeated. The temporary `last.ckpt` is removed after successful training and validation; the top 10 and `best-epoch=*.ckpt` remain.

## Output layout

```text
experiment_revise/
  run_e4.py, run_e3.py, run_e5.py, common.py, complexity.py
  results/
    E4/r32_d128/seed_42/train/checkpoints/best-epoch=59.ckpt
    E4/r32_d128/seed_42/eval/summary_metrics.txt
    E4/r32_d128/summary/{per_seed.csv,mean_std.csv}
    E4/summary/{rank_sensitivity.csv,dimension_sensitivity.csv}
    E3/r16_d128/fullrank/seed_42/{train,eval}/...
    E3/r16_d128/fullrank/summary/{svd_by_seed_mode.csv,svd_mean_std.csv,svd_selected_ranks.csv,svd_cumulative_energy.png}
    E3/r16_d128/summary/comparison.csv
    E5/r16_d128/full_uni/seed_42/{train,eval}/...
    E5/r16_d128/summary/{comparison.csv,metric_note.txt}
    smoke/{E3,E5}/r16_d128/<variant>/seed_42/train/...
```

Each full group summary requires exactly seeds 42–46. `per_seed.csv` records every run; `mean_std.csv` contains five-seed mean and sample standard deviation. E5 reports both `val_*` (before refinement) and `val_new_*` (after refinement) from the **same jointly trained model**. E3 FLOPs use one fixed AV2 sample, THOP, and an analytical count for adapter operations; custom Mamba kernels may be undercounted, so interpret them as consistent estimates.

## Short training checks

After code changes, run `python experiment_revise/run_e3.py --smoke --gpus 4,5,6,7` and the corresponding `run_e5.py` command. Each variant completes two training batches and one validation batch on four GPUs, then exits. Smoke outputs remain under `results/smoke/` and never enter formal aggregates. Both E3 and E5 default to the selected `r=16,D=128` pair.

The E4 runner also supports `--smoke` for the three hidden dimensions at r=16, or `--smoke --only r16_d064` for a single configuration, without starting the full sweep.

For a direct `train.py` invocation, resume by using the same `hydra.run.dir` as the interrupted run; `train.py` automatically picks up `checkpoints/last.ckpt` if present. You can also pass `checkpoint=/absolute/path/to/checkpoint.ckpt` explicitly. A new default timestamped output directory is a new run and will not find the previous checkpoint.
