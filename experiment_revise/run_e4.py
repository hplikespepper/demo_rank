"""E4 capacity sweep. Run manually after reviewing --help."""

import argparse
import csv
import json
import shutil

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from common import (PRIMARY_METRICS, RESULTS, ROOT, SEEDS, e4_name,
                    evaluate_one, gpu_ids, seed_dir,
                    metrics_from_file, summary_for_group, train_one)


RANKS = (8, 16, 32, 48, 64)
DIMS = (64, 128, 256)
DIMENSION_RANK = 16
CONFIGS = [(rank, 128) for rank in RANKS] + [
    (DIMENSION_RANK, dim) for dim in DIMS if dim != 128
]
LEGACY_TRAIN = ROOT.parent / "outputs" / "lowrank_r32_simplify_seed"


def group_evaluated(group):
    for seed in SEEDS:
        path = seed_dir(group, seed) / "eval" / "summary_metrics.txt"
        if not path.exists() or not all(key in metrics_from_file(path) for key in PRIMARY_METRICS):
            return False
    return True


def import_default_seed(seed):
    group = RESULTS / "E4" / e4_name(32, 128)
    dest = seed_dir(group, seed) / "train"
    existing = sorted((dest / "checkpoints").glob("best-epoch=*.ckpt"))
    if len(existing) == 1 and (dest / "provenance.json").exists():
        return
    if dest.exists():
        raise RuntimeError(f"Incomplete legacy import: {dest}")
    source_dirs = sorted((LEGACY_TRAIN / f"seed_{seed}").glob("20*"))
    if len(source_dirs) != 1:
        raise RuntimeError(f"Expected one legacy training run for seed {seed}, found {len(source_dirs)}")
    source = source_dirs[0]
    events = EventAccumulator(str(source / "logs" / "version_0"), size_guidance={"scalars": 0})
    events.Reload()
    scores = list(events.Scalars("val_new_minFDE6"))
    # train.py runs a second validation after fit; that final event has no checkpoint.
    if (len(scores) > 1 and scores[-1].step == scores[-2].step + 1
            and abs(scores[-1].value - scores[-2].value) < 1e-8):
        scores.pop()
    best = min(scores, key=lambda event: event.value)
    epochs = {int(round(event.value)) for event in events.Scalars("epoch") if event.step == best.step}
    if len(epochs) != 1:
        raise RuntimeError(f"Could not map refined best score to one epoch: seed {seed}")
    epoch = next(iter(epochs))
    checkpoint = source / "checkpoints" / f"epoch={epoch}.ckpt"
    if not checkpoint.exists():
        raise RuntimeError(f"Legacy refined best checkpoint is missing: {checkpoint}")
    (dest / "checkpoints").mkdir(parents=True)
    shutil.copy2(checkpoint, dest / "checkpoints" / f"best-epoch={epoch}.ckpt")
    shutil.copy2(source / "parameters.txt", dest / "parameters.txt")
    shutil.copy2(source / ".hydra" / "config.yaml", dest / "legacy_config.yaml")
    (dest / "provenance.json").write_text(json.dumps({
        "source_checkpoint": str(checkpoint),
        "source_seed": seed,
        "selected_epoch": epoch,
        "selection_metric": "val_new_minFDE6",
        "training_event_score": best.value,
        "training_was_reused": True,
    }, indent=2) + "\n")


def report_if_complete():
    groups = {}
    for rank, dim in CONFIGS:
        group = RESULTS / "E4" / e4_name(rank, dim)
        if not group_evaluated(group):
            print(f"E4 still incomplete: {group.name}")
            return
        summary_for_group(group, include_base=True)
        with (group / "summary" / "mean_std.csv").open() as f:
            groups[(rank, dim)] = {row["metric"]: row for row in csv.DictReader(f)}
    report_dir = RESULTS / "E4" / "summary"
    report_dir.mkdir(parents=True, exist_ok=True)
    for filename, pairs in (("rank_sensitivity.csv", [(rank, 128) for rank in RANKS]),
                            ("dimension_sensitivity.csv", [(DIMENSION_RANK, dim) for dim in DIMS])):
        with (report_dir / filename).open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("rank", "dim", "source", *[f"{k}_mean" for k in PRIMARY_METRICS],
                             *[f"{k}_std" for k in PRIMARY_METRICS], "parameters"))
            for rank, dim in pairs:
                stats = groups[(rank, dim)]
                writer.writerow((rank, dim, "legacy_reevaluated" if (rank, dim) == (32, 128) else "new",
                                 *[stats[k]["mean"] for k in PRIMARY_METRICS],
                                 *[stats[k]["std"] for k in PRIMARY_METRICS],
                                 stats["parameters"]["mean"]))
    print(f"E4 reports written to {report_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=gpu_ids, default="0,1,2,3", help="Four GPU IDs for training")
    parser.add_argument("--only", choices=["baseline", *[e4_name(r, d) for r, d in CONFIGS]],
                        help="Run only one configuration; baseline imports and re-evaluates old weights")
    parser.add_argument("--summary-only", action="store_true", help="Aggregate completed runs without training")
    parser.add_argument("--smoke", action="store_true", help="Two train batches and one validation batch for one seed")
    args = parser.parse_args()
    if args.summary_only:
        report_if_complete()
        return
    if args.smoke:
        if args.only == "baseline":
            parser.error("--smoke cannot be combined with --only baseline")
        selected = [(DIMENSION_RANK, dim) for dim in DIMS] if not args.only else [
            pair for pair in CONFIGS if e4_name(*pair) == args.only
        ]
        for rank, dim in selected:
            train_one(RESULTS / "smoke" / "E4" / e4_name(rank, dim), SEEDS[0],
                      rank, dim, args.gpus, smoke=True)
        return
    chosen = [(32, 128)] if args.only == "baseline" else CONFIGS
    if args.only and args.only != "baseline":
        chosen = [pair for pair in CONFIGS if e4_name(*pair) == args.only]
    for rank, dim in chosen:
        group = RESULTS / "E4" / e4_name(rank, dim)
        if group_evaluated(group):
            print(f"Skipping completed E4 group: {group.name}", flush=True)
            continue
        for seed in SEEDS:
            if (rank, dim) == (32, 128):
                import_default_seed(seed)
            else:
                train_one(group, seed, rank, dim, args.gpus)
            evaluate_one(group, seed, rank, dim, args.gpus)
        summary_for_group(group, include_base=True)
    report_if_complete()


if __name__ == "__main__":
    main()
