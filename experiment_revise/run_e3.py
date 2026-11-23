"""E3 offset study; defaults to the E4-selected rank 16 and dimension 128."""

import argparse
import csv
import json
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from common import (PRIMARY_METRICS, RESULTS, SEEDS, best_checkpoint, capacity_name,
                    ensure_e4_complete, evaluate_one, gpu_ids, metrics_from_file, seed_dir,
                    summary_for_group, train_one)
from complexity import measure


OFFSETS = ("global", "fullrank", "lowrank")
COMPARISON_METRICS = (*PRIMARY_METRICS, "val_new_minADE1", "val_new_b-minFDE6", "val_new_minFDE1")
DEFAULT_RANK = 16
DEFAULT_DIM = 128


def analyze_svd(group):
    out = group / "summary"
    raw = []
    for seed in SEEDS:
        checkpoint = best_checkpoint(seed_dir(group, seed) / "train")
        state = torch.load(checkpoint, map_location="cpu")["state_dict"]
        delta = state["net.time_decoder.full_rank_adapter.mode_specific_delta"].double()
        for mode, matrix in enumerate(delta):
            singular = torch.linalg.svdvals(matrix)
            energy = singular.square()
            cumulative = (energy.cumsum(0) / energy.sum().clamp_min(1e-12)).tolist()
            raw.extend((seed, mode, rank, value) for rank, value in enumerate(cumulative, 1))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "svd_by_seed_mode.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("seed", "mode", "rank", "cumulative_energy"))
        writer.writerows(raw)
    ranks = sorted({row[2] for row in raw})
    summary = []
    for rank in ranks:
        values = [row[3] for row in raw if row[2] == rank]
        summary.append((rank, mean(values), stdev(values), len(values)))
    with (out / "svd_mean_std.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("rank", "mean", "std", "count"))
        writer.writerows(summary)
    by_rank = {row[0]: row for row in summary}
    with (out / "svd_selected_ranks.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("requested_rank", "effective_rank", "mean", "std", "count"))
        for requested in (8, 16, 32, 48, 64):
            effective = min(requested, ranks[-1])
            _, avg, sd, count = by_rank[effective]
            writer.writerow((requested, effective, avg, sd, count))
    plt.figure(figsize=(6, 4))
    plt.plot(ranks, [row[1] for row in summary], label="Full-rank learned offsets")
    plt.xlabel("Rank")
    plt.ylabel("Cumulative singular-value energy")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "svd_cumulative_energy.png", dpi=200)
    plt.close()


def write_comparison(base, e4_group):
    complexities = {
        offset: json.loads((base / offset / "summary" / "complexity.json").read_text())
        for offset in OFFSETS
    }
    # The E3 variants share all parameters outside their offset adapters.
    shared_parameters = complexities["global"]["parameters"]
    rows = []
    for offset in OFFSETS:
        group = base / offset
        source = e4_group if offset == "lowrank" else group
        seed_metrics = []
        for seed in SEEDS:
            path = seed_dir(source, seed) / "eval" / "summary_metrics.txt"
            metrics = metrics_from_file(path)
            missing = [key for key in COMPARISON_METRICS if key not in metrics]
            if missing:
                raise RuntimeError(f"Missing {missing} in {path}")
            seed_metrics.append(metrics)
        means = [mean(metrics[key] for metrics in seed_metrics) for key in COMPARISON_METRICS]
        complexity = complexities[offset]
        parameters = complexity["parameters"]
        rows.append((offset, *means, parameters, parameters - shared_parameters, complexity["flops_g"]))

    path = base / "summary" / "comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("offset", *[f"{key}_mean" for key in COMPARISON_METRICS],
                         "parameters", "offset_parameters", "flops_g"))
        writer.writerows(rows)
    return path


def report_if_complete(base, e4_group, rank, dim, gpu_ids_value):
    for offset in OFFSETS:
        group = base / offset
        source = e4_group if offset == "lowrank" else group
        if not all((seed_dir(source, seed) / "eval" / "summary_metrics.txt").exists() for seed in SEEDS):
            print(f"E3 still incomplete: {offset}")
            return
        summary_for_group(group, source=source, include_base=True)
        measure(group, rank, dim, offset, gpu_ids_value.split(",")[0])
    analyze_svd(base / "fullrank")
    comparison = write_comparison(base, e4_group)
    print(f"E3 reports written to {comparison.parent}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--gpus", type=gpu_ids, default="0,1,2,3")
    parser.add_argument("--smoke", action="store_true", help="Two train batches, one validation batch, one seed")
    parser.add_argument("--only", choices=OFFSETS, help="Run just one offset variant")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    e4_group = None if args.smoke else ensure_e4_complete(args.rank, args.dim)
    base = RESULTS / ("smoke/E3" if args.smoke else "E3") / capacity_name(args.rank, args.dim)
    offsets = (args.only,) if args.only else OFFSETS
    if not args.summary_only:
        for offset in offsets:
            group = base / offset
            if not args.smoke and offset == "lowrank":
                summary_for_group(group, source=e4_group, include_base=True)
                continue
            for seed in (SEEDS[:1] if args.smoke else SEEDS):
                train_one(group, seed, args.rank, args.dim, args.gpus, offset=offset,
                          smoke=args.smoke)
                if not args.smoke:
                    evaluate_one(group, seed, args.rank, args.dim, args.gpus, offset=offset)
            if not args.smoke:
                summary_for_group(group, include_base=True)
    if not args.smoke:
        report_if_complete(base, e4_group, args.rank, args.dim, args.gpus)


if __name__ == "__main__":
    main()
