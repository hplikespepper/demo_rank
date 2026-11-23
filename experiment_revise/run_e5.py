"""E5 refinement study; defaults to the E4-selected rank 16 and dimension 128."""

import argparse
import csv

from common import (BASE_METRICS, PRIMARY_METRICS, RESULTS, SEEDS, capacity_name,
                    ensure_e4_complete, evaluate_one, gpu_ids, seed_dir,
                    summary_for_group, train_one)


# Each row is (MHCA, MHSA, Mamba, bidirectional Mamba).
VARIANTS = {
    "mhca": (True, False, False, True),
    "mhsa": (False, True, False, True),
    "bimamba": (False, False, True, True),
    "mhca_mhsa": (True, True, False, True),
    "mhca_bimamba": (True, False, True, True),
    "mhsa_bimamba": (False, True, True, True),
    "full_bi": (True, True, True, True),
    "full_uni": (True, True, True, False),
}
DEFAULT_RANK = 16
DEFAULT_DIM = 128


def report_if_complete(base, e4_group):
    for name in VARIANTS:
        group = base / name
        source = e4_group if name == "full_bi" else group
        if not all((seed_dir(source, seed) / "eval" / "summary_metrics.txt").exists() for seed in SEEDS):
            print(f"E5 still incomplete: {name}")
            return
        summary_for_group(group, source=source, include_base=True)
    report_dir = base / "summary"
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics = BASE_METRICS + PRIMARY_METRICS
    with (report_dir / "comparison.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("variant", "mhca", "mhsa", "mamba", "direction",
                         *[f"{key}_mean" for key in metrics],
                         *[f"{key}_std" for key in metrics]))
        for name, flags in VARIANTS.items():
            with (base / name / "summary" / "mean_std.csv").open() as stats_file:
                stats = {row["metric"]: row for row in csv.DictReader(stats_file)}
            writer.writerow((name, *flags[:3], "bi" if flags[3] else "uni",
                             *[stats[key]["mean"] for key in metrics],
                             *[stats[key]["std"] for key in metrics]))
    (report_dir / "metric_note.txt").write_text(
        "val_* are unrefined outputs of the same jointly trained model; "
        "they are not an independently trained no-refinement model.\n"
    )
    print(f"E5 reports written to {report_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--gpus", type=gpu_ids, default="0,1,2,3")
    parser.add_argument("--smoke", action="store_true", help="Two train batches, one validation batch, one seed")
    parser.add_argument("--only", choices=VARIANTS, help="Run just one refinement variant")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    e4_group = None if args.smoke else ensure_e4_complete(args.rank, args.dim)
    base = RESULTS / ("smoke/E5" if args.smoke else "E5") / capacity_name(args.rank, args.dim)
    names = (args.only,) if args.only else VARIANTS
    if not args.summary_only:
        for name in names:
            group = base / name
            if not args.smoke and name == "full_bi":
                summary_for_group(group, source=e4_group, include_base=True)
                continue
            mhca, mhsa, mamba, bidirectional = VARIANTS[name]
            for seed in (SEEDS[:1] if args.smoke else SEEDS):
                train_one(group, seed, args.rank, args.dim, args.gpus,
                          modules=(mhca, mhsa, mamba), bidirectional=bidirectional,
                          smoke=args.smoke)
                if not args.smoke:
                    evaluate_one(group, seed, args.rank, args.dim, args.gpus,
                                 modules=(mhca, mhsa, mamba), bidirectional=bidirectional)
            if not args.smoke:
                summary_for_group(group, include_base=True)
    if not args.smoke:
        report_if_complete(base, e4_group)


if __name__ == "__main__":
    main()
