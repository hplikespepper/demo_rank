"""Shared runner and reporting helpers for the E3–E5 experiments."""

import csv
import errno
import json
import os
import pty
import shutil
import subprocess
import sys
import termios
import fcntl
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
RESULTS = ROOT / "results"
SEEDS = (42, 43, 44, 45, 46)
PRIMARY_METRICS = ("val_new_minADE6", "val_new_minFDE6", "val_new_MR")
BASE_METRICS = ("val_minADE6", "val_minFDE6", "val_MR")


def capacity_name(rank, dim):
    return f"r{rank:02d}_d{dim:03d}"


def e4_name(rank, dim):
    if dim == 128 and rank in (8, 16, 32, 48, 64):
        return capacity_name(rank, dim)
    if rank in (16, 32) and dim in (64, 128, 256):
        return capacity_name(rank, dim)
    raise ValueError("E3/E5 capacity must be one of the E4-tested r,D pairs")


def seed_dir(group, seed):
    return group / f"seed_{seed}"


def model_overrides(rank, dim, offset="lowrank", modules=(True, True, True), bidirectional=True):
    flags = ("refine_mhca", "refine_mhsa", "refine_mamba")
    values = [
        f"model.target.model.low_rank_dim={rank}",
        f"model.target.model.embed_dim={dim}",
        f"model.target.model.use_low_rank={str(offset == 'lowrank').lower()}",
        f"model.target.model.offset_mode={offset}",
        f"model.target.model.refine_mamba_bidirectional={str(bidirectional).lower()}",
    ]
    values.extend(f"model.target.model.{key}={str(value).lower()}" for key, value in zip(flags, modules))
    return values


def best_checkpoint(train_dir):
    files = sorted((train_dir / "checkpoints").glob("best-epoch=*.ckpt"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one best checkpoint in {train_dir}, found {len(files)}")
    return files[0]


def resume_checkpoint(train_dir):
    checkpoints = train_dir / "checkpoints"
    last = checkpoints / "last.ckpt"
    if last.is_file() and last.stat().st_size:
        return last
    # A machine can stop while last.ckpt is being created. The most recent
    # retained epoch still contains optimizer and scheduler state.
    candidates = []
    for path in checkpoints.glob("epoch=*.ckpt"):
        try:
            epoch = int(path.name.split("=", 1)[1].split("-", 1)[0].split(".", 1)[0])
        except ValueError:
            continue
        candidates.append((epoch, path))
    return max(candidates)[1] if candidates else None


def metrics_from_file(path):
    metrics = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        try:
            metrics[name.strip()] = float(raw.strip())
        except ValueError:
            continue
    return metrics


def _run(cmd, directory, gpu_ids):
    directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    env["PYTHONUNBUFFERED"] = "1"
    log_path = directory / "console.txt"
    print(f"Live output and log: {log_path}", flush=True)
    terminal = sys.stdout.isatty()
    master = slave = None
    try:
        if terminal:
            master, slave = pty.openpty()
            try:
                size = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, b"\0" * 8)
                fcntl.ioctl(slave, termios.TIOCSWINSZ, size)
            except OSError:
                pass
        with log_path.open("ab") as log:
            log.write(f"\n=== Started {datetime.now().astimezone().isoformat()} ===\n".encode())
            log.flush()
            with subprocess.Popen(cmd, cwd=PROJECT, env=env,
                                  stdout=slave if terminal else subprocess.PIPE,
                                  stderr=subprocess.STDOUT) as process:
                if slave is not None:
                    os.close(slave)
                    slave = None
                stream = master if terminal else process.stdout.fileno()
                while True:
                    try:
                        chunk = os.read(stream, 65536)
                    except OSError as error:
                        # A PTY reports EIO after the child closes its end.
                        if terminal and error.errno == errno.EIO:
                            break
                        raise
                    if not chunk:
                        break
                    view = memoryview(chunk)
                    while view:
                        view = view[os.write(sys.stdout.fileno(), view):]
                    log.write(chunk)
                    log.flush()
                returncode = process.wait()
    finally:
        if slave is not None:
            os.close(slave)
        if master is not None:
            os.close(master)
    if returncode:
        raise RuntimeError(f"Command failed ({returncode}); see {log_path}")


def _archive_incomplete(directory):
    if not directory.exists():
        return
    archive_root = directory.parent / f"{directory.name}_attempts"
    archive_root.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = archive_root / timestamp
    suffix = 1
    while destination.exists():
        destination = archive_root / f"{timestamp}_{suffix}"
        suffix += 1
    shutil.move(str(directory), str(destination))
    print(f"Archived incomplete attempt: {destination}", flush=True)


def train_one(group, seed, rank, dim, gpu_ids, offset="lowrank", modules=(True, True, True),
              bidirectional=True, smoke=False):
    run = seed_dir(group, seed)
    train_dir = run / "train"
    record = train_dir / "train_time.txt"
    if record.exists() and "status: completed" in record.read_text():
        if smoke or len(list((train_dir / "checkpoints").glob("best-epoch=*.ckpt"))) == 1:
            return run
    checkpoint = None
    if train_dir.exists():
        checkpoint = resume_checkpoint(train_dir)
        if checkpoint:
            print(f"Resuming {group.name} seed {seed} from {checkpoint}", flush=True)
        else:
            print(f"No complete epoch checkpoint for {group.name} seed {seed}; restarting this seed", flush=True)
            _archive_incomplete(train_dir)
    gpu_count = len(gpu_ids.split(","))
    cmd = [
        sys.executable, "train.py", f"hydra.run.dir={train_dir}", f"seed={seed}",
        f"gpus={gpu_count}", "num_workers=4", "batch_size=16",
        "monitor=val_new_minFDE6", f"epochs={1 if smoke else 60}",
        f"save_top_k={0 if smoke else 10}",
        *model_overrides(rank, dim, offset, modules, bidirectional),
    ]
    if smoke:
        cmd += ["limit_train_batches=2", "limit_val_batches=1"]
    if checkpoint:
        cmd.append("checkpoint=" + str(checkpoint).replace("=", "\\="))
    print(f"Training {group.name} seed {seed} ({'smoke' if smoke else 'full'})", flush=True)
    _run(cmd, train_dir, gpu_ids)
    if not record.exists() or "status: completed" not in record.read_text():
        raise RuntimeError(f"Training did not finish cleanly: {train_dir}")
    if not smoke:
        best_checkpoint(train_dir)
    return run


def evaluate_one(group, seed, rank, dim, gpu_ids, offset="lowrank", modules=(True, True, True),
                 bidirectional=True):
    run = seed_dir(group, seed)
    eval_dir = run / "eval"
    summary = eval_dir / "summary_metrics.txt"
    if summary.exists() and all(k in metrics_from_file(summary) for k in PRIMARY_METRICS):
        return summary
    if eval_dir.exists():
        _archive_incomplete(eval_dir)
    checkpoint = best_checkpoint(run / "train")
    checkpoint_override = "checkpoint=" + str(checkpoint).replace("=", "\\=")
    cmd = [
        sys.executable, "eval.py", f"hydra.run.dir={eval_dir}",
        checkpoint_override, f"seed={seed}", "gpus=1",
        *model_overrides(rank, dim, offset, modules, bidirectional),
    ]
    print(f"Evaluating {group.name} seed {seed}: {checkpoint.name}", flush=True)
    _run(cmd, eval_dir, gpu_ids.split(",")[0])
    if not summary.exists():
        raise RuntimeError(f"Missing validation summary: {summary}")
    return summary


def summary_for_group(group, source=None, include_base=False):
    source = source or group
    keys = PRIMARY_METRICS + (BASE_METRICS if include_base else ())
    rows = []
    for seed in SEEDS:
        run = seed_dir(source, seed)
        path = run / "eval" / "summary_metrics.txt"
        if not path.exists():
            raise RuntimeError(f"Five-seed summary requires {path}")
        metrics = metrics_from_file(path)
        missing = [key for key in keys if key not in metrics]
        if missing:
            raise RuntimeError(f"Missing {missing} in {path}")
        row = {"seed": seed, **{key: metrics[key] for key in keys}}
        parameter_path = run / "train" / "parameters.txt"
        if parameter_path.exists():
            for line in parameter_path.read_text().splitlines():
                if line.startswith("Total parameters:"):
                    row["parameters"] = int(line.split(":", 1)[1].strip().replace(",", ""))
        rows.append(row)
    out = group / "summary"
    out.mkdir(parents=True, exist_ok=True)
    columns = ["seed", *keys]
    if all("parameters" in row for row in rows):
        columns.append("parameters")
    with (out / "per_seed.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (out / "mean_std.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("metric", "mean", "std", "count"))
        for key in columns[1:]:
            values = [row[key] for row in rows]
            writer.writerow((key, mean(values), stdev(values), len(values)))
    if source != group:
        (out / "reference.json").write_text(json.dumps({"source": str(source.relative_to(ROOT))}, indent=2) + "\n")
    return out


def ensure_e4_complete(rank, dim):
    name = e4_name(rank, dim)
    report_dir = RESULTS / "E4" / "summary"
    if not all((report_dir / name).exists() for name in
               ("rank_sensitivity.csv", "dimension_sensitivity.csv")):
        raise RuntimeError("Complete the E4 sweep and inspect its reports before full E3/E5 runs")
    group = RESULTS / "E4" / name
    for seed in SEEDS:
        if not (seed_dir(group, seed) / "eval" / "summary_metrics.txt").exists():
            raise RuntimeError(f"E4 has not finished five seeds for {capacity_name(rank, dim)}")
    summary_for_group(group, include_base=True)
    return group


def gpu_ids(value):
    parts = value.split(",")
    if (len(parts) != 4 or not all(p.strip().isdigit() for p in parts)
            or len(set(parts)) != 4):
        raise ValueError("Supply four distinct GPU IDs, e.g. 0,1,2,3")
    return ",".join(part.strip() for part in parts)
