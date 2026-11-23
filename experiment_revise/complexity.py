"""Measure model parameters and THOP-estimated FLOPs on one fixed AV2 sample."""

import json
import os
import sys

import torch
from thop import profile

from common import PROJECT

sys.path.insert(0, str(PROJECT))
from src.datamodule.av2_dataset import Av2Dataset, collate_fn  # noqa: E402
from src.model.model_forecast import ModelForecast  # noqa: E402


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def measure(group, rank, dim, offset, gpu_id):
    output = group / "summary" / "complexity.json"
    if output.exists():
        return json.loads(output.read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    torch.cuda.set_device(0)
    model = ModelForecast(embed_dim=dim, use_low_rank=offset == "lowrank",
                          low_rank_dim=rank, offset_mode=offset).eval().cuda()
    dataset = Av2Dataset(PROJECT / "data" / "DeMo_processed", split="val")
    files = sorted((PROJECT / "eval_bench_data" / "flops_data").glob("*.pt"))
    if len(files) != 1:
        raise RuntimeError("Expected exactly one fixed FLOPs input sample")
    dataset.file_list = files
    batch = collate_fn([dataset[0]])
    batch = batch[0] if isinstance(batch, list) else batch
    batch = _to_device(batch, "cuda")
    with torch.no_grad():
        macs, _ = profile(model, inputs=(batch,), verbose=False)
    # THOP does not count the adapter's functional matmul or elementwise ops.
    offset_flops = 0 if offset == "global" else 2 * 6 * 60 * dim
    if offset == "lowrank":
        offset_flops += 2 * 6 * 60 * rank * dim
    result = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "flops_g": (2 * macs + offset_flops) / 1e9,
        "offset_flops": offset_flops,
        "method": "THOP MACs multiplied by 2 plus analytical adapter FLOPs; custom Mamba kernels may be undercounted",
        "sample": str(files[0].relative_to(PROJECT)),
        "gpu_id": gpu_id,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result
