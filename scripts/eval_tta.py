"""Evaluation-only test-time variants of a checkpoint on its internal holdout.

For each eval ``crop_pct`` the holdout is read once; every batch is scored as
is, horizontally flipped, and as the average of the two (mean logits and mean
softmax). The softmax average over every crop and flip is scored too. Nothing
is trained and the checkpoint is not modified.

The holdout list is the run's own ``splits/internal_val.txt``. The variant that
matches the training-time eval transform must reproduce the run's final
holdout accuracy; the script fails if it does not (``--gate-tolerance``).

Writes ``<run>/analysis/eval_tta.json`` (or ``--output``).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cs604.data.dataset import CS604ImageDataset, load_split_index  # noqa: E402
from cs604.engine.checkpoint_io import sha256_file  # noqa: E402
from cs604.models.build import build_from_config, load_model_state, model_data_config  # noqa: E402


def _final_holdout_accuracy(run_dir: Path) -> float | None:
    path = run_dir / "metrics.csv"
    if not path.is_file():
        return None
    rows = [row for row in csv.DictReader(path.open(encoding="utf-8")) if row.get("phase") == "val"]
    return float(rows[-1]["accuracy"]) if rows else None


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--crop-pct", type=float, nargs="+", default=[0.875, 0.9, 0.95, 1.0])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="first N holdout images (smoke test)")
    parser.add_argument("--gate-tolerance", type=float, default=0.0005)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    from timm.data import create_transform

    run_dir = args.checkpoint.parent
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model = build_from_config(cfg)
    load_model_state(model, checkpoint)
    model.cuda().eval()
    precision = str(cfg.get("precision", "bf16"))
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)

    samples = load_split_index(run_dir / "splits" / "internal_val.txt")
    if args.limit is not None:
        samples = samples[: args.limit]
    labels_all = torch.tensor([label for _path, label in samples], device="cuda")
    data_cfg = model_data_config(model)
    default_crop = float(data_cfg["crop_pct"])
    ensemble = torch.zeros(len(samples), int(cfg["num_classes"]), device="cuda", dtype=torch.float16)

    results: dict[str, dict] = {}
    for crop_pct in args.crop_pct:
        transform = create_transform(**{**data_cfg, "crop_pct": crop_pct}, is_training=False)
        dataset = CS604ImageDataset(cfg["data_root"], "internal_val", transform=transform, samples=samples)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        correct = {"plain": 0, "flip": 0, "mean_logits": 0, "mean_probs": 0}
        loss_sum = {"plain": 0.0, "mean_probs": 0.0}
        offset = 0
        started = time.perf_counter()
        for images, labels, _paths in loader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(images).float()
                flipped = model(torch.flip(images, dims=[3])).float()
            probs = (logits.softmax(1) + flipped.softmax(1)) / 2
            correct["plain"] += int((logits.argmax(1) == labels).sum())
            correct["flip"] += int((flipped.argmax(1) == labels).sum())
            correct["mean_logits"] += int(((logits + flipped).argmax(1) == labels).sum())
            correct["mean_probs"] += int((probs.argmax(1) == labels).sum())
            loss_sum["plain"] += float(F.cross_entropy(logits, labels, reduction="sum"))
            loss_sum["mean_probs"] += float(F.nll_loss(probs.clamp_min(1e-12).log(), labels, reduction="sum"))
            ensemble[offset: offset + len(labels)] += probs.half()
            offset += len(labels)
        count = offset
        key = f"crop_{crop_pct:g}"
        results[key] = {
            "crop_pct": crop_pct,
            **{f"acc_{name}": value / count for name, value in correct.items()},
            **{f"loss_{name}": value / count for name, value in loss_sum.items()},
            "elapsed_sec": time.perf_counter() - started,
        }
        print(json.dumps({key: results[key]}), flush=True)

    ensemble_acc = float((ensemble.argmax(1) == labels_all).float().mean())
    results["all_crops_flip_mean_probs"] = {"crop_pct": args.crop_pct, "acc": ensemble_acc}
    print(f"all crops + flip, mean softmax: {ensemble_acc:.5f}", flush=True)

    reference = _final_holdout_accuracy(run_dir)
    gate_key = f"crop_{default_crop:g}"
    gate = None
    if args.limit is None and reference is not None and gate_key in results:
        reproduced = results[gate_key]["acc_plain"]
        gate = {"reference": reference, "reproduced": reproduced, "default_crop_pct": default_crop}
        if abs(reproduced - reference) > args.gate_tolerance:
            raise RuntimeError(f"gate failed: {reproduced:.5f} vs run's final holdout {reference:.5f}")

    from cs604.engine.loop import _git_metadata

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": checkpoint.get("step"),
        "samples": len(samples),
        "data_config": {k: list(v) if isinstance(v, tuple) else v for k, v in data_cfg.items()},
        "gate": gate,
        "git": _git_metadata(),
        "device": torch.cuda.get_device_name(0),
        "results": results,
    }
    output = args.output or run_dir / "analysis" / ("eval_tta.json" if args.limit is None else "eval_tta_smoke.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
