"""Build one class prototype from pretrained image features and score it.

Each prototype is the average feature of that class's training images. The
fixed internal validation images are excluded, then used only to measure
top-1 accuracy. No classifier weights are trained.

The output file records the backbone it was built with (``timm_checkpoint``);
training refuses to pair it with a different one. An existing output is never
overwritten, so each backbone needs its own folder.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cs604.data.dataset import (  # noqa: E402
    CS604ImageDataset,
    load_split_index,
    stratified_split,
    write_split_index,
)


def _loader(dataset, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


@torch.inference_mode()
def _features(model, images: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.forward_head(model.forward_features(images), pre_logits=True)
    return F.normalize(hidden.float(), dim=1)


def _run_loader(loader: DataLoader, consume, progress_every: int) -> None:
    started = time.perf_counter()
    seen = 0
    total = len(loader.dataset)
    for batch_index, batch in enumerate(loader, start=1):
        images, labels, _paths = batch
        consume(images, labels)
        seen += images.size(0)
        if batch_index % progress_every == 0 or seen == total:
            rate = seen / max(time.perf_counter() - started, 1e-9)
            print(f"{seen}/{total} images ({rate:.1f} img/s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, help="prototype .pt file; defaults to the config's prototype_output"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--per-class", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.output is None:
        if not cfg.get("prototype_output"):
            raise SystemExit("Pass --output or set prototype_output in the config")
        args.output = Path(cfg["prototype_output"])
    existing = [
        path for path in (args.output, args.output.with_suffix(".json"), args.output.parent / "splits")
        if path.exists()
    ]
    if existing:
        raise SystemExit(
            "Refusing to overwrite an existing prototype build: "
            + ", ".join(str(path) for path in existing)
        )
    from cs604.models.build import build_vit, model_data_config

    model = build_vit(
        checkpoint=cfg["timm_checkpoint"],
        num_classes=int(cfg["num_classes"]),
        mode="head",
        image_size=None if cfg.get("image_size") is None else int(cfg["image_size"]),
    ).cuda().eval()
    from timm.data import create_transform

    transform = create_transform(**model_data_config(model), is_training=False)
    samples = load_split_index(Path(cfg["data_root"]) / "train.txt")
    per_class = args.per_class if args.per_class is not None else cfg.get("prototype_per_class")
    train_samples, val_samples = stratified_split(
        samples,
        val_per_class=int(cfg.get("val_per_class", 5)),
        seed=int(cfg.get("split_seed", 0)),
        train_per_class=None if per_class is None else int(per_class),
    )
    split_dir = args.output.parent / "splits"
    write_split_index(split_dir / "prototype_train.txt", train_samples)
    write_split_index(split_dir / "internal_val.txt", val_samples)

    feature_size = model.num_features
    prototypes_sum = torch.zeros(int(cfg["num_classes"]), feature_size, device="cuda")
    counts = torch.zeros(int(cfg["num_classes"]), device="cuda")

    def accumulate(images, labels) -> None:
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        prototypes_sum.index_add_(0, labels, _features(model, images))
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))

    print(f"averaging {len(train_samples)} training images", flush=True)
    started = time.perf_counter()
    _run_loader(
        _loader(
            CS604ImageDataset(cfg["data_root"], "train", transform, train_samples),
            args.batch_size,
            args.num_workers,
        ),
        accumulate,
        progress_every=20,
    )
    if bool((counts == 0).any()):
        raise RuntimeError("At least one class has no prototype images")
    prototypes = F.normalize(prototypes_sum / counts[:, None], dim=1)

    correct = 0
    evaluated = 0

    def score(images, labels) -> None:
        nonlocal correct, evaluated
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        predictions = (_features(model, images) @ prototypes.T).argmax(dim=1)
        correct += int((predictions == labels).sum())
        evaluated += images.size(0)

    print(f"scoring {len(val_samples)} internal validation images", flush=True)
    _run_loader(
        _loader(
            CS604ImageDataset(cfg["data_root"], "internal_val", transform, val_samples),
            args.batch_size,
            args.num_workers,
        ),
        score,
        progress_every=20,
    )
    elapsed = time.perf_counter() - started
    accuracy = correct / max(1, evaluated)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prototypes": prototypes.cpu(),
            "samples_per_class": counts.cpu(),
            "timm_checkpoint": cfg["timm_checkpoint"],
            "split_seed": int(cfg.get("split_seed", 0)),
            "val_per_class": int(cfg.get("val_per_class", 5)),
        },
        args.output,
    )
    summary = {
        "timm_checkpoint": cfg["timm_checkpoint"],
        "train_images": len(train_samples),
        "val_images": evaluated,
        "internal_top1_accuracy": accuracy,
        "elapsed_sec": elapsed,
        "images_per_sec": (len(train_samples) + evaluated) / max(elapsed, 1e-9),
        "device": torch.cuda.get_device_name(0),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"internal top1={100 * accuracy:.2f}%")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
