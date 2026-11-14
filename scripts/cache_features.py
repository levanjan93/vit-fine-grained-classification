"""Cache frozen-backbone features of the training split and the fixed holdout.

Writes the feature cache described in ``src/cs604/engine/feature_cache.py``:
one eval-transform view of the holdout and of the training images, and seeded
augmented views made with exactly the training transform (``build_transforms``,
the function ``scripts/train.py`` uses). Finished views are skipped, so an
interrupted run continues where it stopped (at most one view is redone).

The split is recomputed from ``train.txt`` and must match the prototype build's
split files byte for byte (``reference_splits`` in the config).

    python scripts/cache_features.py --config configs/experiments/feature_cache_orig_in21k.yaml --groups holdout clean
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
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
from cs604.engine.baseline import LEGACY_PROTOTYPE_BACKBONE  # noqa: E402
from cs604.engine.feature_cache import (  # noqa: E402
    CACHE_FORMAT,
    SEED_STRIDE,
    SeededViewDataset,
    sha256_file,
    view_dir,
    view_name,
    write_json_atomic,
)
from cs604.engine.loop import (  # noqa: E402
    TRAIN_TRANSFORM_KEYS,
    _git_metadata,
    _transform_crop_scale,
    build_transforms,
)
from cs604.models.build import build_vit, model_data_config  # noqa: E402


def _power_reading() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.limit,power.draw,pstate", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@torch.inference_mode()
def _features(model, images: torch.Tensor) -> torch.Tensor:
    # The classifier input in scripts/train.py: bf16 backbone, pre-logits token.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model.forward_head(model.forward_features(images), pre_logits=True).float()


def _write_view(model, dataset, path: Path, dtype: str, batch_size: int, workers: int) -> dict:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=False,
        prefetch_factor=4 if workers > 0 else None,
    )
    rows, dim = len(dataset), model.num_features
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.dtype(dtype), shape=(rows, dim))
    written = np.zeros(rows, dtype=bool)
    started = time.perf_counter()
    seen = 0
    power = "unknown"
    for batch_index, (images, _labels, indices) in enumerate(loader, start=1):
        features = _features(model, images.cuda(non_blocking=True)).cpu().numpy()
        indices = indices.numpy()
        out[indices] = features.astype(out.dtype)
        written[indices] = True
        seen += len(indices)
        if batch_index == 50:
            power = _power_reading()  # under load, for timing claims
        if batch_index % 100 == 0 or seen == rows:
            rate = seen / max(time.perf_counter() - started, 1e-9)
            print(f"  {seen}/{rows} images ({rate:.1f} img/s)", flush=True)
    elapsed = time.perf_counter() - started
    if not written.all():
        raise RuntimeError(f"{path}: {int((~written).sum())} rows were not written")
    out.flush()
    del out
    gc.collect()
    os.replace(tmp, path)
    return {
        "shape": [rows, dim],
        "dtype": dtype,
        "sha256": sha256_file(path),
        "elapsed_sec": elapsed,
        "images_per_sec": rows / max(elapsed, 1e-9),
        "batch_size": batch_size,
        "num_workers": workers,
        "gpu_power_under_load": power,
    }


def _prototype_check(cache_dir: Path, cfg: dict, labels: np.ndarray, holdout: Path) -> dict:
    """Nearest-prototype top-1 on the cached holdout, as stored and cast to fp16."""

    blob = torch.load(cfg["prototype_file"], map_location="cpu", weights_only=False)
    if blob.get("timm_checkpoint", LEGACY_PROTOTYPE_BACKBONE) != cfg["timm_checkpoint"]:
        raise ValueError("prototype_file was built with a different backbone")
    prototypes = F.normalize(blob["prototypes"].float().cuda(), dim=1)
    features = torch.from_numpy(np.load(holdout)).cuda()
    target = torch.from_numpy(labels).cuda()
    result = {}
    for name, feats in (("stored", features.float()), ("fp16", features.half().float())):
        predictions = (F.normalize(feats, dim=1) @ prototypes.T).argmax(dim=1)
        result[f"prototype_top1_{name}"] = float((predictions == target).float().mean())
    result["prototype_file_sha256"] = sha256_file(Path(cfg["prototype_file"]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", help="view groups to build (default: all in the config)")
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cache_dir = Path(cfg["cache_dir"])
    groups = args.groups or list(cfg["views"])
    unknown = [group for group in groups if group not in cfg["views"]]
    if unknown:
        raise SystemExit(f"Unknown view groups: {unknown}")
    workers = args.num_workers if args.num_workers is not None else int(cfg.get("num_workers", 8))

    torch.backends.cudnn.benchmark = True
    model = build_vit(
        checkpoint=cfg["timm_checkpoint"],
        num_classes=int(cfg["num_classes"]),
        mode="head",
        image_size=None if cfg.get("image_size") is None else int(cfg["image_size"]),
    ).cuda().eval()
    train_transform, eval_transform = build_transforms(cfg, model)

    # The split, in the same order scripts/train.py uses (train_per_class null = all 40).
    samples = load_split_index(Path(cfg["data_root"]) / "train.txt")
    train_samples, holdout_samples = stratified_split(
        samples, val_per_class=int(cfg.get("val_per_class", 5)),
        seed=int(cfg.get("split_seed", 0)), train_per_class=None,
    )
    split_dir = cache_dir / "splits"
    split_files = {"train": split_dir / "internal_train.txt", "holdout": split_dir / "internal_val.txt"}
    for name, rows in (("train", train_samples), ("holdout", holdout_samples)):
        write_split_index(split_files[name], rows)
    split_hashes = {name: sha256_file(path) for name, path in split_files.items()}
    for name, reference in cfg["reference_splits"].items():
        if sha256_file(Path(reference)) != split_hashes[name]:
            raise SystemExit(f"The {name} split differs from {reference}; refusing to cache")
    labels = {
        "train": np.array([label for _path, label in train_samples], dtype=np.int64),
        "holdout": np.array([label for _path, label in holdout_samples], dtype=np.int64),
    }
    for name, values in labels.items():
        np.save(cache_dir / f"labels_{name}.npy", values)

    manifest_path = cache_dir / "manifest.json"
    header = {
        "format": CACHE_FORMAT,
        "timm_checkpoint": cfg["timm_checkpoint"],
        "image_size": cfg.get("image_size"),
        "feature": "model.forward_head(model.forward_features(x), pre_logits=True), bf16 autocast backbone",
        "feature_dim": int(model.num_features),
        "data_config": {key: list(v) if isinstance(v, tuple) else v for key, v in model_data_config(model).items()},
        "eval_transform": repr(eval_transform),
        "splits": {
            name: {"file": str(split_files[name]), "sha256": split_hashes[name], "rows": len(labels[name])}
            for name in split_files
        },
        "reference_splits": cfg["reference_splits"],
        "split_seed": int(cfg.get("split_seed", 0)),
        "val_per_class": int(cfg.get("val_per_class", 5)),
        "seed_rule": f"random.seed and torch.manual_seed(view_seed * {SEED_STRIDE} + row) before each augmented image",
        "config_file": args.config.as_posix(),
        "versions": {"torch": torch.__version__, "timm": __import__("timm").__version__},
    }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("format", "timm_checkpoint", "image_size", "feature_dim", "data_config", "splits"):
            if manifest.get(key) != header[key]:
                raise SystemExit(f"Existing cache has a different {key}; use a new cache_dir")
    else:
        manifest = {**header, "views": {}, "groups": {}}
    write_json_atomic(manifest_path, manifest)

    for group in groups:
        spec = cfg["views"][group]
        split = spec["split"]
        rows = train_samples if split == "train" else holdout_samples
        if spec["transform"] == "train":
            overrides = {key: spec[key] for key in (*TRAIN_TRANSFORM_KEYS, "train_crop_scale") if key in spec}
            transform = build_transforms({**cfg, **overrides}, model)[0] if overrides else train_transform
            base_seed = int(spec["seed"])
        elif spec["transform"] == "eval":
            transform, base_seed, overrides = eval_transform, None, {}
        else:
            raise SystemExit(f"{group}: transform must be 'train' or 'eval'")
        group_info = {
            "split": split,
            "transform": spec["transform"],
            "transform_overrides": overrides,
            "transform_repr": repr(transform),
            "train_crop_scale": _transform_crop_scale(transform) if spec["transform"] == "train" else None,
            "count": int(spec.get("count", 1)),
        }
        previous = manifest["groups"].get(group)
        if previous and previous != group_info:
            raise SystemExit(f"{group}: settings differ from the existing cache; use a new group name")
        manifest["groups"][group] = group_info
        for number in range(int(spec.get("count", 1))):
            name = view_name(number)
            path = view_dir(cache_dir, group, name) / "features.npy"
            done = manifest["views"].get(group, {}).get(name)
            if done and path.is_file():
                print(f"{group}/{name}: already cached", flush=True)
                continue
            view_seed = None if base_seed is None else base_seed + number
            print(f"{group}/{name}: {len(rows)} {split} images, seed {view_seed}", flush=True)
            dataset = SeededViewDataset(
                CS604ImageDataset(cfg["data_root"], "train", transform, rows), view_seed
            )
            entry = _write_view(
                model, dataset, path, str(spec.get("dtype", "float16")),
                int(spec.get("batch_size", cfg.get("batch_size", 256))), workers,
            )
            entry.update({"view_seed": view_seed, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                          "device": torch.cuda.get_device_name(0), **_git_metadata()})
            if group == "holdout":
                entry.update(_prototype_check(cache_dir, cfg, labels["holdout"], path))
                print(f"  prototype top-1 on cached holdout: {entry['prototype_top1_stored']:.5f} "
                      f"(fp16: {entry['prototype_top1_fp16']:.5f})", flush=True)
            manifest["views"].setdefault(group, {})[name] = entry
            write_json_atomic(manifest_path, manifest)
            print(f"  wrote {path} in {entry['elapsed_sec']:.0f}s "
                  f"({entry['images_per_sec']:.1f} img/s, GPU {entry['gpu_power_under_load']})", flush=True)


if __name__ == "__main__":
    main()
