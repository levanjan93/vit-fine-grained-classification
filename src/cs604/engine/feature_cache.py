"""Cached frozen-backbone features for fast head-only experiments.

With a frozen backbone (``mode: head``) the classifier only ever sees the
backbone's pre-logits feature of each (augmented) image. Caching those features
once lets a head be trained in seconds per epoch instead of minutes, with the
same model code (see ``cached_head.py``).

Layout under ``cache_dir``::

    manifest.json              backbone, split hashes, every view's settings
    splits/internal_train.txt  the training images, in cache row order
    splits/internal_val.txt    the fixed holdout, in cache row order
    labels_train.npy           int64 labels, one per training row
    labels_holdout.npy         int64 labels, one per holdout row
    <group>/view_NN/features.npy  [rows, feature_dim] features of one view

Groups: ``holdout`` and ``clean`` (one eval-transform view each) and augmented
groups such as ``aug_default`` (several views with the training transform).

A view is one pass over a split with one transform. Augmented views are
reproducible: before transforming image ``index`` of a view with seed ``s``,
Python's ``random`` (timm's RandomResizedCrop) and torch's RNG (flip, color
jitter) are both seeded with ``s * 10_000_000 + index``, so a view does not
depend on the number of workers or the batch order.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

CACHE_FORMAT = 1
SEED_STRIDE = 10_000_000


def per_sample_seed(view_seed: int, index: int) -> int:
    if not 0 <= index < SEED_STRIDE:
        raise ValueError(f"index {index} outside [0, {SEED_STRIDE})")
    return int(view_seed) * SEED_STRIDE + int(index)


class SeededViewDataset(Dataset):
    """Wraps an image dataset so each item's random transform is seeded by its index."""

    def __init__(self, dataset: Dataset, view_seed: int | None):
        self.dataset = dataset
        self.view_seed = view_seed

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        if self.view_seed is not None:
            seed = per_sample_seed(self.view_seed, index)
            random.seed(seed)
            torch.manual_seed(seed)
        image, label, _path = self.dataset[index]
        return image, label, index


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def view_name(number: int) -> str:
    return f"view_{number:02d}"


def view_dir(cache_dir: Path, group: str, view: str) -> Path:
    return Path(cache_dir) / group / view


def read_manifest(cache_dir: Path) -> dict:
    path = Path(cache_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"No feature cache manifest at {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != CACHE_FORMAT:
        raise ValueError(f"Unsupported feature cache format {manifest.get('format')}")
    return manifest


def write_json_atomic(path: Path, payload: dict) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def group_view_names(manifest: dict, group: str) -> list[str]:
    """The finished views of a group, in name order (``["view_00", ...]``)."""

    views = manifest.get("views", {}).get(group)
    if not views:
        raise KeyError(f"Feature cache has no finished views in group {group!r}")
    return sorted(views)


def load_view_features(cache_dir: Path, manifest: dict, group: str, view: str) -> np.ndarray:
    """A read-only memory map of one finished view, checked against the manifest."""

    entry = manifest["views"][group][view]
    path = view_dir(cache_dir, group, view) / "features.npy"
    features = np.load(path, mmap_mode="r")
    if list(features.shape) != entry["shape"] or str(features.dtype) != entry["dtype"]:
        raise ValueError(
            f"{path}: shape/dtype {features.shape}/{features.dtype} differ from the manifest"
        )
    return features
