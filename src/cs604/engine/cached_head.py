"""Train the classifier head on cached frozen-backbone features.

Equivalent to ``scripts/train.py`` with ``mode: head`` except that the backbone
forward pass is replaced by a lookup in the feature cache (``feature_cache.py``).
It reuses the training code's pieces so results transfer:

- the head: ``build_feature_head`` (same classes and options as ``build_from_config``),
  initialized by ``apply_class_prototypes`` after ``check_prototype_split``;
- the optimizer: AdamW over ``optimizer_param_groups``, with ``lr``, ``head_lr``
  and ``weight_decay`` as in the loop;
- the schedule: ``cosine_lr_factor`` (5% linear warmup, cosine to zero), stepped
  after every update, over ``max_steps`` or ``epochs`` x steps per epoch;
- the batch order: a ``DataLoader`` over row indices shuffled with the loop's
  generator seed (``seed + epoch``), so every batch holds the same images as in
  an end-to-end run with the same seed and batch size;
- the step-zero gate: ``assess_initial_accuracy`` on the cached holdout.

What differs: each epoch reads one cached view per image instead of a fresh
augmentation. ``train_groups`` lists the cache groups to draw from; each image
cycles through a random permutation of the available views, so up to as many
epochs as there are views every epoch sees a different view of each image.
Training features are stored in float16 (the holdout in float32).
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from cs604.engine.baseline import assess_initial_accuracy, check_prototype_split
from cs604.engine.budget import cosine_lr_factor
from cs604.engine.feature_cache import group_view_names, load_view_features, read_manifest
from cs604.models.build import apply_class_prototypes, build_feature_head, optimizer_param_groups


class _Rows(Dataset):
    def __init__(self, rows: int):
        self.rows = rows

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, index: int) -> int:
        return index


def epoch_batches(rows: int, batch_size: int, seed: int, epoch: int) -> DataLoader:
    """Row-index batches in the order scripts/train.py's loader visits images."""

    generator = torch.Generator().manual_seed(seed + epoch)
    return DataLoader(_Rows(rows), batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)


def view_schedule(rows: int, views: int, epochs: int, seed: int) -> torch.Tensor:
    """``[epochs, rows]`` view index per epoch: per image, cycles of random permutations."""

    generator = torch.Generator().manual_seed(seed)
    cycles = math.ceil(epochs / views)
    columns = [torch.rand(rows, views, generator=generator).argsort(dim=1) for _ in range(cycles)]
    return torch.cat(columns, dim=1)[:, :epochs].T.contiguous()


class CachedFeatures:
    """The holdout and selected training views of one feature cache, on the GPU."""

    def __init__(self, cache_dir: str | Path, train_groups: list[str], device: torch.device):
        self.cache_dir = Path(cache_dir)
        self.manifest = read_manifest(self.cache_dir)
        self.device = device
        self.train_groups = list(train_groups)
        holdout = load_view_features(self.cache_dir, self.manifest, "holdout", "view_00")
        self.holdout = torch.from_numpy(np.array(holdout)).to(device)
        self.holdout_labels = torch.from_numpy(np.load(self.cache_dir / "labels_holdout.npy")).to(device)
        self.train_labels = torch.from_numpy(np.load(self.cache_dir / "labels_train.npy")).to(device)
        self.view_names = [
            f"{group}/{name}" for group in self.train_groups
            for name in group_view_names(self.manifest, group)
        ]
        if not self.view_names:
            raise ValueError("No training views selected")
        self.train: torch.Tensor | None = None  # [views, rows, dim], filled one view at a time
        for number, full_name in enumerate(self.view_names):
            group, name = full_name.split("/")
            array = load_view_features(self.cache_dir, self.manifest, group, name)
            if self.train is None:
                self.train = torch.empty(
                    (len(self.view_names), *array.shape),
                    dtype=getattr(torch, str(array.dtype)), device=device,
                )
            self.train[number].copy_(torch.from_numpy(np.array(array)))
        if self.train.shape[1] != len(self.train_labels) or len(self.holdout) != len(self.holdout_labels):
            raise ValueError("Feature rows and labels differ in length")

    @property
    def feature_dim(self) -> int:
        return int(self.train.shape[-1])


@torch.no_grad()
def evaluate_features(head, features: torch.Tensor, labels: torch.Tensor, amp_dtype, chunk: int = 4096) -> dict:
    """Holdout top-1 and mean cross-entropy, as the loop's ``evaluate`` computes them."""

    head.eval()
    correct = 0
    total_loss = 0.0
    for start in range(0, len(features), chunk):
        batch = features[start:start + chunk].float()
        target = labels[start:start + chunk]
        with torch.autocast(batch.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = head(batch)
            loss = F.cross_entropy(logits, target, reduction="sum")
        total_loss += float(loss)
        correct += int((logits.argmax(dim=1) == target).sum())
    head.train()
    return {"accuracy": correct / len(features), "loss": total_loss / len(features)}


def _log_scale(head) -> float | None:
    log_scale = getattr(head.get_classifier(), "log_scale", None)
    return None if log_scale is None else float(log_scale.detach().float().exp())


def train_cached_head(cfg: dict, data: CachedFeatures, log=print) -> dict:
    """Train one head on cached features; returns settings, per-epoch curve and results."""

    if str(cfg.get("mode", "head")) != "head":
        raise ValueError("Cached features need mode: head (frozen backbone)")
    if int(cfg.get("grad_accum", 1)) != 1:
        raise ValueError("grad_accum is not supported on cached features")
    if data.manifest["timm_checkpoint"] != cfg["timm_checkpoint"]:
        raise ValueError("The feature cache was built with a different backbone")
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    device = data.device

    head = build_feature_head(cfg, data.feature_dim).to(device)
    blob = torch.load(cfg["prototype_file"], map_location="cpu", weights_only=False)
    check_prototype_split({**cfg, "evaluate_before_training": True}, blob)
    apply_class_prototypes(head, blob["prototypes"] if isinstance(blob, dict) else blob)

    head_lr = None if cfg.get("head_lr") is None else float(cfg["head_lr"])
    optimizer = torch.optim.AdamW(
        optimizer_param_groups(head, float(cfg["lr"]), head_lr),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.05)),
    )
    rows = int(data.train.shape[1])
    batch_size = int(cfg["batch_size"])
    epochs = int(cfg.get("epochs", 1))
    steps_per_epoch = math.ceil(rows / batch_size)
    max_steps = None if cfg.get("max_steps") is None else int(cfg["max_steps"])
    total_steps = max_steps or steps_per_epoch * epochs
    warmup_steps = int(cfg.get("warmup_steps", max(1, round(total_steps * 0.05))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_lr_factor(step, warmup_steps, total_steps)
    )
    precision = str(cfg.get("precision", "bf16"))
    amp_dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if precision == "fp16" else None)
    label_smoothing = float(cfg.get("label_smoothing", 0.0))
    clip = cfg.get("clip_grad_norm")
    trainable = [p for p in head.parameters() if p.requires_grad]

    started = time.perf_counter()
    initial = evaluate_features(head, data.holdout, data.holdout_labels, amp_dtype)
    gate = assess_initial_accuracy(initial["accuracy"], cfg)
    log(f"step 0: holdout top1={100 * initial['accuracy']:.3f}% loss={initial['loss']:.4f}")
    if not gate["passed"]:
        raise ValueError(f"Step-zero gate failed: {gate}")

    schedule = view_schedule(rows, data.train.shape[0], epochs, int(cfg.get("view_seed", seed))).to(device)
    curve = []
    step = 0
    best = {"accuracy": initial["accuracy"], "epoch": 0, "step": 0}
    head.train()
    for epoch in range(epochs):
        views = schedule[epoch]
        loss_sum = torch.zeros((), device=device)
        updates = 0
        for indices in epoch_batches(rows, batch_size, seed, epoch):
            indices = indices.to(device, non_blocking=True)
            features = data.train[views[indices], indices].float()
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = F.cross_entropy(head(features), data.train_labels[indices], label_smoothing=label_smoothing)
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(trainable, float(clip))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            updates += 1
            loss_sum += loss.detach()
            if max_steps is not None and step >= max_steps:
                break
        result = evaluate_features(head, data.holdout, data.holdout_labels, amp_dtype)
        row = {
            "epoch": epoch + 1, "step": step, "train_loss": float(loss_sum) / max(1, updates),
            "accuracy": result["accuracy"], "loss": result["loss"], "scale": _log_scale(head),
            "elapsed_sec": time.perf_counter() - started,
        }
        curve.append(row)
        log(f"epoch {epoch + 1} step {step}: train loss {row['train_loss']:.4f} "
            f"holdout top1={100 * row['accuracy']:.3f}% loss={row['loss']:.4f}"
            + ("" if row["scale"] is None else f" scale={row['scale']:.2f}"))
        if result["accuracy"] > best["accuracy"]:
            best = {"accuracy": result["accuracy"], "epoch": epoch + 1, "step": step}
        if max_steps is not None and step >= max_steps:
            break
    return {
        "initial_accuracy": initial["accuracy"],
        "initial_loss": initial["loss"],
        "final_accuracy": curve[-1]["accuracy"] if curve else initial["accuracy"],
        "final_loss": curve[-1]["loss"] if curve else initial["loss"],
        "best_accuracy": best["accuracy"],
        "best_epoch": best["epoch"],
        "final_scale": _log_scale(head),
        "optimizer_steps": step,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "train_views": data.view_names,
        "elapsed_sec": time.perf_counter() - started,
        "curve": curve,
        "head": head,
    }
