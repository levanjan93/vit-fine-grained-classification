"""Checkpoint inference and strict CodaBench submission validation."""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cs604.data.dataset import CS604ImageDataset, load_split_index
from cs604.engine.checkpoint_io import sha256_file
from cs604.models.build import build_from_config, load_model_state


def validate_submission(
    submission_path: str | Path,
    expected_index: str | Path,
    num_classes: int = 10_000,
) -> None:
    expected = [path for path, _label in load_split_index(Path(expected_index))]
    lines = Path(submission_path).read_text(encoding="utf-8").splitlines()
    if len(lines) != len(expected):
        raise ValueError(f"Expected {len(expected)} predictions, found {len(lines)}")
    for number, (line, expected_path) in enumerate(zip(lines, expected), start=1):
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Line {number} must contain image_path and class_id")
        rel_path, raw_label = parts
        if rel_path != expected_path:
            raise ValueError(
                f"Line {number} path is {rel_path!r}; expected {expected_path!r}"
            )
        try:
            label = int(raw_label)
        except ValueError as error:
            raise ValueError(f"Line {number} has a non-integer class ID") from error
        if not 0 <= label < num_classes:
            raise ValueError(f"Line {number} class ID {label} is outside the valid range")


@torch.inference_mode()
def predict_split(
    checkpoint_path: str | Path,
    data_root: str | Path,
    split: str,
    output_path: str | Path,
    batch_size: int = 128,
    num_workers: int = 4,
    make_zip: bool = True,
    crop_pct: float | None = None,
    tta_flip: bool = False,
    check_fingerprint: bool = True,
    zoom: bool = False,
    zoom_mass: float | None = None,
) -> Path:
    """Predict ``split`` with the training-time eval transform.

    ``crop_pct`` overrides the checkpoint's center-crop fraction; ``tta_flip`` adds a
    horizontally flipped pass and averages the softmax of both (two forward passes,
    as ``scripts/eval_tta.py`` scores ``mean_probs``). ``zoom`` adds the
    attention-guided zoomed view (``cs604.engine.zoom``, class-matched ``cam`` map,
    ``zoom_mass`` default 0.7) and averages its softmax with the others (with
    ``tta_flip``, the zoomed view is flipped too: 4 passes). All are recorded in the
    ``.inference.json``.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for prediction")
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model = build_from_config(cfg)
    load_model_state(model, checkpoint, check_fingerprint=check_fingerprint)
    model.cuda().eval()

    from timm.data import create_transform

    from cs604.models.build import model_data_config

    data_cfg = model_data_config(model)
    if crop_pct is not None:
        if not 0.0 < crop_pct <= 1.0:
            raise ValueError(f"crop_pct must be in (0, 1], got {crop_pct}")
        data_cfg = {**data_cfg, "crop_pct": float(crop_pct)}
    from cs604.engine import zoom as zoom_lib

    zoom_mass = zoom_lib.DEFAULT_MASS if zoom_mass is None else float(zoom_mass)
    if zoom:
        transform = zoom_lib.two_view_transform(data_cfg)
    else:
        transform = create_transform(**data_cfg, is_training=False)
    dataset = CS604ImageDataset(data_root, split, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    precision = str(cfg.get("precision", "bf16"))
    amp_dtype = torch.bfloat16 if precision == "bf16" else (
        torch.float16 if precision == "fp16" else None
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for batch, _labels, paths in loader:
            images = (batch[0] if zoom else batch).cuda(non_blocking=True)
            views = [images]
            if tta_flip:
                views.append(torch.flip(images, dims=[3]))
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                tokens = model.forward_features(images)
                logits = model.forward_head(tokens).float()
                if zoom:
                    zoomed, _boxes = zoom_lib.zoom_view(
                        model, tokens, logits, batch[1].cuda(non_blocking=True), mass=zoom_mass)
                    views.append(zoomed)
                    if tta_flip:
                        views.append(torch.flip(zoomed, dims=[3]))
                if len(views) > 1:
                    probs = logits.softmax(1)
                    for view in views[1:]:
                        probs += model(view).float().softmax(1)
                    logits = probs / len(views)
            predictions = logits.argmax(dim=1).cpu().tolist()
            for rel_path, label in zip(paths, predictions):
                handle.write(f"{rel_path} {label}\n")
            count += len(predictions)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    index_path = Path(data_root) / f"{split}.txt"
    validate_submission(output_path, index_path, int(cfg["num_classes"]))
    from cs604.engine.loop import _git_metadata

    metrics = {
        "checkpoint": str(checkpoint_path),
        # Hashes tie an uploaded file to the exact checkpoint.
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_best_accuracy": checkpoint.get("best_accuracy"),
        "config_file": cfg.get("config_file", ""),
        "predict_git_commit": _git_metadata().get("git_commit", "unknown"),
        "split": split,
        "crop_pct": float(data_cfg["crop_pct"]),
        "tta_flip": bool(tta_flip),
        "zoom": {"map": zoom_lib.DEFAULT_MAP, "mass": zoom_mass, "min_side": zoom_lib.DEFAULT_MIN_SIDE,
                 "factor": zoom_lib.DEFAULT_FACTOR} if zoom else None,
        "base_fingerprint_checked": bool(check_fingerprint),
        "forward_passes_per_image": (2 if tta_flip else 1) * (2 if zoom else 1),
        "samples": count,
        "elapsed_sec": elapsed,
        "images_per_sec": count / max(elapsed, 1e-9),
        "milliseconds_per_image": 1000 * elapsed / max(count, 1),
        "device": torch.cuda.get_device_name(0),
        "submission_sha256": sha256_file(output_path),
    }
    if make_zip:
        zip_path = output_path.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(output_path, arcname=output_path.name)
        metrics["zip_sha256"] = sha256_file(zip_path)
    output_path.with_suffix(".inference.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return output_path
