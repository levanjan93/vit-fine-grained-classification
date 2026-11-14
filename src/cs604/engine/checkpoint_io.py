"""Verified checkpoint writes for slow or interruptible storage (e.g. Google Drive)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import torch

from cs604.models.build import checkpoint_model_state, use_compact_checkpoints

REQUIRED_KEYS = (
    "model",
    "optimizer",
    "scheduler",
    "config",
    "step",
    "epoch",
    "next_batch",
    "best_accuracy",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _staging_dir() -> Path:
    override = os.environ.get("CS604_CHECKPOINT_STAGING")
    if override:
        path = Path(override)
    elif Path("/content").is_dir():
        path = Path("/content") / "cs604-checkpoint-staging"
    else:
        path = Path(tempfile.gettempdir()) / "cs604-checkpoint-staging"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_payload(
    model,
    optimizer,
    scheduler,
    cfg: dict,
    step: int,
    epoch: int,
    next_batch: int,
    best_accuracy: float,
    elapsed_sec: float,
) -> dict:
    return {
        # LoRA/head runs omit the frozen pretrained backbone (~0.37 GB -> ~32 MB).
        **checkpoint_model_state(model, use_compact_checkpoints(cfg)),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": cfg,
        "step": int(step),
        "epoch": int(epoch),
        "next_batch": int(next_batch),
        "best_accuracy": float(best_accuracy),
        # Run time so far, so a resumed run keeps counting GPU-hours from here.
        "elapsed_sec": float(elapsed_sec),
    }


def _verify_loaded(loaded: dict, expected: dict) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in loaded]
    if missing:
        raise ValueError(f"Checkpoint missing keys: {', '.join(missing)}")
    if int(loaded["step"]) != int(expected["step"]):
        raise ValueError("Checkpoint step mismatch after reload")
    if int(loaded["epoch"]) != int(expected["epoch"]):
        raise ValueError("Checkpoint epoch mismatch after reload")
    if int(loaded["next_batch"]) != int(expected["next_batch"]):
        raise ValueError("Checkpoint next_batch mismatch after reload")
    if float(loaded["best_accuracy"]) != float(expected["best_accuracy"]):
        raise ValueError("Checkpoint best_accuracy mismatch after reload")
    if loaded["config"] != expected["config"]:
        raise ValueError("Checkpoint config mismatch after reload")
    if set(loaded["model"].keys()) != set(expected["model"].keys()):
        raise ValueError("Checkpoint model keys mismatch after reload")
    if set(loaded["optimizer"].keys()) != set(expected["optimizer"].keys()):
        raise ValueError("Checkpoint optimizer keys mismatch after reload")
    if set(loaded["scheduler"].keys()) != set(expected["scheduler"].keys()):
        raise ValueError("Checkpoint scheduler keys mismatch after reload")


def _verify_copy(path: Path, expected_digest: str, payload: dict) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Checkpoint copy missing or empty: {path}")
    digest = sha256_file(path)
    if digest != expected_digest:
        raise ValueError(f"Checkpoint checksum mismatch for {path}")
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    _verify_loaded(loaded, payload)


def _versioned_name(dest: Path, step: int) -> str:
    if dest.name == "last.pt":
        return f"resume_step_{step:06d}.pt"
    if dest.name == "baseline.pt":
        return f"baseline_step_{step:06d}.pt"
    if dest.name == "best.pt":
        return f"best_step_{step:06d}.pt"
    return f"{dest.stem}_step_{step:06d}.pt"


def _append_manifest(output_dir: Path, entry: dict) -> None:
    manifest = output_dir / "checkpoint_hashes.jsonl"
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def save_training_checkpoint(
    dest: Path,
    model,
    optimizer,
    scheduler,
    cfg: dict,
    step: int,
    epoch: int,
    next_batch: int,
    best_accuracy: float,
    elapsed_sec: float = 0.0,
) -> dict:
    """Stage on local disk, verify, then copy to a versioned file and dest."""

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_payload(
        model, optimizer, scheduler, cfg, step, epoch, next_batch, best_accuracy, elapsed_sec
    )
    staging_root = _staging_dir()
    token = f"{dest.name}_{step:06d}_{os.getpid()}_{time.time_ns()}"
    staging_path = staging_root / f"write_{token}.pt"
    dest_staging = staging_root / f"dest_{token}.pt"
    try:
        torch.save(payload, staging_path)
        loaded = torch.load(staging_path, map_location="cpu", weights_only=False)
        _verify_loaded(loaded, payload)
        staging_digest = sha256_file(staging_path)

        checkpoints_dir = dest.parent / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        versioned = checkpoints_dir / _versioned_name(dest, step)
        shutil.copy2(staging_path, versioned)
        _verify_copy(versioned, staging_digest, payload)

        # Copy through staging again so dest is never replaced from a partial Drive write.
        shutil.copy2(versioned, dest_staging)
        _verify_copy(dest_staging, staging_digest, payload)
        shutil.copy2(dest_staging, dest)
        _verify_copy(dest, staging_digest, payload)

        entry = {
            "timestamp": time.time(),
            "dest": str(dest),
            "versioned": str(versioned),
            "step": int(step),
            "epoch": int(epoch),
            "next_batch": int(next_batch),
            "sha256": staging_digest,
        }
        _append_manifest(dest.parent, entry)
        print(f"saved checkpoint step={step} -> {dest.name} ({versioned.name})", flush=True)
        return entry
    finally:
        staging_path.unlink(missing_ok=True)
        dest_staging.unlink(missing_ok=True)
