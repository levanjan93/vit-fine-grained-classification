"""Checks for preserving and measuring a model before optimizer updates."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def baseline_requested(cfg: dict) -> bool:
    return bool(cfg.get("evaluate_only") or cfg.get("evaluate_before_training"))


def validate_baseline_setup(cfg: dict, output_dir: Path) -> None:
    """Reject settings that would skip validation or overwrite a saved baseline."""
    if not baseline_requested(cfg):
        return
    if cfg.get("synthetic") or not cfg.get("validate", True):
        raise ValueError("Baseline validation requires a real, labeled internal holdout")
    if cfg.get("evaluate_only") and cfg.get("resume"):
        raise ValueError("Evaluation-only runs cannot resume training checkpoints")
    expected = cfg.get("expected_initial_accuracy")
    tolerance = float(cfg.get("initial_accuracy_tolerance", 0.001))
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("initial_accuracy_tolerance must be finite and non-negative")
    if expected is not None and not 0 <= float(expected) <= 1:
        raise ValueError("expected_initial_accuracy must be a fraction in [0, 1]")
    initializers = [key for key in ("prototype_file", "init_checkpoint", "head_init_file") if cfg.get(key)]
    if len(initializers) > 1:
        raise ValueError(f"Choose one of prototype_file, init_checkpoint, head_init_file; got {initializers}")
    if cfg.get("resume"):
        required = ("baseline.pt", "baseline.json", "best.pt")
        if not all((output_dir / name).is_file() for name in required):
            raise ValueError("Resume a baseline-protected run in its original output_dir")
        baseline = json.loads((output_dir / "baseline.json").read_text(encoding="utf-8"))
        if not baseline.get("passed"):
            raise ValueError("Cannot resume a run whose initial validation failed")
        return
    artifacts = ("baseline.pt", "baseline.json", "best.pt", "last.pt", "run.json", "metrics.csv", "config.json")
    if any((output_dir / name).exists() for name in artifacts):
        raise FileExistsError(
            f"Refusing to overwrite baseline/run artifacts in {output_dir}; "
            "use a new output directory or resume the original training run"
        )


# Prototype files written before build_prototypes.py recorded its backbone.
LEGACY_PROTOTYPE_BACKBONE = "vit_base_patch16_224.augreg2_in21k_ft_in1k"


def check_prototype_split(cfg: dict, prototype_blob) -> None:
    """Prototypes must come from this backbone and the split used for evaluation."""
    if isinstance(prototype_blob, dict):
        built_with = prototype_blob.get("timm_checkpoint", LEGACY_PROTOTYPE_BACKBONE)
        if built_with != cfg.get("timm_checkpoint"):
            raise ValueError(
                f"Prototypes were built with {built_with}, not {cfg.get('timm_checkpoint')}; "
                "build prototypes for this backbone"
            )
    if not baseline_requested(cfg):
        return
    if not isinstance(prototype_blob, dict):
        raise ValueError("Baseline validation requires prototype split metadata")
    for key, default in (("split_seed", 0), ("val_per_class", 5)):
        if prototype_blob.get(key) != int(cfg.get(key, default)):
            raise ValueError(f"Prototype {key} does not match the experiment config")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def baseline_provenance(cfg: dict, split_path: Path, samples: int) -> dict:
    """Record the exact source weights and require the original holdout if given."""
    split_hash = _sha256(split_path)
    reference = cfg.get("baseline_reference_split")
    if reference and _sha256(Path(reference)) != split_hash:
        raise ValueError("Internal holdout does not match baseline_reference_split")
    expected_samples = cfg.get("expected_initial_samples")
    if expected_samples is not None and samples != int(expected_samples):
        raise ValueError(f"Expected {expected_samples} held-out images, found {samples}")
    result = {"val_images": samples, "internal_val_sha256": split_hash}
    for key in ("prototype_file", "init_checkpoint", "head_init_file"):
        if cfg.get(key):
            path = Path(cfg[key])
            result[key] = str(path.resolve())
            result[f"{key}_sha256"] = _sha256(path)
    return result


def assess_initial_accuracy(accuracy: float, cfg: dict) -> dict:
    expected = cfg.get("expected_initial_accuracy")
    tolerance = float(cfg.get("initial_accuracy_tolerance", 0.001))
    delta = None if expected is None else accuracy - float(expected)
    passed = math.isfinite(accuracy) and 0 <= accuracy <= 1
    if delta is not None:
        passed = passed and abs(delta) <= tolerance + 1e-12
    return {
        "initial_internal_accuracy": accuracy,
        "expected_initial_accuracy": expected,
        "initial_accuracy_tolerance": tolerance,
        "accuracy_delta": delta,
        "passed": passed,
    }
