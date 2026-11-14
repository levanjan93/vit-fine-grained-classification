"""Training-time budget in the course's unit: hours on 48 GB of GPU memory.

The course allows 2 hours x 48 GB, or the same GB-hours on another card
(4 hours on 24 GB). A run with ``budget_48gb_hours`` set fits its learning-rate
schedule to that budget: schedule progress is the larger of the step fraction
and the time fraction, so a slower GPU still finishes the full decay on time.
"""

from __future__ import annotations

import math

COURSE_VRAM_GB = 48.0


def budget_wall_seconds(budget_48gb_hours: float, vram_gb: float) -> float:
    """Wall-clock seconds this GPU may train for under the normalized budget."""

    if budget_48gb_hours <= 0 or vram_gb <= 0:
        raise ValueError("budget_48gb_hours and vram_gb must be positive")
    return budget_48gb_hours * COURSE_VRAM_GB / vram_gb * 3600.0


def cosine_lr_factor(step: float, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup, then cosine decay to zero at ``total_steps``."""

    if step < warmup_steps:
        return max(1e-3, (step + 1) / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


LR_SCHEDULES = ("cosine", "constant")


def lr_schedule_from_config(cfg: dict) -> str:
    """``lr_schedule`` (default cosine); constant keeps the warmup, then holds the peak rate."""

    schedule = str(cfg.get("lr_schedule") or "cosine")
    if schedule not in LR_SCHEDULES:
        raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}, got {schedule!r}")
    return schedule


def lr_schedule_factor(step: float, warmup_steps: int, total_steps: int, schedule: str) -> float:
    """The learning-rate multiplier at ``step`` for the named schedule."""

    if schedule == "constant":
        if step < warmup_steps:
            return max(1e-3, (step + 1) / max(1, warmup_steps))
        return 1.0
    return cosine_lr_factor(step, warmup_steps, total_steps)


def effective_step(step: int, total_steps: int, elapsed_sec: float, limit_sec: float | None) -> float:
    """The schedule position: real steps, or further along if time runs ahead."""

    if limit_sec is None or limit_sec <= 0:
        return float(step)
    return max(float(step), min(1.0, elapsed_sec / limit_sec) * total_steps)


def training_limit_seconds(budget_sec: float, final_eval_sec: float) -> float:
    """Stop training early enough that the final holdout evaluation fits too."""

    return max(0.0, budget_sec - max(0.0, final_eval_sec))
