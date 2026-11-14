"""Rank completed experiments by internal accuracy without leaderboard peeking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/finalist_selection.json")
    )
    args = parser.parse_args()
    candidates = []
    for path in args.root.rglob("run.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        accuracy = record.get("best_internal_accuracy")
        if accuracy is None:
            continue
        candidates.append(
            {
                "run": str(path.parent),
                "checkpoint": str(path.parent / "best.pt"),
                "internal_accuracy": accuracy,
                "mode": record.get("mode"),
                "trainable_pct": record.get("trainable_pct"),
                "gpu_hours": record.get("gpu_hours"),
                "git_commit": record.get("git_commit"),
                "git_dirty": record.get("git_dirty"),
            }
        )
    candidates.sort(key=lambda item: item["internal_accuracy"], reverse=True)
    selected = candidates[: args.top]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    for rank, item in enumerate(selected, start=1):
        print(
            f"{rank}. {100 * item['internal_accuracy']:.2f}% "
            f"{item['mode']} {item['run']}"
        )


if __name__ == "__main__":
    main()
