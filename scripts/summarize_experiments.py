"""Build a compact CSV index from local or downloaded Colab run folders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL = ROOT.parent / "cs604-outputs"

FIELDS = [
    "run",
    "config_file",
    "status",
    "mode",
    "git_commit",
    "git_dirty",
    "device",
    "best_internal_accuracy",
    "public_accuracy",
    "submission_id",
    "trainable_params",
    "trainable_pct",
    "elapsed_sec",
    "gpu_hours",
    "normalized_48gb_hours",
    "max_mem_gb",
    "notes",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=Path, default=[DEFAULT_LOCAL])
    parser.add_argument("--output", type=Path, default=ROOT / "report" / "experiment_index.csv")
    args = parser.parse_args()
    rows: list[dict] = []
    for root in args.roots:
        for run_path in sorted(root.rglob("run.json")):
            run = json.loads(run_path.read_text(encoding="utf-8"))
            manual_path = run_path.with_name("leaderboard.json")
            manual = (
                json.loads(manual_path.read_text(encoding="utf-8"))
                if manual_path.is_file()
                else {}
            )
            rows.append(
                {
                    **{field: run.get(field, "") for field in FIELDS},
                    "run": str(run_path.parent),
                    "public_accuracy": manual.get("public_accuracy", ""),
                    "submission_id": manual.get("submission_id", ""),
                    "notes": manual.get("notes", ""),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"indexed {len(rows)} runs in {args.output}")


if __name__ == "__main__":
    main()
