"""Run a head-only sweep on cached features (one YAML, many sweep points).

The sweep YAML has ``cache_dir``, ``output_dir``, a ``base`` training config
(the keys of a ``mode: head`` experiment YAML) and ``points``: each point has a
unique ``name``, a ``round`` and a ``note``, and overrides base keys. Each
finished point appends one row to ``<output_dir>/sweep.csv`` and writes its
per-epoch curve to ``<output_dir>/points/<name>.json``. Finished points are
skipped, so the sweep can be stopped and rerun; a finished name whose settings
changed is an error (give the new setting a new name).

    python scripts/sweep_cached_head.py --config configs/experiments/head_sweep_orig_in21k.yaml [--only NAME ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cs604.engine.cached_head import CachedFeatures, train_cached_head  # noqa: E402
from cs604.engine.feature_cache import write_json_atomic  # noqa: E402
from cs604.engine.loop import _git_metadata  # noqa: E402

POINT_KEYS = ("name", "round", "note")
CSV_FIELDS = [
    "name", "round", "train_groups", "head_type", "lr", "head_lr", "weight_decay", "epochs",
    "max_steps", "batch_size", "label_smoothing", "cosine_scale", "cosine_scale_trainable",
    "seed", "initial_accuracy", "final_accuracy", "best_accuracy", "best_epoch",
    "final_loss", "final_scale", "optimizer_steps", "elapsed_sec", "git_commit", "git_dirty", "note",
]


def resolve_points(sweep: dict) -> list[dict]:
    names = set()
    points = []
    for point in sweep["points"]:
        name = point["name"]
        if name in names:
            raise SystemExit(f"Duplicate sweep point name {name}")
        names.add(name)
        settings = {**sweep["base"], **{k: v for k, v in point.items() if k not in POINT_KEYS}}
        points.append({"name": name, "round": point.get("round"), "note": point.get("note", ""), "cfg": settings})
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--only", nargs="+", help="run only these point names")
    parser.add_argument("--save-heads", action="store_true", help="also save each trained head's state dict")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    sweep = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = Path(sweep["output_dir"])
    (output_dir / "points").mkdir(parents=True, exist_ok=True)
    points = resolve_points(sweep)
    if args.only:
        missing = set(args.only) - {p["name"] for p in points}
        if missing:
            raise SystemExit(f"Unknown points: {sorted(missing)}")
        points = [p for p in points if p["name"] in args.only]

    todo = []
    for point in points:
        path = output_dir / "points" / f"{point['name']}.json"
        if path.is_file():
            done = json.loads(path.read_text(encoding="utf-8"))
            if done["cfg"] != point["cfg"]:
                raise SystemExit(f"{point['name']} finished with other settings; rename the changed point")
            continue
        todo.append(point)
    print(f"{len(points) - len(todo)} points done, {len(todo)} to run", flush=True)

    csv_path = output_dir / "sweep.csv"
    new_csv = not csv_path.is_file()
    git = _git_metadata()
    device = torch.device("cuda")
    data = None
    # Run points grouped by their training views, so each cache group loads once.
    todo.sort(key=lambda p: json.dumps(p["cfg"].get("train_groups")))
    for point in todo:
        cfg = point["cfg"]
        groups = list(cfg["train_groups"])
        if data is None or data.train_groups != groups:
            data = None
            torch.cuda.empty_cache()
            loading = time.perf_counter()
            data = CachedFeatures(sweep["cache_dir"], groups, device)
            print(f"loaded {len(data.view_names)} training views ({groups}) in "
                  f"{time.perf_counter() - loading:.0f}s", flush=True)
        print(f"===== {point['name']} (round {point['round']}) =====", flush=True)
        result = train_cached_head(cfg, data)
        head = result.pop("head")
        if args.save_heads:
            # Loadable by training as head_init_file (cs604.models.build.load_head_init).
            torch.save(
                {"head": head.state_dict(), "timm_checkpoint": cfg["timm_checkpoint"],
                 "point": point["name"], "final_accuracy": result["final_accuracy"], "cfg": cfg},
                output_dir / "points" / f"{point['name']}.head.pt",
            )
        record = {**point, **result, **git, "device": torch.cuda.get_device_name(0),
                  "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        write_json_atomic(output_dir / "points" / f"{point['name']}.json", record)
        row = {key: cfg.get(key) for key in CSV_FIELDS if key in cfg}
        row.update({
            "name": point["name"], "round": point["round"], "note": point["note"],
            "train_groups": "+".join(groups),
            "cosine_scale": cfg.get("cosine_scale", 20.0),
            "cosine_scale_trainable": cfg.get("cosine_scale_trainable", True),
            **{key: result[key] for key in (
                "initial_accuracy", "final_accuracy", "best_accuracy", "best_epoch",
                "final_loss", "final_scale", "optimizer_steps", "elapsed_sec")},
            "git_commit": git.get("git_commit"), "git_dirty": git.get("git_dirty"),
        })
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            if new_csv:
                writer.writeheader()
                new_csv = False
            writer.writerow(row)
        print(f"{point['name']}: final {100 * result['final_accuracy']:.3f}% "
              f"best {100 * result['best_accuracy']:.3f}% (epoch {result['best_epoch']}) "
              f"in {result['elapsed_sec']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
