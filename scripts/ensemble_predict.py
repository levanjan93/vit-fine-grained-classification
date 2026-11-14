"""Average logits from finalist checkpoints and write a CodaBench submission."""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cs604.data.dataset import CS604ImageDataset  # noqa: E402
from cs604.engine.predict import validate_submission  # noqa: E402
from cs604.models.build import build_from_config, load_model_state  # noqa: E402


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    models = []
    configs = []
    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_from_config(checkpoint["config"])
        load_model_state(model, checkpoint)
        models.append(model.cuda().eval())
        configs.append(checkpoint["config"])
    if len({int(cfg["num_classes"]) for cfg in configs}) != 1:
        raise ValueError("All checkpoints must have the same class count")

    from timm.data import create_transform

    from cs604.models.build import model_data_config

    transform = create_transform(
        **model_data_config(models[0]), is_training=False
    )
    dataset = CS604ImageDataset(args.data_root, args.split, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    count = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for images, _labels, paths in loader:
            images = images.cuda(non_blocking=True)
            logits = None
            for model, cfg in zip(models, configs):
                dtype = torch.bfloat16 if cfg.get("precision", "bf16") == "bf16" else torch.float16
                with torch.autocast("cuda", dtype=dtype):
                    current = model(images).float()
                logits = current if logits is None else logits + current
            predictions = logits.argmax(dim=1).cpu().tolist()
            for rel_path, label in zip(paths, predictions):
                handle.write(f"{rel_path} {label}\n")
            count += len(predictions)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    validate_submission(
        args.output,
        args.data_root / f"{args.split}.txt",
        int(configs[0]["num_classes"]),
    )
    args.output.with_suffix(".inference.json").write_text(
        json.dumps(
            {
                "checkpoints": [str(path) for path in args.checkpoints],
                "samples": count,
                "elapsed_sec": elapsed,
                "milliseconds_per_image": 1000 * elapsed / max(1, count),
                "device": torch.cuda.get_device_name(0),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with zipfile.ZipFile(
        args.output.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED
    ) as archive:
        archive.write(args.output, arcname=args.output.name)
    print(f"validated ensemble submission: {args.output}")


if __name__ == "__main__":
    main()
