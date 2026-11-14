"""Evaluation-only "look, then zoom in" test-time views on a checkpoint's internal holdout.

For each image the model first classifies the usual eval view. A saliency map on the
patch grid then picks the region the model relied on; that region is cut from a
2x-resolution copy of the same view, resized to the model's input size, and
classified again. The zoomed prediction is averaged (softmax) with the plain one.

Saliency maps:

- ``attn``: last block's CLS-to-patch attention, mean over heads.
- ``cam``: cosine between each final patch token and the predicted class's head
  weight (class-specific), minus its per-image mean, clamped at zero.
- ``center``: control with no saliency: the ``cam`` box's size, moved to the centre.

Box: the fewest patches holding ``mass`` of the map, their bounding box grown to a
square of at least ``min_side`` patches and kept inside the image.

Scored per view combination (mean softmax): plain, flip, plain+flip (the current
submission recipe), zoom, plain+zoom, and plain+flip+zoom+zoom-flip (4 passes). A
4-pass reference without zoom (two eval crops, each with its flip) is scored too.
Accuracies are reported on all holdout images and on its even / odd halves, so a
setting chosen on one half can be checked on the other.

The plain view at the run's training eval crop must reproduce the run's final
holdout accuracy (``--gate-tolerance``). Nothing is trained.

Writes ``<run>/analysis/eval_zoom.json`` plus per-image results in ``eval_zoom_images.pt``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cs604.data.dataset import CS604ImageDataset, load_split_index  # noqa: E402
from cs604.engine.checkpoint_io import sha256_file  # noqa: E402
from cs604.engine.zoom import (  # noqa: E402
    AttentionTap, TwoViews, cam_map, center_boxes, crop_zoom, select_boxes,
)
from cs604.models.build import build_from_config, load_model_state, model_data_config  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from eval_tta import _final_holdout_accuracy  # noqa: E402


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--crop-pct", type=float, default=0.75, help="eval crop for the zoom study")
    parser.add_argument("--reference-crop-pct", type=float, default=None,
                        help="second crop for the 4-pass reference and the gate (default: the run's)")
    parser.add_argument("--maps", nargs="+", default=["attn", "cam"], choices=["attn", "cam", "center"])
    parser.add_argument("--mass", type=float, nargs="+", default=[0.5, 0.8])
    parser.add_argument("--min-side", type=int, default=8, help="smallest box, in patches")
    parser.add_argument("--zoom-factor", type=int, default=2, help="resolution of the copy zoomed from")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="first N holdout images (smoke test)")
    parser.add_argument("--gate-tolerance", type=float, default=0.0005)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    from timm.data import create_transform

    run_dir = args.checkpoint.parent
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model = build_from_config(cfg)
    load_model_state(model, checkpoint)
    model.cuda().eval()
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(str(cfg.get("precision", "bf16")))
    tap = AttentionTap(model)

    samples = load_split_index(run_dir / "splits" / "internal_val.txt")
    if args.limit is not None:
        samples = samples[: args.limit]
    count = len(samples)
    labels_all = torch.tensor([label for _path, label in samples], device="cuda")
    data_cfg = model_data_config(model)
    size = int(data_cfg["input_size"][-1])
    patch = int(model.patch_embed.patch_size[0])
    grid = size // patch
    reference_crop = args.reference_crop_pct or float(data_cfg["crop_pct"])
    configs = [(name, mass) for name in args.maps for mass in args.mass]

    def forward(images):
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            tokens = model.forward_features(images)
            return tokens, model.forward_head(tokens).float()

    # Per-image results. Probabilities are not kept (50,000 x 10,000); only the
    # 4-pass multi-crop reference is accumulated, in fp16 on the GPU.
    reference_probs = torch.zeros(count, int(cfg["num_classes"]), device="cuda", dtype=torch.float16)
    correct: dict[str, torch.Tensor] = {}
    box_side: dict[str, torch.Tensor] = {}

    def record(name: str, offset: int, hits: torch.Tensor) -> None:
        correct.setdefault(name, torch.zeros(count, dtype=torch.bool))[offset: offset + len(hits)] = hits.cpu()

    timings = {}
    for crop_pct, zoom in [(reference_crop, False), (args.crop_pct, True)]:
        small = create_transform(**{**data_cfg, "crop_pct": crop_pct}, is_training=False)
        transform = small
        if zoom:
            big_size = (3, size * args.zoom_factor, size * args.zoom_factor)
            big = create_transform(**{**data_cfg, "crop_pct": crop_pct, "input_size": big_size}, is_training=False)
            transform = TwoViews(small, big)
        dataset = CS604ImageDataset(cfg["data_root"], "internal_val", transform=transform, samples=samples)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        tag = f"crop{crop_pct:g}"
        started = time.perf_counter()
        offset = 0
        for batch, labels, _paths in loader:
            labels = labels.cuda(non_blocking=True)
            images = (batch[0] if zoom else batch).cuda(non_blocking=True)
            tokens, logits = forward(images)
            attention = tap.cls_attention()
            _tokens, flipped = forward(torch.flip(images, dims=[3]))
            p_plain, p_flip = logits.softmax(1), flipped.softmax(1)
            record(f"{tag}/plain", offset, logits.argmax(1) == labels)
            record(f"{tag}/flip", offset, flipped.argmax(1) == labels)
            record(f"{tag}/plain+flip", offset, (p_plain + p_flip).argmax(1) == labels)
            reference_probs[offset: offset + len(labels)] += ((p_plain + p_flip) / 2).half()
            if zoom:
                big_images = batch[1].cuda(non_blocking=True)
                predicted = logits.argmax(1)
                maps = {"attn": attention, "cam": cam_map(model, tokens, predicted)}
                for name, mass in configs:
                    key = f"{tag}/{name}_m{mass:g}"
                    boxes = select_boxes(maps["cam" if name == "center" else name], grid, mass, args.min_side)
                    if name == "center":
                        boxes = center_boxes(boxes, grid)
                    zoomed = crop_zoom(big_images, boxes, patch * args.zoom_factor, size)
                    _t, z_logits = forward(zoomed)
                    _t, zf_logits = forward(torch.flip(zoomed, dims=[3]))
                    p_zoom, p_zflip = z_logits.softmax(1), zf_logits.softmax(1)
                    record(f"{key}/zoom", offset, z_logits.argmax(1) == labels)
                    record(f"{key}/plain+zoom", offset, (p_plain + p_zoom).argmax(1) == labels)
                    record(f"{key}/4pass", offset, (p_plain + p_flip + p_zoom + p_zflip).argmax(1) == labels)
                    box_side.setdefault(key, torch.zeros(count, dtype=torch.uint8))[
                        offset: offset + len(labels)] = boxes[:, 2].to(torch.uint8).cpu()
            offset += len(labels)
        timings[tag] = time.perf_counter() - started
        print(f"{tag}: {timings[tag]:.0f} s", flush=True)
    reference_hits = reference_probs.argmax(1) == labels_all
    record(f"crop{reference_crop:g}+crop{args.crop_pct:g}/4pass_flips", 0, reference_hits)

    def summary(hits: torch.Tensor) -> dict:
        hits = hits.float()
        return {"all": float(hits.mean()), "even": float(hits[0::2].mean()), "odd": float(hits[1::2].mean())}

    results = {name: summary(hits) for name, hits in correct.items()}
    # Does zooming help most where the salient region is small? Bucket by box side.
    by_box = {}
    base = correct[f"crop{args.crop_pct:g}/plain"]
    for key, sides in box_side.items():
        zoomed = correct[f"{key}/plain+zoom"]
        buckets = {}
        for low, high in [(0, 10), (10, 14), (14, 18), (18, grid + 1)]:
            mask = (sides >= low) & (sides < high)
            if int(mask.sum()):
                buckets[f"side_{low}-{high - 1}"] = {
                    "images": int(mask.sum()),
                    "plain": float(base[mask].float().mean()),
                    "plain+zoom": float(zoomed[mask].float().mean()),
                }
        by_box[key] = {"mean_side": float(sides.float().mean()), "buckets": buckets}
    for name, value in results.items():
        print(f"{name:42s} {value['all'] * 100:7.3f}  (even {value['even'] * 100:.3f}, odd {value['odd'] * 100:.3f})")

    reference = _final_holdout_accuracy(run_dir)
    gate = None
    if args.limit is None and reference is not None and reference_crop == float(data_cfg["crop_pct"]):
        reproduced = results[f"crop{reference_crop:g}/plain"]["all"]
        gate = {"reference": reference, "reproduced": reproduced, "crop_pct": reference_crop}
        if abs(reproduced - reference) > args.gate_tolerance:
            raise RuntimeError(f"gate failed: {reproduced:.5f} vs run's final holdout {reference:.5f}")

    from cs604.engine.loop import _git_metadata

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": checkpoint.get("step"),
        "samples": count,
        "settings": {"crop_pct": args.crop_pct, "reference_crop_pct": reference_crop, "maps": args.maps,
                     "mass": args.mass, "min_side": args.min_side, "zoom_factor": args.zoom_factor,
                     "grid": grid},
        "gate": gate,
        "git": _git_metadata(),
        "device": torch.cuda.get_device_name(0),
        "elapsed_sec": timings,
        "results": results,
        "by_box_side": by_box,
    }
    suffix = "" if args.limit is None else "_smoke"
    output = args.output or run_dir / "analysis" / f"eval_zoom{suffix}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save({"correct": correct, "box_side": box_side, "labels": labels_all.cpu()},
               output.with_name(output.stem + "_images.pt"))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
