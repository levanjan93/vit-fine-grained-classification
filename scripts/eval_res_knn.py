"""Evaluation-only: predict above the training resolution, and blend in a kNN over training images.

Two prediction-time ideas, measured on a checkpoint's internal holdout. Nothing is trained.

1. **Higher resolution.** The model is rebuilt at each ``--resolutions`` size with its
   position embeddings resampled (timm ``set_input_size``, bicubic). The eval view keeps
   the same field of view (``--crop-pct``) at more pixels. Each resolution gets the plain
   view and the ``cam`` m0.7 zoom view (``cs604.engine.zoom``, cut from a 2x copy, resized
   to that resolution).
2. **kNN.** One forward pass over the run's own training images (``internal_train``, the
   eval view at the training size) stores their pooled, normalized features. Each holdout
   image's nearest training images vote for their classes, weighted ``exp(cosine / T)``
   over the top ``k``, and the vote distribution is blended with the classifier's
   softmax: ``(1 - lam) * p_classifier + lam * p_knn``.

Per view, the GPU pass stores the top ``--keep`` softmax probabilities and the top
``--keep`` neighbours, so view combinations and the kNN grid are scored afterwards
(``--analyze-only`` reruns just that); a rerun reuses saved stores and the bank. A class outside a view's stored top ``--keep``
counts as probability 0 in that view (smaller than the ``--keep``-th probability).

Settings are chosen on the even half of the holdout and reported on the odd half.
Gate: the training-resolution plain and plain+zoom views must reproduce the run's
``analysis/eval_zoom.json`` (same crop) within ``--gate-tolerance``.

Writes ``<run>/analysis/eval_res_knn.json``; the stores go to ``analysis/res_knn/``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cs604.data.dataset import CS604ImageDataset, load_split_index  # noqa: E402
from cs604.engine.checkpoint_io import sha256_file  # noqa: E402
from cs604.engine.zoom import (  # noqa: E402
    DEFAULT_FACTOR, DEFAULT_MASS, DEFAULT_MIN_SIDE, TwoViews, cam_map, crop_zoom, select_boxes,
)
from cs604.models.build import build_from_config, load_model_state, model_data_config  # noqa: E402


def build_model(checkpoint: dict, size: int | None):
    model = build_from_config(checkpoint["config"])
    load_model_state(model, checkpoint)
    if size is not None and size != model.patch_embed.img_size[0]:
        model.set_input_size(img_size=(size, size))
    return model.cuda().eval()


def make_forward(model, amp_dtype):
    def forward(images):
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            tokens = model.forward_features(images)
            pooled = model.forward_head(tokens, pre_logits=True)
            logits = model.get_classifier()(pooled)
        return tokens, F.normalize(pooled.float(), dim=-1), logits.float()
    return forward


@torch.inference_mode()
def bank_pass(model, forward, cfg, data_cfg, crop_pct, run_dir, args) -> tuple[torch.Tensor, torch.Tensor]:
    from timm.data import create_transform

    samples = load_split_index(run_dir / "splits" / "internal_train.txt")
    if args.limit is not None:
        samples = samples[: args.limit * 8]
    transform = create_transform(**{**data_cfg, "crop_pct": crop_pct}, is_training=False)
    dataset = CS604ImageDataset(cfg["data_root"], "internal_train", transform=transform, samples=samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    features = torch.empty(len(samples), model.num_features, dtype=torch.float16, device="cuda")
    offset = 0
    for images, _labels, _paths in loader:
        _tokens, pooled, _logits = forward(images.cuda(non_blocking=True))
        features[offset: offset + len(pooled)] = pooled.half()
        offset += len(pooled)
    labels = torch.tensor([label for _path, label in samples], dtype=torch.int16)
    return features, labels


@torch.inference_mode()
def holdout_pass(model, forward, cfg, data_cfg, crop_pct, samples, bank, args) -> dict:
    """Plain + cam zoom view at the model's size; top-``keep`` probabilities and neighbours."""

    from timm.data import create_transform

    size = int(data_cfg["input_size"][-1])
    patch = int(model.patch_embed.patch_size[0])
    grid = size // patch
    view = {**data_cfg, "crop_pct": crop_pct}
    big = {**view, "input_size": (3, size * DEFAULT_FACTOR, size * DEFAULT_FACTOR)}
    transform = TwoViews(create_transform(**view, is_training=False), create_transform(**big, is_training=False))
    dataset = CS604ImageDataset(cfg["data_root"], "internal_val", transform=transform, samples=samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    bank_features = bank[0]
    out = {}

    def keep(name, offset, values, indices):
        for part, tensor in (("p", values), ("i", indices)):
            key = f"{name}/{part}"
            if key not in out:
                out[key] = torch.empty(len(samples), args.keep, dtype=tensor.dtype)
            out[key][offset: offset + len(tensor)] = tensor.cpu()

    offset = 0
    for (small, large), _labels, _paths in loader:
        small = small.cuda(non_blocking=True)
        tokens, f_plain, logits = forward(small)
        boxes = select_boxes(cam_map(model, tokens, logits.argmax(1)), grid, DEFAULT_MASS, DEFAULT_MIN_SIDE)
        zoomed = crop_zoom(large.cuda(non_blocking=True), boxes, patch * DEFAULT_FACTOR, size)
        _tokens, f_zoom, z_logits = forward(zoomed)
        for name, lg in (("plain", logits), ("zoom", z_logits)):
            keep(name, offset, *lg.softmax(1).topk(args.keep, dim=1))
        queries = {"plain": f_plain, "plainzoom": F.normalize(f_plain + f_zoom, dim=-1)}
        for name, query in queries.items():
            sims, where = (query.half() @ bank_features.T).float().topk(args.keep, dim=1)
            keep(f"knn_{name}", offset, sims, where.to(torch.int32))
        offset += len(small)
    return out


# ---------------------------------------------------------------- scoring (CPU/GPU, no model)

def dense(store: dict, view: str, rows: slice, classes: int) -> torch.Tensor:
    probs = store[f"{view}/p"][rows].cuda().float()
    index = store[f"{view}/i"][rows].cuda().long()
    return torch.zeros(len(probs), classes, device="cuda").scatter_add_(1, index, probs)


def knn_dense(store: dict, query: str, bank_labels: torch.Tensor, rows: slice, classes: int,
              k: int, temperature: float) -> torch.Tensor:
    sims = store[f"knn_{query}/p"][rows, :k].cuda()
    labels = bank_labels.cuda()[store[f"knn_{query}/i"][rows, :k].cuda().long()].long()
    votes = torch.zeros(len(sims), classes, device="cuda").scatter_add_(1, labels, (sims / temperature).exp())
    return votes / votes.sum(1, keepdim=True)


def hits_for(stores: dict, combo: list[tuple[str, str]], labels: torch.Tensor, classes: int,
             knn: tuple | None = None, bank_labels: torch.Tensor | None = None,
             chunk: int = 5000) -> torch.Tensor:
    """``combo`` is a list of (resolution key, view); views are averaged. ``knn`` = (res, query, k, T, lam)."""

    hits = torch.empty(len(labels), dtype=torch.bool)
    for start in range(0, len(labels), chunk):
        rows = slice(start, start + chunk)
        probs = sum(dense(stores[res], view, rows, classes) for res, view in combo) / max(len(combo), 1)
        if knn is not None:
            res, query, k, temperature, lam = knn
            vote = knn_dense(stores[res], query, bank_labels, rows, classes, k, temperature)
            probs = vote if not combo else (1 - lam) * probs + lam * vote
        hits[rows] = (probs.argmax(1).cpu() == labels[rows])
    return hits


def summary(hits: torch.Tensor) -> dict:
    hits = hits.float()
    return {"all": float(hits.mean()), "even": float(hits[0::2].mean()), "odd": float(hits[1::2].mean())}


def paired_z(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = a.float() - b.float()
    return float(diff.mean() / (diff.std() / len(diff) ** 0.5).clamp_min(1e-12))


def analyze(stores: dict, bank_labels: torch.Tensor, labels: torch.Tensor, classes: int,
            train_res: int, args) -> dict:
    base_res = str(train_res)
    resolutions = list(stores)
    results: dict[str, dict] = {}
    hits_cache: dict[str, torch.Tensor] = {}

    def score(name, combo, knn=None):
        hits = hits_for(stores, combo, labels, classes, knn, bank_labels)
        hits_cache[name] = hits
        results[name] = summary(hits)
        return results[name]

    # 1. Resolution: per resolution, and stacked onto the training-size plain + zoom.
    two_pass = {}
    for res in resolutions:
        score(f"r{res}/plain", [(res, "plain")])
        score(f"r{res}/zoom", [(res, "zoom")])
        two_pass[f"r{res}/plain+zoom"] = [(res, "plain"), (res, "zoom")]
        if res != base_res:
            two_pass[f"r{res}/plain + r{base_res}/zoom"] = [(res, "plain"), (base_res, "zoom")]
            score(f"r{base_res}/plain+zoom + r{res}/plain",
                  [(base_res, "plain"), (base_res, "zoom"), (res, "plain")])
            score(f"r{base_res}/plain+zoom + r{res}/plain+zoom",
                  [(base_res, "plain"), (base_res, "zoom"), (res, "plain"), (res, "zoom")])
    for name, combo in two_pass.items():
        score(name, combo)
    reference = f"r{base_res}/plain+zoom"
    best_two = max(two_pass, key=lambda n: results[n]["even"])

    # 2. kNN alone, then blended with the reference and with the best 2-pass resolution view.
    knn_grid = list(itertools.product(args.k, args.temperature))
    for query, (k, temperature) in itertools.product(["plain", "plainzoom"], knn_grid):
        score(f"knn_{query}/k{k}_T{temperature:g}", [], (base_res, query, k, temperature, 1.0))
    bases = {"ref": [(base_res, "plain"), (base_res, "zoom")]}
    if best_two != reference:
        bases["best_res"] = two_pass[best_two]
    blends = {}
    for base_name, combo in bases.items():
        rows = []
        for query, (k, temperature), lam in itertools.product(["plain", "plainzoom"], knn_grid, args.lam):
            name = f"{base_name} + knn_{query}/k{k}_T{temperature:g}_lam{lam:g}"
            rows.append((score(name, combo, (base_res, query, k, temperature, lam))["even"], name))
        chosen = max(rows)[1]
        blends[base_name] = {"chosen_on_even": chosen, "result": results[chosen],
                             "paired_z_vs_ref": paired_z(hits_cache[chosen], hits_cache[reference])}
    return {
        "reference": reference,
        "best_two_pass_on_even": best_two,
        "best_two_pass_paired_z_vs_ref": paired_z(hits_cache[best_two], hits_cache[reference]),
        "knn_blends": blends,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[336, 384, 448])
    parser.add_argument("--crop-pct", type=float, default=0.75)
    parser.add_argument("--keep", type=int, default=100, help="top probabilities / neighbours stored per view")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    parser.add_argument("--temperature", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    parser.add_argument("--lam", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="first N holdout images (smoke test)")
    parser.add_argument("--gate-tolerance", type=float, default=0.0005)
    parser.add_argument("--analyze-only", action="store_true", help="rescore the saved stores")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    run_dir = args.checkpoint.parent
    suffix = "" if args.limit is None else "_smoke"
    store_dir = run_dir / "analysis" / f"res_knn{suffix}"
    store_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    classes = int(cfg["num_classes"])
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(str(cfg.get("precision", "bf16")))
    samples = load_split_index(run_dir / "splits" / "internal_val.txt")
    if args.limit is not None:
        samples = samples[: args.limit]
    labels = torch.tensor([label for _path, label in samples])

    timings = {}
    model = build_model(checkpoint, None)
    train_res = int(model.patch_embed.img_size[0])
    bank_path = store_dir / f"bank_r{train_res}.pt"
    if not args.analyze_only:
        started = time.perf_counter()
        if bank_path.is_file():
            blob = torch.load(bank_path)
            bank = (blob["features"].cuda(), blob["labels"])
        else:
            bank = bank_pass(model, make_forward(model, amp_dtype), cfg, model_data_config(model),
                             args.crop_pct, run_dir, args)
            torch.save({"features": bank[0].cpu(), "labels": bank[1]}, bank_path)
        timings["bank"] = time.perf_counter() - started
        print(f"bank {tuple(bank[0].shape)}: {timings['bank']:.0f} s", flush=True)
        for res in args.resolutions:
            if (store_dir / f"holdout_r{res}.pt").is_file():
                print(f"r{res}: reusing the saved store", flush=True)
                continue
            del model
            torch.cuda.empty_cache()
            model = build_model(checkpoint, res)
            started = time.perf_counter()
            store = holdout_pass(model, make_forward(model, amp_dtype), cfg, model_data_config(model),
                                 args.crop_pct, samples, bank, args)
            torch.save(store, store_dir / f"holdout_r{res}.pt")
            timings[f"r{res}"] = time.perf_counter() - started
            print(f"r{res}: {timings[f'r{res}']:.0f} s", flush=True)
        del model, bank
        torch.cuda.empty_cache()

    bank_labels = torch.load(bank_path)["labels"]
    stores = {str(res): torch.load(store_dir / f"holdout_r{res}.pt") for res in args.resolutions}
    with torch.inference_mode():
        report = analyze(stores, bank_labels, labels, classes, train_res, args)

    gate = None
    zoom_json = run_dir / "analysis" / "eval_zoom.json"
    if args.limit is None and zoom_json.is_file():
        previous = json.loads(zoom_json.read_text(encoding="utf-8"))["results"]
        tag = f"crop{args.crop_pct:g}"
        pairs = {f"r{train_res}/plain": f"{tag}/plain",
                 f"r{train_res}/plain+zoom": f"{tag}/cam_m{DEFAULT_MASS:g}/plain+zoom"}
        gate = {ours: {"reproduced": report["results"][ours]["all"], "eval_zoom": previous[theirs]["all"]}
                for ours, theirs in pairs.items() if theirs in previous}
        for ours, values in gate.items():
            if abs(values["reproduced"] - values["eval_zoom"]) > args.gate_tolerance:
                raise RuntimeError(f"gate failed on {ours}: {values}")

    for name, value in report["results"].items():
        if "_lam" not in name:
            print(f"{name:52s} {value['all'] * 100:7.3f}  (even {value['even'] * 100:.3f}, odd {value['odd'] * 100:.3f})")
    print("best 2-pass on even:", report["best_two_pass_on_even"],
          f"z vs ref {report['best_two_pass_paired_z_vs_ref']:.1f}")
    for base_name, blend in report["knn_blends"].items():
        value = blend["result"]
        print(f"{base_name}: {blend['chosen_on_even']}  all {value['all'] * 100:.3f} odd {value['odd'] * 100:.3f}"
              f"  z vs ref {blend['paired_z_vs_ref']:.1f}")

    from cs604.engine.loop import _git_metadata

    report.update({
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": checkpoint.get("step"),
        "samples": len(samples),
        "settings": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "zoom": {"map": "cam", "mass": DEFAULT_MASS, "min_side": DEFAULT_MIN_SIDE, "factor": DEFAULT_FACTOR},
        "gate": gate,
        "git": _git_metadata(),
        "device": torch.cuda.get_device_name(0),
        "elapsed_sec": timings,
    })
    output = run_dir / "analysis" / f"eval_res_knn{suffix}.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
