"""Evaluation-only: re-check the kNN re-rank weight (beta) of the team's submission pipeline.

The team's CodaBench pipeline (written by a teammate) takes the
crop 0.9 + zoom probabilities ``p = (softmax(plain) + softmax(zoom)) / 2`` and re-scores their
top ``K`` classes with

    score(c) = log p(c) + beta * knn(c),   knn(c) = mean of the m best cosines between the
                                                    plain-view feature and c's training photos

K = 5, m = 3, beta = 20 were fixed on the team's all-class holdout (LoRA r24 at 336, no zoom
in the beta grid 10 / 20 / 40). This script asks whether the 448 checkpoints and the zoom view
want another beta, on this repo's internal holdout (same 5 photos per class for every run).

The bank is every non-holdout photo (``--bank-split``, default the run's own
``internal_train.txt``; pass a 40-per-class list for the 10-per-class screens), embedded with
the plain eval view at the model's default crop, as predict_rerank does. Settings are chosen on
one half of the holdout classes (alternating class order) and scored on the other, then
swapped, as rerank_knn.py / predict_rerank.py do.

Writes ``<run>/analysis/rerank_beta.json``; the bank and holdout store go to
``analysis/rerank_beta/`` so ``--analyze-only`` rescores without the GPU passes.
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
from cs604.engine.zoom import DEFAULT_FACTOR, DEFAULT_MASS, TwoViews, zoom_view  # noqa: E402
from cs604.models.build import build_from_config, load_model_state, model_data_config  # noqa: E402

TOP = 10  # candidates stored per view; K is swept up to this
KS = (3, 5, 10)
MS = (1, 3, 5)
BETAS = (0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80)


def make_forward(model, amp_dtype):
    def forward(images):
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            tokens = model.forward_features(images)
            pooled = model.forward_head(tokens, pre_logits=True)
            logits = model.get_classifier()(pooled)
        return tokens, F.normalize(pooled.float(), dim=-1), logits.float()
    return forward


@torch.inference_mode()
def bank_pass(forward, cfg, data_cfg, samples, args):
    from timm.data import create_transform

    dataset = CS604ImageDataset(cfg["data_root"], "internal_train",
                                transform=create_transform(**data_cfg, is_training=False), samples=samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    features, offset = None, 0
    for images, _labels, _paths in loader:
        _tokens, pooled, _logits = forward(images.cuda(non_blocking=True))
        if features is None:
            features = torch.empty(len(samples), pooled.shape[1], dtype=torch.float16, device="cuda")
        features[offset: offset + len(pooled)] = pooled.half()
        offset += len(pooled)
    return features.cpu(), torch.tensor([label for _path, label in samples], dtype=torch.long)


@torch.inference_mode()
def holdout_pass(model, forward, cfg, data_cfg, samples, args) -> dict:
    """Plain-view features and the top-``TOP`` log-probabilities of the plain and plain+zoom views.

    With ``--flip`` also the flipped plain and flipped zoom views: ``zoom3`` averages the
    softmax of plain, flip and zoom, ``zoom4`` adds the flipped zoom; ``features_pf`` is the
    normalized sum of the plain and flipped plain features (a second kNN query).
    """

    from timm.data import create_transform

    size = int(data_cfg["input_size"][-1])
    big = {**data_cfg, "input_size": (3, size * DEFAULT_FACTOR, size * DEFAULT_FACTOR)}
    transform = TwoViews(create_transform(**data_cfg, is_training=False), create_transform(**big, is_training=False))
    dataset = CS604ImageDataset(cfg["data_root"], "internal_val", transform=transform, samples=samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    out = {}
    for (small, large), _labels, _paths in loader:
        small = small.cuda(non_blocking=True)
        tokens, feature, logits = forward(small)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zoomed, _boxes = zoom_view(model, tokens, logits, large.cuda(non_blocking=True), mass=DEFAULT_MASS)
        _tokens, _feature, z_logits = forward(zoomed)
        p, pz = logits.softmax(1), z_logits.softmax(1)
        views = {"plain": logits.log_softmax(1), "zoom": ((p + pz) / 2).clamp_min(1e-12).log()}
        features = {"features": feature}
        if args.flip:
            _tokens, f_flip, f_logits = forward(small.flip(-1))
            _tokens, _feature, zf_logits = forward(zoomed.flip(-1))
            pf, pzf = f_logits.softmax(1), zf_logits.softmax(1)
            views["zoom3"] = ((p + pf + pz) / 3).clamp_min(1e-12).log()
            views["zoom4"] = ((p + pf + pz + pzf) / 4).clamp_min(1e-12).log()
            features["features_pf"] = F.normalize(feature + f_flip, dim=-1)
        for name, lp in views.items():
            values, index = lp.topk(TOP, dim=1)
            out.setdefault(f"{name}/lp", []).append(values.cpu())
            out.setdefault(f"{name}/i", []).append(index.cpu())
        for name, f in features.items():
            out.setdefault(name, []).append(f.half().cpu())
    return {key: torch.cat(value) for key, value in out.items()}


def class_index(labels: torch.Tensor, classes: int) -> torch.Tensor:
    """(C, L) bank rows of each class, padded with -1."""

    order = torch.argsort(labels, stable=True)
    counts = torch.bincount(labels, minlength=classes)
    index = torch.full((classes, int(counts.max())), -1, dtype=torch.long)
    starts = torch.cumsum(counts, 0) - counts
    for c in range(classes):
        index[c, : counts[c]] = order[starts[c]: starts[c] + counts[c]]
    return index


@torch.inference_mode()
def knn_means(features, cand, bank, index, chunk=2000) -> dict[int, torch.Tensor]:
    """{m: (N, TOP) mean of the m best cosines to each candidate class}, as predict_rerank.knn_scores."""

    out = {m: torch.empty(cand.shape, dtype=torch.float32) for m in MS}
    for start in range(0, len(cand), chunk):
        f = features[start: start + chunk].cuda().float()
        rows = index[cand[start: start + chunk].cuda()]                     # (B, TOP, L)
        valid = rows >= 0
        sims = torch.einsum("bd,bkld->bkl", f, bank[rows.clamp_min(0)].float()).masked_fill(~valid, -2.0)
        ranked = sims.sort(dim=-1, descending=True).values
        for m in MS:
            best = ranked[..., :m]
            out[m][start: start + chunk] = (best.masked_fill(best <= -2.0, float("nan"))
                                            .nanmean(-1).nan_to_num(-1.0).cpu())
    return out


def score_grid(store, variants, labels) -> dict:
    """``variants``: name -> (view, {m: knn means}); every K 5 / m 3 row is also compared, photo by
    photo, with the current submission (``zoom`` view, plain-feature query, beta 20)."""

    classes_sorted = sorted(set(labels.tolist()))
    half_of = {c: i % 2 for i, c in enumerate(classes_sorted)}
    half = torch.tensor([half_of[c] for c in labels.tolist()])

    def hits(view, knn_m, k, beta):
        lp, cand = store[f"{view}/lp"], store[f"{view}/i"]
        score = lp[:, :k] + beta * knn_m[:, :k]
        return cand[:, :k].gather(1, score.argmax(1, keepdim=True)).squeeze(1) == labels

    reference = hits("zoom", variants["zoom"][1][3], 5, 20)
    report = {}
    for name, (view, knn) in variants.items():
        cand = store[f"{view}/i"]
        base_ok = cand[:, 0] == labels
        grid = []
        for k, m, beta in itertools.product(KS, MS, BETAS):
            ok = hits(view, knn[m], k, beta)
            row = {"K": k, "m": m, "beta": beta, "all": float(ok.float().mean()),
                   "half0": float(ok[half == 0].float().mean()), "half1": float(ok[half == 1].float().mean()),
                   "fixes": int((ok & ~base_ok).sum()), "breaks": int((~ok & base_ok).sum())}
            if k == 5 and m == 3:
                diff = ok.float() - reference.float()
                row["vs_zoom_b20_points"] = 100 * float(diff.mean())
                row["vs_zoom_b20_z"] = float(diff.mean() / (diff.std() / len(diff) ** 0.5).clamp_min(1e-12))
            grid.append(row)

        def cross(rows):
            picks = []
            for tune, test in ((0, 1), (1, 0)):
                best = max(rows, key=lambda r: (r[f"half{tune}"], -r["beta"]))  # ties -> smaller beta
                before = float(base_ok[half == test].float().mean())
                picks.append({"tuned_on_half": tune, "K": best["K"], "m": best["m"], "beta": best["beta"],
                              "test_half_before": before, "test_half_after": best[f"half{test}"],
                              "test_half_gain_points": 100 * (best[f"half{test}"] - before)})
            return {"picks": picks, "held_out_gain_points": sum(p["test_half_gain_points"] for p in picks) / 2}

        k5m3 = [r for r in grid if r["K"] == 5 and r["m"] == 3]
        b20 = next(r for r in k5m3 if r["beta"] == 20)
        report[name] = {
            "view": view,
            "base_top1": float(base_ok.float().mean()),
            "k5_m3_by_beta": {r["beta"]: {key: r[key] for key in ("all", "half0", "half1", "fixes", "breaks",
                                                                  "vs_zoom_b20_points", "vs_zoom_b20_z")}
                              for r in k5m3},
            "k5_m3_beta20_gain_points": 100 * (b20["all"] - float(base_ok.float().mean())),
            "beta_only_cross": cross(k5m3),
            "full_grid_cross": cross(grid),
            "best_on_all_optimistic": max(grid, key=lambda r: r["all"]),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank-split", type=Path, default=None,
                        help="non-holdout photo list for the bank (default: the run's internal_train.txt)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="first N holdout images, N*8 bank images (smoke)")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--flip", action="store_true",
                        help="also flipped plain and zoom views (zoom3, zoom4) and the plain+flip kNN query")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    run_dir = args.checkpoint.parent
    suffix = "" if args.limit is None else "_smoke"
    store_dir = run_dir / "analysis" / f"rerank_beta{suffix}"
    store_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    classes = int(cfg["num_classes"])
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(str(cfg.get("precision", "bf16")))
    holdout = load_split_index(run_dir / "splits" / "internal_val.txt")
    bank_samples = load_split_index(args.bank_split or run_dir / "splits" / "internal_train.txt")
    held = {path for path, _label in holdout}
    if any(path in held for path, _label in bank_samples):
        raise RuntimeError("the bank list contains holdout photos")
    if args.limit is not None:
        holdout, bank_samples = holdout[: args.limit], bank_samples[: args.limit * 8]
    labels = torch.tensor([label for _path, label in holdout])

    timings = {}
    bank_path = store_dir / "bank.pt"
    store_path = store_dir / ("holdout_flip.pt" if args.flip else "holdout.pt")
    if not args.analyze_only:
        model = build_from_config(cfg)
        load_model_state(model, checkpoint)
        model = model.cuda().eval()
        forward, data_cfg = make_forward(model, amp_dtype), model_data_config(model)
        if not bank_path.is_file():
            started = time.perf_counter()
            features, bank_labels = bank_pass(forward, cfg, data_cfg, bank_samples, args)
            torch.save({"features": features, "labels": bank_labels, "split": str(args.bank_split)}, bank_path)
            timings["bank_s"] = time.perf_counter() - started
            print(f"bank {tuple(features.shape)}: {timings['bank_s']:.0f} s", flush=True)
        if not store_path.is_file():
            started = time.perf_counter()
            torch.save(holdout_pass(model, forward, cfg, data_cfg, holdout, args), store_path)
            timings["holdout_s"] = time.perf_counter() - started
            print(f"holdout: {timings['holdout_s']:.0f} s", flush=True)
        del model
        torch.cuda.empty_cache()

    blob, store = torch.load(bank_path), torch.load(store_path)
    bank = blob["features"].cuda()
    index = class_index(blob["labels"], classes).cuda()
    variants = {}
    for view in [v for v in ("plain", "zoom", "zoom3", "zoom4") if f"{v}/i" in store]:
        for query in [q for q in ("features", "features_pf") if q in store]:
            name = view if query == "features" else f"{view}+query_pf"
            variants[name] = (view, knn_means(store[query], store[f"{view}/i"], bank, index))
    report = score_grid(store, variants, labels)

    for name, r in report.items():
        print(f"\n{name}: top-1 {100 * r['base_top1']:.3f}; K5 m3 by beta (all / half0 / half1, fixes-breaks,"
              " vs zoom b20):")
        for beta, v in r["k5_m3_by_beta"].items():
            print(f"  beta {beta:>3}: {100 * v['all']:.3f} / {100 * v['half0']:.3f} / {100 * v['half1']:.3f}"
                  f"  +{v['fixes']} -{v['breaks']}  {v['vs_zoom_b20_points']:+.3f} (z {v['vs_zoom_b20_z']:.1f})")
        for name in ("beta_only_cross", "full_grid_cross"):
            c = r[name]
            chosen = ", ".join(f"K{p['K']} m{p['m']} b{p['beta']}" for p in c["picks"])
            print(f"  {name}: chose [{chosen}], held-out gain {c['held_out_gain_points']:+.2f}")

    from cs604.engine.loop import _git_metadata

    report = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint),
              "image_size": int(cfg.get("image_size", 0)), "holdout_photos": len(holdout),
              "bank_photos": int(len(blob["labels"])), "bank_split": blob.get("split"),
              "pipeline": "crop default (0.9) plain; zoom = (softmax plain + softmax cam m0.7 zoom) / 2; "
                          "zoom3 adds the flipped plain, zoom4 also the flipped zoom (mean of softmaxes); "
                          "+query_pf = kNN query normalize(plain + flipped plain feature)",
              "grid": {"K": KS, "m": MS, "beta": BETAS}, "timings": timings, "git": _git_metadata(),
              "results": report}
    out = run_dir / "analysis" / f"rerank_beta{'_flip' if args.flip else ''}{suffix}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
