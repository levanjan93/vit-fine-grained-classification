"""Error analysis of a checkpoint on its internal holdout (evaluation only).

Stage 1 scores the run's ``splits/internal_val.txt`` once at ``--crop-pct`` and
saves the top-10 classes per image to ``<run>/analysis/errors_top10.pt`` (reused
on later calls). If ``analysis/eval_tta.json`` has the same crop, the plain
top-1 must reproduce it (``--gate-tolerance``).

Stage 2 writes ``<run>/analysis/errors.json`` (top-1/top-5, rank of the true
class when wrong, per-class accuracy, most-confused pairs, accuracy by aspect
ratio and by confidence) and ``errors.html``: a contact sheet of sampled errors.
Each row shows the holdout image with the eval crop drawn on it, then training
images of the true and of the predicted class. Rows can be tagged in the page;
tags stay in the browser and download as CSV.

Only holdout images (labels we own) are shown; never use val/test images here.
"""

from __future__ import annotations

import argparse
import base64
import collections
import html
import io
import json
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cs604.data.dataset import CS604ImageDataset, load_split_index  # noqa: E402
from cs604.models.build import build_from_config, load_model_state, model_data_config  # noqa: E402

TAGS = [
    "look-alike species", "small/distant subject", "crop cuts subject",
    "not the organism (tracks, habitat...)", "multiple subjects", "blurry/dark",
    "likely label noise", "true class images inconsistent", "other",
]


@torch.inference_mode()
def predict(checkpoint_path: Path, crop_pct: float, batch_size: int, workers: int) -> dict:
    from timm.data import create_transform

    run_dir = checkpoint_path.parent
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model = build_from_config(cfg)
    load_model_state(model, checkpoint)
    model.cuda().eval()
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(str(cfg.get("precision", "bf16")))
    samples = load_split_index(run_dir / "splits" / "internal_val.txt")
    data_cfg = model_data_config(model)
    transform = create_transform(**{**data_cfg, "crop_pct": crop_pct}, is_training=False)
    dataset = CS604ImageDataset(cfg["data_root"], "internal_val", transform=transform, samples=samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    top_idx, top_prob = [], []
    started = time.perf_counter()
    for index, (images, _labels, _paths) in enumerate(loader, start=1):
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            probs = model(images.cuda(non_blocking=True)).float().softmax(1)
        values, indices = probs.topk(10, dim=1)
        top_idx.append(indices.cpu())
        top_prob.append(values.cpu())
        if index % 50 == 0:
            print(f"scored {index * batch_size}/{len(samples)}", flush=True)
    return {
        "checkpoint": str(checkpoint_path), "crop_pct": crop_pct,
        "image_size": int(data_cfg["input_size"][-1]), "data_root": cfg["data_root"],
        "paths": [path for path, _label in samples],
        "labels": torch.tensor([label for _path, label in samples]),
        "top_idx": torch.cat(top_idx), "top_prob": torch.cat(top_prob),
        "elapsed_sec": time.perf_counter() - started,
    }


def _thumb(path: Path, size: int, crop_pct: float | None = None) -> str:
    image = Image.open(path).convert("RGB")
    if crop_pct is not None:
        # The eval view: resize the short side to image_size / crop_pct, center crop image_size,
        # i.e. a centered square of side crop_pct * short side.
        width, height = image.size
        side = crop_pct * min(width, height)
        left, top = (width - side) / 2, (height - side) / 2
        ImageDraw.Draw(image).rectangle(
            [left, top, left + side, top + side], outline=(255, 64, 64), width=max(2, int(min(width, height) / 100))
        )
    image.thumbnail((size, size))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def analyze(pred: dict, train_samples: list[tuple[str, int]], rows: int, seed: int) -> tuple[dict, list[dict]]:
    labels, top_idx, top_prob = pred["labels"], pred["top_idx"], pred["top_prob"]
    data_root = Path(pred["data_root"])
    hit = top_idx == labels[:, None]
    correct = hit[:, 0]
    n = len(labels)
    rank = torch.where(hit.any(1), hit.float().argmax(1) + 1, torch.full_like(labels, 11))
    wrong = (~correct).nonzero().flatten().tolist()

    per_class = collections.defaultdict(list)
    for label, ok in zip(labels.tolist(), correct.tolist()):
        per_class[label].append(ok)
    class_acc = {label: sum(v) / len(v) for label, v in per_class.items()}
    acc_hist = collections.Counter(round(acc, 2) for acc in class_acc.values())

    pairs = collections.Counter()
    for i in wrong:
        pairs[(int(labels[i]), int(top_idx[i, 0]))] += 1
    undirected = collections.Counter()
    for (a, b), count in pairs.items():
        undirected[tuple(sorted((a, b)))] += count
    mutual = sum(1 for (a, b) in undirected if pairs[(a, b)] and pairs[(b, a)])

    sizes = [Image.open(data_root / path).size for path in pred["paths"]]
    aspect = torch.tensor([max(w, h) / min(w, h) for w, h in sizes])
    short = torch.tensor([min(w, h) for w, h in sizes])

    def binned(values, edges):
        out = []
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (values >= low) & (values < high)
            if mask.any():
                out.append({"range": [low, high], "images": int(mask.sum()),
                            "accuracy": float(correct[mask].float().mean())})
        return out

    conf = top_prob[:, 0]
    stats = {
        "checkpoint": pred["checkpoint"], "crop_pct": pred["crop_pct"], "images": n,
        "top1": float(correct.float().mean()), "top5": float(hit[:, :5].any(1).float().mean()),
        "top10": float(hit.any(1).float().mean()),
        "wrong": len(wrong),
        "rank_of_true_when_wrong": {str(r): int(((rank == r) & ~correct).sum()) for r in range(2, 12)},
        "rank_note": "rank 11 = not in the top 10",
        "classes_by_holdout_accuracy": {str(k): acc_hist[k] for k in sorted(acc_hist)},
        "distinct_confused_pairs": len(undirected),
        "mutually_confused_pairs": mutual,
        "errors_in_top_100_pairs": sum(c for _p, c in undirected.most_common(100)),
        "top_confused_pairs": [{"classes": list(p), "errors": c, "a_to_b": pairs[p], "b_to_a": pairs[p[::-1]]}
                               for p, c in undirected.most_common(30)],
        "accuracy_by_confidence": binned(conf, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.01]),
        "accuracy_by_aspect_ratio": binned(aspect, [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 99]),
        "accuracy_by_short_side_px": binned(short.float(), [0, 200, 300, 400, 500, 10_000]),
        "error_mean_confidence": float(conf[~correct].mean()),
        "correct_mean_confidence": float(conf[correct].mean()),
    }

    # Contact-sheet sample: random errors, confident errors, errors from the most-confused pairs.
    rng = random.Random(seed)
    confident = [i for i in wrong if conf[i] >= 0.7]
    top_pairs = {p for p, _c in undirected.most_common(50)}
    in_pairs = [i for i in wrong if tuple(sorted((int(labels[i]), int(top_idx[i, 0])))) in top_pairs]
    chosen, kinds = [], {}
    for kind, pool, k in (("confident (p >= 0.7)", confident, rows * 3 // 10),
                          ("top-50 confused pair", in_pairs, rows * 3 // 10),
                          ("random error", wrong, rows)):
        pool = [i for i in pool if i not in kinds]
        for i in rng.sample(pool, min(len(pool), k if kind != "random error" else rows - len(chosen))):
            kinds[i] = kind
            chosen.append(i)
    by_class = collections.defaultdict(list)
    for path, label in train_samples:
        by_class[label].append(path)
    sheet = []
    for i in chosen:
        true, guess = int(labels[i]), int(top_idx[i, 0])
        sheet.append({
            "index": i, "path": pred["paths"][i], "kind": kinds[i], "true": true, "pred": guess,
            "p_pred": float(conf[i]),
            "p_true": float(top_prob[i][hit[i]].sum()),
            "rank_true": int(rank[i]), "size": list(sizes[i]),
            "true_examples": rng.sample(by_class[true], 3), "pred_examples": rng.sample(by_class[guess], 3),
        })
    return stats, sheet


def write_html(out: Path, stats: dict, sheet: list[dict], data_root: Path, crop_pct: float) -> None:
    def img(path, size, crop=None):
        return f'<img src="{_thumb(data_root / path, size, crop)}" loading="lazy">'

    tag_buttons = "".join(f'<label><input type="checkbox" value="{html.escape(t)}">{html.escape(t)}</label>'
                          for t in TAGS)
    body = []
    for n, row in enumerate(sheet, start=1):
        true_rank = "not in top 10" if row["rank_true"] == 11 else f"rank {row['rank_true']}"
        body.append(f"""
<section class="row" data-key="{row['index']}">
  <div class="meta"><b>#{n}</b> {html.escape(row['kind'])} &middot; true <b>{row['true']}</b> ({true_rank},
  p={row['p_true']:.2f}) &rarr; predicted <b>{row['pred']}</b> (p={row['p_pred']:.2f}) &middot;
  {row['size'][0]}&times;{row['size'][1]} &middot; <span class="path">{html.escape(row['path'])}</span></div>
  <div class="imgs">
    <figure class="query">{img(row['path'], 320, crop_pct)}<figcaption>holdout (red = eval crop)</figcaption></figure>
    <figure class="group true">{''.join(img(p, 170) for p in row['true_examples'])}<figcaption>train: true class {row['true']}</figcaption></figure>
    <figure class="group pred">{''.join(img(p, 170) for p in row['pred_examples'])}<figcaption>train: predicted class {row['pred']}</figcaption></figure>
  </div>
  <div class="tags">{tag_buttons}<input class="note" placeholder="note"></div>
</section>""")
    summary = {k: stats[k] for k in ("images", "top1", "top5", "top10", "wrong",
                                      "error_mean_confidence", "correct_mean_confidence")}
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Holdout errors</title>
<style>
:root {{ --bg:#fafaf9; --fg:#1c1917; --muted:#78716c; --card:#fff; --line:#e7e5e4; --true:#15803d; --pred:#b91c1c; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1c1917; --fg:#f5f5f4; --muted:#a8a29e; --card:#292524; --line:#44403c; --true:#4ade80; --pred:#f87171; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif; margin:0 auto; max-width:1200px; padding:16px; }}
.row {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; margin:12px 0; }}
.meta {{ margin-bottom:8px; }} .path {{ color:var(--muted); font-size:12px; }}
.imgs {{ display:flex; flex-wrap:wrap; gap:12px; align-items:flex-start; }}
figure {{ margin:0; }} figcaption {{ color:var(--muted); font-size:12px; }}
.group img {{ height:120px; margin-right:4px; border-radius:4px; }} .query img {{ max-height:240px; border-radius:4px; }}
.true figcaption {{ color:var(--true); }} .pred figcaption {{ color:var(--pred); }}
.tags {{ display:flex; flex-wrap:wrap; gap:4px 12px; margin-top:8px; font-size:13px; }}
.note {{ flex:1 1 200px; background:var(--bg); color:var(--fg); border:1px solid var(--line); border-radius:4px; padding:2px 6px; }}
header {{ position:sticky; top:0; background:var(--bg); padding:8px 0; border-bottom:1px solid var(--line); }}
button {{ font:inherit; padding:4px 10px; }} pre {{ white-space:pre-wrap; font-size:12px; }}
</style></head><body>
<header><b>Holdout errors</b> &middot; {len(sheet)} sampled of {stats['wrong']} &middot; top-1 {100*stats['top1']:.2f}%,
top-5 {100*stats['top5']:.2f}% &middot; crop {crop_pct} &middot; <span id="count"></span>
<button id="dl">Download tags (CSV)</button></header>
<details><summary>Summary statistics</summary><pre>{html.escape(json.dumps(summary, indent=2))}</pre>
<p>Full statistics: errors.json beside this file.</p></details>
{''.join(body)}
<script>
const KEY = "cs604-error-tags:" + {json.dumps(stats['checkpoint'])};
let store = {{}};
try {{ store = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{}}
const save = () => {{ try {{ localStorage.setItem(KEY, JSON.stringify(store)); }} catch (e) {{}} count(); }};
const count = () => {{ document.getElementById("count").textContent =
  Object.values(store).filter(v => v.tags.length || v.note).length + " tagged"; }};
document.querySelectorAll(".row").forEach(row => {{
  const k = row.dataset.key, s = store[k] || {{tags: [], note: ""}};
  row.querySelectorAll("input[type=checkbox]").forEach(cb => {{
    cb.checked = s.tags.includes(cb.value);
    cb.onchange = () => {{ s.tags = [...row.querySelectorAll("input:checked")].map(x => x.value); store[k] = s; save(); }};
  }});
  const note = row.querySelector(".note"); note.value = s.note;
  note.oninput = () => {{ s.note = note.value; store[k] = s; save(); }};
}});
document.getElementById("dl").onclick = () => {{
  const rows = [["row", "holdout_index", "path", "kind", "true", "pred", "tags", "note"]];
  document.querySelectorAll(".row").forEach((row, n) => {{
    const s = store[row.dataset.key] || {{tags: [], note: ""}};
    const m = {json.dumps({str(r['index']): [r['path'], r['kind'], r['true'], r['pred']] for r in sheet})}[row.dataset.key];
    rows.push([n + 1, row.dataset.key, ...m, s.tags.join("; "), s.note]);
  }});
  const csv = rows.map(r => r.map(v => '"' + String(v).replace(/"/g, '""') + '"').join(",")).join("\\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], {{type: "text/csv"}})); a.download = "error_tags.csv"; a.click();
}};
count();
</script></body></html>"""
    out.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--crop-pct", type=float, default=0.75)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gate-tolerance", type=float, default=0.0005)
    args = parser.parse_args()

    run_dir = args.checkpoint.parent
    analysis = run_dir / "analysis"
    analysis.mkdir(exist_ok=True)
    cache = analysis / "errors_top10.pt"
    pred = torch.load(cache, weights_only=False) if cache.is_file() else None
    if pred is None or pred["crop_pct"] != args.crop_pct or pred["checkpoint"] != str(args.checkpoint):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        pred = predict(args.checkpoint, args.crop_pct, args.batch_size, args.num_workers)
        torch.save(pred, cache)
    top1 = float((pred["top_idx"][:, 0] == pred["labels"]).float().mean())
    tta = analysis / "eval_tta.json"
    if tta.is_file():
        reference = json.loads(tta.read_text(encoding="utf-8")).get("results", {}).get(f"crop_{args.crop_pct:g}")
        if reference and abs(reference["acc_plain"] - top1) > args.gate_tolerance:
            raise RuntimeError(f"gate failed: top-1 {top1:.5f} vs eval_tta.json {reference['acc_plain']:.5f}")
        print(f"gate: top-1 {top1:.5f} vs eval_tta.json {reference['acc_plain'] if reference else None}")

    train_samples = load_split_index(run_dir / "splits" / "internal_train.txt")
    stats, sheet = analyze(pred, train_samples, args.rows, args.seed)
    (analysis / "errors.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_html(analysis / "errors.html", stats, sheet, Path(pred["data_root"]), args.crop_pct)
    print(json.dumps({k: stats[k] for k in ("top1", "top5", "top10", "wrong", "rank_of_true_when_wrong",
                                            "classes_by_holdout_accuracy", "mutually_confused_pairs",
                                            "errors_in_top_100_pairs")}, indent=1))
    print(f"wrote {analysis / 'errors.json'} and {analysis / 'errors.html'}")


if __name__ == "__main__":
    main()
