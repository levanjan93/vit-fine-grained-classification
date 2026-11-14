"""Train-vs-holdout fit check: plain-view top-1 and CE loss on 5 train images/class vs the holdout.

Usage: eval_fit.py <run_dir> <crop_pct>. Uses <run_dir>/best.pt; writes <run_dir>/analysis/fit_check_crop<crop>.json.
"""
import json, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cs604.data.dataset import CS604ImageDataset, load_split_index
from cs604.models.build import build_from_config, load_model_state, model_data_config
from timm.data import create_transform


@torch.inference_mode()
def score(model, cfg, split, samples, crop, amp):
    data_cfg = model_data_config(model)
    tf = create_transform(**{**data_cfg, "crop_pct": crop}, is_training=False)
    ds = CS604ImageDataset(cfg["data_root"], split, transform=tf, samples=samples)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
    hits = loss = top5 = 0
    for images, labels, _ in loader:
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=amp):
            logits = model(images).float()
        hits += int((logits.argmax(1) == labels).sum())
        top5 += int((logits.topk(5, 1).indices == labels[:, None]).any(1).sum())
        loss += float(F.cross_entropy(logits, labels, reduction="sum"))
    n = len(samples)
    return {"top1": 100 * hits / n, "top5": 100 * top5 / n, "ce": loss / n, "n": n}


def main():
    run_dir = Path(sys.argv[1])
    crop = float(sys.argv[2])
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_from_config(cfg)
    load_model_state(model, ckpt)
    model.cuda().eval()
    amp = torch.bfloat16
    train = load_split_index(run_dir / "splits" / "internal_train.txt")
    per_class, sample = {}, []
    for path, label in train:  # first 5 per class in file order (deterministic)
        if per_class.get(label, 0) < 5:
            per_class[label] = per_class.get(label, 0) + 1
            sample.append((path, label))
    val = load_split_index(run_dir / "splits" / "internal_val.txt")
    out = {"run": run_dir.name, "step": ckpt.get("step"), "crop": crop}
    for name, split, s in [("holdout", "internal_val", val), ("train5", "internal_train", sample)]:
        t = time.perf_counter()
        out[name] = score(model, cfg, split, s, crop, amp)
        out[name]["seconds"] = round(time.perf_counter() - t)
        print(name, out[name], flush=True)
    out["gap_top1"] = out["train5"]["top1"] - out["holdout"]["top1"]
    (run_dir / "analysis").mkdir(exist_ok=True)
    (run_dir / "analysis" / f"fit_check_crop{crop:g}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
