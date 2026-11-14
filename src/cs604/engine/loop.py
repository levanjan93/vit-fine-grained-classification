"""Full fine-tuning loop with step checkpoints and a metrics log.

Checkpoints are written often enough that a Colab disconnect or a laptop sleep
can resume the same run. The metrics CSV is the loss curve for the report.
"""

from __future__ import annotations

import csv
import json
import math
import random
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from cs604.data.dataset import (
    CS604ImageDataset,
    load_split_index,
    stratified_split,
    write_split_index,
)
from cs604.models.build import (
    CosineClassifier,
    adapter_dim_from_config,
    apply_class_prototypes,
    build_from_config,
    calibrate_adapters_pca,
    load_head_init,
    build_vit,
    load_model_state,
    lora_blocks_from_config,
    lora_targets_from_config,
    model_data_config,
    optimizer_param_groups,
    parameter_report,
)
from cs604.engine.baseline import (
    LEGACY_PROTOTYPE_BACKBONE,
    assess_initial_accuracy,
    baseline_provenance,
    baseline_requested,
    check_prototype_split,
    validate_baseline_setup,
)
from cs604.engine.budget import (
    budget_wall_seconds,
    effective_step,
    lr_schedule_factor,
    lr_schedule_from_config,
    training_limit_seconds,
)
from cs604.engine.checkpoint_io import save_training_checkpoint


class SyntheticImages(Dataset):
    """Fixed random tensors so the GPU path can be tested before images are extracted."""

    def __init__(self, length: int, num_classes: int, image_size: int):
        self.length = length
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        generator = torch.Generator()
        generator.manual_seed(index)
        image = torch.randn(3, self.image_size, self.image_size, generator=generator)
        label = index % self.num_classes
        return image, label, f"synthetic/{index:06d}.jpg"


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_loader(cfg: dict, train_transform) -> DataLoader:
    if cfg.get("synthetic"):
        dataset: Dataset = SyntheticImages(
            length=int(cfg.get("limit_samples") or 64),
            num_classes=int(cfg["num_classes"]),
            image_size=int(cfg["image_size"]),
        )
    else:
        dataset = CS604ImageDataset(cfg["data_root"], "train", transform=train_transform)
        limit = cfg.get("limit_samples")
        if limit is not None and int(limit) < len(dataset):
            rng = random.Random(int(cfg.get("seed", 0)))
            dataset.samples = rng.sample(dataset.samples, int(limit))

    workers = int(cfg.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _save(path: Path, model, optimizer, step: int, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        },
        path,
    )


def _run_training_legacy(cfg: dict) -> Path:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. Install a CUDA build of PyTorch (see README); "
            "on Colab, switch the runtime to GPU."
        )

    _seed(int(cfg.get("seed", 0)))
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_vit(
        checkpoint=cfg["timm_checkpoint"],
        num_classes=int(cfg["num_classes"]),
        grad_checkpointing=bool(cfg.get("grad_checkpointing", False)),
    )
    model.to(device)
    report = parameter_report(model)

    from timm.data import create_transform, resolve_model_data_config

    transform = None if cfg.get("synthetic") else create_transform(
        **resolve_model_data_config(model),
        is_training=True,
    )
    loader = _build_loader(cfg, transform)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.05)),
    )
    start_step = 0
    resume = cfg.get("resume")
    if resume:
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"])

    precision = str(cfg.get("precision", "bf16"))
    use_amp = precision == "bf16"
    autocast_dtype = torch.bfloat16 if use_amp else torch.float32
    grad_accum = max(1, int(cfg.get("grad_accum", 1)))
    max_steps = cfg.get("max_steps")
    max_steps = None if max_steps is None else int(max_steps)
    log_every = int(cfg.get("log_every", 10))
    ckpt_every = int(cfg.get("ckpt_every", 100))
    epochs = int(cfg.get("epochs", 1))

    props = torch.cuda.get_device_properties(0)
    hardware = {
        "device": torch.cuda.get_device_name(0),
        "vram_gb": round(props.total_memory / (1024**3), 2),
        "precision": precision,
        "batch_size": int(cfg["batch_size"]),
        "grad_accum": grad_accum,
        **report,
    }
    (output_dir / "hardware.json").write_text(json.dumps(hardware, indent=2), encoding="utf-8")
    print(
        f"device={hardware['device']} vram={hardware['vram_gb']}GB "
        f"trainable={report['trainable_params']:,} ({report['trainable_pct']:.2f}%)"
    )

    metrics_path = output_dir / "metrics.csv"
    metrics = metrics_path.open("a" if start_step else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        metrics,
        fieldnames=["step", "loss", "images_per_sec", "max_mem_gb", "elapsed_sec"],
    )
    if start_step == 0:
        writer.writeheader()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    micro = 0
    images_since_log = 0
    log_t0 = time.perf_counter()
    run_t0 = log_t0
    window_loss = 0.0
    window_steps = 0
    micro_loss = 0.0

    for epoch in range(epochs):
        for images, labels, _paths in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_amp):
                loss = F.cross_entropy(model(images), labels)
            (loss / grad_accum).backward()
            micro_loss += float(loss.detach())
            micro += 1
            images_since_log += images.size(0)
            if micro % grad_accum != 0:
                continue

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            window_loss += micro_loss / grad_accum
            window_steps += 1
            micro_loss = 0.0

            done = max_steps is not None and step >= max_steps
            if step % log_every == 0 or done:
                now = time.perf_counter()
                elapsed = now - log_t0
                row = {
                    "step": step,
                    "loss": window_loss / window_steps,
                    "images_per_sec": images_since_log / elapsed if elapsed else 0.0,
                    "max_mem_gb": torch.cuda.max_memory_allocated() / (1024**3),
                    "elapsed_sec": now - run_t0,
                }
                writer.writerow(row)
                metrics.flush()
                print(
                    f"step {step} loss {row['loss']:.4f} "
                    f"{row['images_per_sec']:.1f} img/s "
                    f"mem {row['max_mem_gb']:.2f} GB"
                )
                window_loss = 0.0
                window_steps = 0
                images_since_log = 0
                log_t0 = now

            if step % ckpt_every == 0:
                _save(output_dir / "last.pt", model, optimizer, step, cfg)

            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break

    _save(output_dir / "last.pt", model, optimizer, step, cfg)
    hardware["elapsed_sec"] = time.perf_counter() - run_t0
    hardware["optimizer_steps"] = step
    (output_dir / "hardware.json").write_text(json.dumps(hardware, indent=2), encoding="utf-8")
    metrics.close()
    print(f"saved {output_dir / 'last.pt'} ({hardware['elapsed_sec']:.1f}s)")
    return output_dir / "last.pt"


def _git_metadata() -> dict[str, str | bool]:
    # Read the checkout containing this code, including when another user account
    # created its worktree. Trust only this explicit path for these read-only calls;
    # do not change the user's global safe.directory configuration.
    repo_dir = Path(__file__).resolve().parents[3]
    git = ["git", "-c", f"safe.directory={repo_dir.as_posix()}", "-C", str(repo_dir)]
    try:
        commit = subprocess.check_output(
            [*git, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            [*git, "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
        branch = subprocess.check_output(
            [*git, "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return {
            "git_commit": commit, "git_dirty": bool(status.strip()),
            "git_branch": branch, "git_worktree": str(repo_dir),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "git_dirty": True}


def _make_datasets(cfg: dict, train_transform, eval_transform):
    if cfg.get("synthetic"):
        train = SyntheticImages(
            int(cfg.get("limit_samples") or 64),
            int(cfg["num_classes"]),
            int(cfg["image_size"]),
        )
        return train, None

    all_samples = load_split_index(Path(cfg["data_root"]) / "train.txt")
    train_samples, val_samples = stratified_split(
        all_samples,
        val_per_class=int(cfg.get("val_per_class", 5)),
        seed=int(cfg.get("split_seed", 0)),
        train_per_class=(
            None if cfg.get("train_per_class") is None else int(cfg["train_per_class"])
        ),
    )
    split_dir = Path(cfg["output_dir"]) / "splits"
    write_split_index(split_dir / "internal_train.txt", train_samples)
    write_split_index(split_dir / "internal_val.txt", val_samples)
    train = CS604ImageDataset(
        cfg["data_root"], "train", transform=train_transform, samples=train_samples
    )
    if not val_samples:
        # val_per_class: 0 trains on every labeled image (final runs); no holdout.
        return train, None
    val = CS604ImageDataset(
        cfg["data_root"], "internal_val", transform=eval_transform, samples=val_samples
    )
    return train, val


def _loader(dataset, cfg: dict, shuffle: bool, seed: int) -> DataLoader:
    workers = int(cfg.get("num_workers", 0))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(cfg.get("eval_batch_size" if not shuffle else "batch_size", cfg["batch_size"])),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=generator,
    )


@torch.inference_mode()
def evaluate(model, loader: DataLoader, device: torch.device, amp_dtype) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    started = time.perf_counter()
    for batch_index, (images, labels, _paths) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        total_loss += float(loss) * images.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum())
        count += images.size(0)
        if batch_index % 50 == 0 or batch_index == len(loader):
            print(f"evaluated {count}/{len(loader.dataset)} images", flush=True)
    elapsed = time.perf_counter() - started
    model.train()
    return {
        "loss": total_loss / max(1, count),
        "accuracy": correct / max(1, count),
        "images_per_sec": count / max(elapsed, 1e-9),
        "elapsed_sec": elapsed,
    }


def neighbour_smoothing(cfg: dict) -> tuple[int, float] | None:
    """``(neighbour_smoothing_window, neighbour_smoothing_share)``, or None when off."""

    window = cfg.get("neighbour_smoothing_window")
    if window is None:
        return None
    window = int(window)
    share = float(cfg.get("neighbour_smoothing_share", 1.0))
    if window < 1:
        raise ValueError(f"neighbour_smoothing_window must be >= 1, got {window}")
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"neighbour_smoothing_share must be in [0, 1], got {share}")
    if not float(cfg.get("label_smoothing", 0.0)) > 0:
        raise ValueError("neighbour_smoothing_window needs label_smoothing > 0 (it redistributes that mass)")
    return window, share


def training_loss(model, logits, labels, label_smoothing: float = 0.0,
                  cosine_margin: float = 0.0,
                  neighbours: tuple[int, float] | None = None) -> torch.Tensor:
    """Cross-entropy, optionally with a CosFace margin on the target class.

    ``cosine_margin`` m subtracts m from the target's cosine before the temperature,
    so the logit becomes s * (cos - m): a class must beat the others by a margin.
    The temperature is detached in the margin term, so the margin does not push it.
    Evaluation always uses the plain logits.

    ``neighbours = (k, share)`` moves ``share`` of the label-smoothing mass from the
    uniform distribution onto the classes whose ids are within k of the target
    (excluding it; ids outside [0, C) are dropped), spread evenly. Class ids appear
    to follow the taxonomy, so these are likely the same genus. share 0 is plain
    label smoothing.
    """

    if cosine_margin:
        scale = model.get_classifier().log_scale.detach().float().exp().clamp(max=100.0)
        target = F.one_hot(labels, logits.shape[1]).to(logits.dtype)
        logits = logits - (cosine_margin * scale) * target
    if neighbours is None:
        return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    window, share = neighbours
    log_probs = F.log_softmax(logits.float(), dim=1)
    nll = -log_probs.gather(1, labels[:, None]).squeeze(1)
    uniform = -log_probs.mean(dim=1)
    offsets = torch.cat([torch.arange(-window, 0), torch.arange(1, window + 1)]).to(labels.device)
    ids = labels[:, None] + offsets[None, :]
    valid = (ids >= 0) & (ids < logits.shape[1])
    near = -log_probs.gather(1, ids.clamp(0, logits.shape[1] - 1))
    near = (near * valid).sum(1) / valid.sum(1)
    eps = label_smoothing
    return ((1 - eps) * nll + eps * (1 - share) * uniform + eps * share * near).mean()


def train_crop_scale(cfg: dict) -> tuple[float, float] | None:
    """RandomResizedCrop area range from ``train_crop_scale``; None keeps timm's default."""

    value = cfg.get("train_crop_scale")
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"train_crop_scale must be [min, max], got {value!r}")
    low, high = float(value[0]), float(value[1])
    if not 0.0 < low <= high <= 1.0:
        raise ValueError(f"train_crop_scale needs 0 < min <= max <= 1, got {value!r}")
    return low, high


TRAIN_TRANSFORM_KEYS = ("hflip", "color_jitter", "auto_augment", "re_prob")


def progressive_resolution(cfg: dict) -> tuple[int, float] | None:
    """``(train_low_image_size, train_low_fraction)``, or None without progressive resizing.

    Training epochs that start before ``train_low_fraction`` of the schedule use
    ``train_low_image_size``; later epochs and every evaluation use ``image_size``.
    """

    low = cfg.get("train_low_image_size")
    if low is None:
        return None
    low = int(low)
    fraction = float(cfg.get("train_low_fraction", 0.75))
    full = int(cfg.get("image_size") or 224)
    if not 0 < low < full:
        raise ValueError(f"train_low_image_size must be below image_size {full}, got {low}")
    if low % 16:
        raise ValueError(f"train_low_image_size must be a multiple of the 16 px patch, got {low}")
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"train_low_fraction must be in (0, 1), got {fraction}")
    if str(cfg.get("mode", "full")) != "full":
        raise ValueError("train_low_image_size is implemented for mode: full only")
    return low, fraction


def build_transforms(cfg: dict, model, train_image_size: int | None = None):
    """``(train_transform, eval_transform)`` for ``model``, as training uses them.

    The train transform is timm's ``is_training=True`` pipeline with the config's
    optional augmentation keys and ``train_crop_scale``. scripts/cache_features.py
    calls this too, so cached augmented views use exactly the training transform.
    ``train_image_size`` changes only the train crop size (progressive resizing).
    """

    from timm.data import create_transform

    data_cfg = model_data_config(model)
    train_cfg = dict(data_cfg)
    if train_image_size is not None:
        train_cfg["input_size"] = (data_cfg["input_size"][0], train_image_size, train_image_size)
    train_kwargs = {"is_training": True}
    for key in TRAIN_TRANSFORM_KEYS:
        if key in cfg:
            train_kwargs[key] = cfg[key]
    crop_scale = train_crop_scale(cfg)
    if crop_scale is not None:
        train_kwargs["scale"] = crop_scale
    return (
        create_transform(**train_cfg, **train_kwargs),
        create_transform(**data_cfg, is_training=False),
    )


def _transform_crop_scale(transform) -> list[float] | None:
    """The RandomResizedCrop scale a built timm transform actually uses."""

    for step in getattr(transform, "transforms", ()):
        scale = getattr(step, "scale", None)
        if scale is not None:
            return [float(scale[0]), float(scale[1])]
    return None


def _check_architecture_compatibility(current: dict, saved: dict) -> None:
    if bool(current.get("cosine_head_fp32", False)) != bool(saved.get("cosine_head_fp32", False)):
        raise ValueError("Checkpoint cosine_head_fp32 does not match the current config")
    keys = (
        "timm_checkpoint",
        "num_classes",
        "mode",
        "head_type",
        "last_blocks",
        "lora_rank",
        "lora_alpha",
        "adapter_dim",
        "adapter_input",
        "adapter_scale_trainable",
    )
    mismatches = [
        key for key in keys
        if current.get(key) != saved.get(key)
        and not (key == "mode" and current.get(key, "full") == saved.get(key, "full"))
    ]
    # Off by default, so an absent key matches false.
    mismatches += [
        key for key in ("adapter_train_bias", "adapter_train_norm", "adapter_ssf", "adapter_lora")
        if bool(current.get(key, False)) != bool(saved.get(key, False))
    ]
    # Absent means [qkv], the only target before the key existed.
    if lora_targets_from_config(current) != lora_targets_from_config(saved):
        mismatches.append("lora_targets")
    # Absent means every block.
    if lora_blocks_from_config(current) != lora_blocks_from_config(saved):
        mismatches.append("lora_blocks")
    if mismatches:
        raise ValueError(
            "Checkpoint architecture does not match the current config: "
            + ", ".join(mismatches)
        )


def check_backbone_budget(cfg: dict, report: dict) -> None:
    """Refuse an AdaptFormer or SSF run at 5% or more of the pretrained backbone (head excluded).

    ``backbone_trainable_params`` counts adapters, LoRA, SSF scales/shifts and any trained
    pretrained biases/norms.
    """

    mode = str(cfg.get("mode"))
    if mode in ("adaptformer", "ssf") and report["backbone_trainable_pct"] >= 5.0:
        raise ValueError(
            f"{mode} trains {report['backbone_trainable_pct']:.2f}% of the backbone; "
            "the course limit is under 5% (head excluded)"
        )


def run_training(cfg: dict) -> Path:
    """Train any configured adaptation mode and evaluate a fixed holdout."""

    output_dir = Path(cfg["output_dir"])
    validate_baseline_setup(cfg, output_dir)
    evaluate_only = bool(cfg.get("evaluate_only", False))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required. Select a GPU runtime or use the CUDA virtualenv.")
    _seed(int(cfg.get("seed", 0)))
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_from_config(cfg).to(device)
    cosine_margin = float(cfg.get("cosine_margin", 0.0))
    neighbours = neighbour_smoothing(cfg)
    if cosine_margin and not isinstance(model.get_classifier(), CosineClassifier):
        raise ValueError("cosine_margin needs head_type: cosine")
    adapter_init = str(cfg.get("adapter_init", "lora"))
    if adapter_init not in ("lora", "pca"):
        raise ValueError(f"adapter_init must be lora or pca, got {adapter_init!r}")
    if adapter_init == "pca" and str(cfg.get("mode")) != "adaptformer":
        raise ValueError("adapter_init: pca needs mode: adaptformer")
    if cfg.get("prototype_file") and not cfg.get("resume"):
        prototype_blob = torch.load(cfg["prototype_file"], map_location="cpu", weights_only=False)
        check_prototype_split(cfg, prototype_blob)
        prototypes = prototype_blob["prototypes"] if isinstance(prototype_blob, dict) else prototype_blob
        apply_class_prototypes(model, prototypes)
        print(f"initialized classifier from {cfg['prototype_file']}")
    if cfg.get("head_init_file") and not cfg.get("resume"):
        head_blob = torch.load(cfg["head_init_file"], map_location="cpu", weights_only=False)
        built_with = head_blob.get("timm_checkpoint", LEGACY_PROTOTYPE_BACKBONE)
        if built_with != cfg.get("timm_checkpoint"):
            raise ValueError(f"head_init_file was trained on {built_with} features, not {cfg.get('timm_checkpoint')}")
        load_head_init(model, head_blob)
        print(f"initialized classifier from trained head {cfg['head_init_file']}")
    if cfg.get("init_checkpoint") and not cfg.get("resume"):
        initial = torch.load(cfg["init_checkpoint"], map_location="cpu", weights_only=False)
        _check_architecture_compatibility(cfg, initial.get("config", {}))
        load_model_state(model, initial)
    report = parameter_report(model)
    check_backbone_budget(cfg, report)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("The selected mode has no trainable parameters")

    crop_scale = train_crop_scale(cfg)
    progressive = progressive_resolution(cfg)
    train_transform, eval_transform = (
        (None, None) if cfg.get("synthetic") else build_transforms(cfg, model)
    )
    low_train_transform = (
        None if progressive is None or cfg.get("synthetic")
        else build_transforms(cfg, model, train_image_size=progressive[0])[0]
    )
    train_dataset, val_dataset = _make_datasets(cfg, train_transform, eval_transform)
    full_image_size = int(cfg.get("image_size") or 224)

    head_lr = None if cfg.get("head_lr") is None else float(cfg["head_lr"])
    layer_decay = None if cfg.get("layer_decay") is None else float(cfg["layer_decay"])
    no_decay_norm_bias = bool(cfg.get("no_decay_norm_bias", False))
    bias_lr = None if cfg.get("bias_lr") is None else float(cfg["bias_lr"])
    optimizer = torch.optim.AdamW(
        optimizer_param_groups(
            model, float(cfg["lr"]), head_lr, layer_decay, no_decay_norm_bias, bias_lr
        ),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.05)),
    )
    grad_accum = max(1, int(cfg.get("grad_accum", 1)))
    epochs = int(cfg.get("epochs", 1))
    batches_per_epoch = math.ceil(len(train_dataset) / int(cfg["batch_size"]))
    steps_per_epoch = math.ceil(batches_per_epoch / grad_accum)
    max_steps = None if cfg.get("max_steps") is None else int(cfg["max_steps"])
    total_steps = max_steps or steps_per_epoch * epochs
    warmup_steps = int(cfg.get("warmup_steps", max(1, round(total_steps * 0.05))))
    lr_schedule = lr_schedule_from_config(cfg)
    # With a time budget, the schedule runs ahead of the step count when the
    # clock does, so the decay still ends inside the budget.
    budget_clock = {"elapsed_sec": 0.0, "limit_sec": None}

    def lr_factor(step: int) -> float:
        position = effective_step(
            step, total_steps, budget_clock["elapsed_sec"], budget_clock["limit_sec"]
        )
        return lr_schedule_factor(position, warmup_steps, total_steps, lr_schedule)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    start_epoch = 0
    resume_batch = 0
    step = 0
    best_accuracy = -1.0
    prior_elapsed = 0.0
    if cfg.get("resume"):
        checkpoint = torch.load(cfg["resume"], map_location="cpu", weights_only=False)
        _check_architecture_compatibility(cfg, checkpoint.get("config", {}))
        # Absent means cosine, the only schedule before the key existed.
        if lr_schedule != lr_schedule_from_config(checkpoint.get("config", {})):
            raise ValueError("Checkpoint lr_schedule does not match the current config")
        load_model_state(model, checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        step = int(checkpoint.get("step", 0))
        start_epoch = int(checkpoint.get("epoch", 0))
        resume_batch = int(checkpoint.get("next_batch", 0))
        best_accuracy = float(checkpoint.get("best_accuracy", -1.0))
        prior_elapsed = float(checkpoint.get("elapsed_sec", 0.0))

    precision = str(cfg.get("precision", "bf16"))
    amp_dtype = torch.bfloat16 if precision == "bf16" else (
        torch.float16 if precision == "fp16" else None
    )
    props = torch.cuda.get_device_properties(0)
    run_info = {
        **_git_metadata(),
        "config_file": cfg.get("config_file", ""),
        "device": torch.cuda.get_device_name(0),
        "vram_gb": round(props.total_memory / 1024**3, 2),
        "precision": precision,
        "mode": cfg.get("mode", "full"),
        "batch_size": int(cfg["batch_size"]),
        "grad_accum": grad_accum,
        "effective_batch_size": int(cfg["batch_size"]) * grad_accum,
        "evaluate_only": evaluate_only,
        "lr": float(cfg["lr"]),
        # null: the head (if trained) shares lr.
        "head_lr": head_lr,
        "head_type": str(cfg.get("head_type", "linear")),
        # Initial cosine temperature, and whether it trains (cosine heads only).
        "cosine_scale": float(cfg.get("cosine_scale", 20.0)),
        "cosine_scale_trainable": bool(cfg.get("cosine_scale_trainable", True)),
        "linear_head_fp32": bool(cfg.get("linear_head_fp32", False)),
        # null: one rate for the whole backbone.
        "layer_decay": layer_decay,
        "no_decay_norm_bias": no_decay_norm_bias,
        "drop_path_rate": float(cfg.get("drop_path_rate", 0.0)),
        "label_smoothing": float(cfg.get("label_smoothing", 0.0)),
        "cosine_margin": cosine_margin,
        **(
            {"neighbour_smoothing_window": neighbours[0], "neighbour_smoothing_share": neighbours[1]}
            if neighbours is not None else {}
        ),
        **(
            {
                "adapter_dim": adapter_dim_from_config(cfg),
                "adapter_scale": float(cfg.get("adapter_scale", 0.1)),
                "adapter_scale_trainable": bool(cfg.get("adapter_scale_trainable", False)),
                "adapter_dropout": float(cfg.get("adapter_dropout", 0.0)),
                "adapter_input": str(cfg.get("adapter_input", "residual")),
                "adapter_init": adapter_init,
                # Pretrained biases / LayerNorm weights trained beside the adapters, at lr.
                "adapter_train_bias": bool(cfg.get("adapter_train_bias", False)),
                "adapter_train_norm": bool(cfg.get("adapter_train_norm", False)),
                # SSF scale/shift after every op beside the adapters, at lr.
                "adapter_ssf": bool(cfg.get("adapter_ssf", False)),
                # LoRA on every block's lora_targets beside the adapters, at lr.
                "adapter_lora": bool(cfg.get("adapter_lora", False)),
                **(
                    {
                        "lora_rank": int(cfg.get("lora_rank", 8)),
                        "lora_alpha": float(cfg.get("lora_alpha", 16.0)),
                        "lora_dropout": float(cfg.get("lora_dropout", 0.0)),
                        "lora_targets": lora_targets_from_config(cfg),
                        # null: every block.
                        "lora_blocks": lora_blocks_from_config(cfg),
                    }
                    if bool(cfg.get("adapter_lora", False)) else {}
                ),
                # null: they train at lr.
                "bias_lr": bias_lr,
            }
            if str(cfg.get("mode")) == "adaptformer" else {}
        ),
        **(
            {"ssf_init": str(cfg.get("ssf_init", "identity"))}
            if str(cfg.get("mode")) == "ssf" or bool(cfg.get("adapter_ssf", False)) else {}
        ),
        "head_init_file": cfg.get("head_init_file"),
        **(
            {"train_low_image_size": progressive[0], "train_low_fraction": progressive[1]}
            if progressive is not None else {}
        ),
        "warmup_steps": warmup_steps,
        # cosine: warmup then decay to zero; constant: warmup then the peak rate to the end.
        "lr_schedule": lr_schedule,
        "total_steps": total_steps,
        "train_images": len(train_dataset),
        # Scale of the built train transform; the configured value with synthetic data.
        "train_crop_scale": (
            _transform_crop_scale(train_transform)
            if train_transform is not None
            else None if crop_scale is None else list(crop_scale)
        ),
        **report,
    }
    budget_sec = None
    if cfg.get("budget_48gb_hours") is not None:
        budget_sec = budget_wall_seconds(float(cfg["budget_48gb_hours"]), run_info["vram_gb"])
        run_info["budget_48gb_hours"] = float(cfg["budget_48gb_hours"])
        run_info["budget_wall_sec"] = budget_sec
        print(
            f"time budget {cfg['budget_48gb_hours']} x 48 GB-hours = "
            f"{budget_sec / 3600:.2f} wall-clock hours on {run_info['vram_gb']} GB"
        )
    if prior_elapsed:
        run_info["resumed_at_elapsed_sec"] = prior_elapsed
    (output_dir / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(
        f"device={run_info['device']} mode={run_info['mode']} "
        f"trainable={report['trainable_params']:,} ({report['trainable_pct']:.3f}%), "
        f"backbone {report['backbone_trainable_params']:,} "
        f"({report['backbone_trainable_pct']:.2f}% of the pretrained backbone, head excluded)"
    )

    metrics_path = output_dir / "metrics.csv"
    append = bool(cfg.get("resume")) and metrics_path.is_file()
    metrics_file = metrics_path.open("a" if append else "w", newline="", encoding="utf-8")
    fields = [
        "phase", "epoch", "step", "loss", "accuracy", "lr",
        "images_per_sec", "max_mem_gb", "elapsed_sec",
    ]
    writer = csv.DictWriter(metrics_file, fieldnames=fields)
    if not append:
        writer.writeheader()

    started = time.perf_counter()

    def run_elapsed() -> float:
        """Seconds this run has used, including sessions before a resume."""
        return prior_elapsed + time.perf_counter() - started

    def finish_run(status: str) -> None:
        elapsed = run_elapsed()
        run_info.update(
            {
                "status": status,
                "elapsed_sec": elapsed,
                "gpu_hours": elapsed / 3600,
                "vram_gb_hours": run_info["vram_gb"] * elapsed / 3600,
                "normalized_48gb_hours": run_info["vram_gb"] * elapsed / (48 * 3600),
                "optimizer_steps": step,
                "best_internal_accuracy": None if best_accuracy < 0 else best_accuracy,
                "max_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
            }
        )
        (output_dir / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
        metrics_file.close()

    if adapter_init == "pca" and not cfg.get("resume"):
        # Counted in the run's time: it is part of training this model.
        calibration_started = time.perf_counter()
        batches = int(cfg.get("adapter_init_batches", 16))
        calibration_loader = _loader(train_dataset, cfg, True, int(cfg.get("seed", 0)) + 10_000)
        calibration_batches = iter(calibration_loader)
        pca = calibrate_adapters_pca(
            model,
            (images for _index, (images, _labels, _paths) in zip(range(batches), calibration_batches)),
            device,
            amp_dtype,
        )
        # Shut its persistent workers down now; left alive beside the training and eval
        # workers they exhausted the laptop's commit limit (2026-10-11).
        del calibration_batches, calibration_loader
        run_info["adapter_init_tokens_per_block"] = pca["tokens_per_block"]
        run_info["adapter_init_variance_explained"] = pca["variance_explained"]
        run_info["adapter_init_sec"] = time.perf_counter() - calibration_started
        print(
            f"adapter_init pca: {batches} batches, variance explained per block "
            + " ".join(f"{value:.2f}" for value in pca["variance_explained"])
        )

    # Seconds the final holdout evaluation takes; the budget keeps room for it.
    final_eval_sec = 0.0
    if baseline_requested(cfg) and not cfg.get("resume"):
        try:
            if val_dataset is None:
                raise ValueError("Initial validation requires an internal holdout")
            provenance = baseline_provenance(
                cfg, output_dir / "splits" / "internal_val.txt", len(val_dataset)
            )
            # A normal prediction-compatible checkpoint, preserved even if evaluation fails.
            save_training_checkpoint(
                output_dir / "baseline.pt", model, optimizer, scheduler, cfg,
                0, 0, 0, best_accuracy, elapsed_sec=run_elapsed(),
            )
            val_loader = _loader(val_dataset, cfg, False, int(cfg.get("seed", 0)))
            result = evaluate(model, val_loader, device, amp_dtype)
            baseline = {
                **provenance, **result, **assess_initial_accuracy(result["accuracy"], cfg),
                "optimizer_steps": 0,
            }
            (output_dir / "baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
            run_info["initial_internal_accuracy"] = result["accuracy"]
            final_eval_sec = result["elapsed_sec"]
            writer.writerow(
                {
                    "phase": "initial_val", "epoch": 0, "step": 0,
                    "loss": result["loss"], "accuracy": result["accuracy"], "lr": "",
                    "images_per_sec": result["images_per_sec"],
                    "max_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
                    "elapsed_sec": run_elapsed(),
                }
            )
            metrics_file.flush()
            print(f"initial validation top1={100 * result['accuracy']:.2f}% (0 optimizer updates)")
            if not baseline["passed"]:
                raise ValueError(
                    "Initial accuracy check failed: "
                    f"actual={result['accuracy']}, expected={baseline['expected_initial_accuracy']}, "
                    f"tolerance={baseline['initial_accuracy_tolerance']}. "
                    "No training started; inspect baseline.json and baseline.pt."
                )
            best_accuracy = result["accuracy"]
            for name in ("baseline.pt", "best.pt"):
                save_training_checkpoint(
                    output_dir / name, model, optimizer, scheduler, cfg,
                    0, 0, 0, best_accuracy, elapsed_sec=run_elapsed(),
                )
        except Exception as error:
            run_info["error"] = str(error)
            finish_run("initial_validation_failed")
            raise
    elif baseline_requested(cfg):
        baseline = json.loads((output_dir / "baseline.json").read_text(encoding="utf-8"))
        run_info["initial_internal_accuracy"] = baseline["initial_internal_accuracy"]
        # Resuming an earlier checkpoint must not lower the existing best-model threshold.
        saved_best = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
        best_accuracy = max(best_accuracy, float(saved_best["best_accuracy"]))

    if evaluate_only:
        finish_run("evaluated")
        return output_dir / "baseline.pt"

    # Keep initialization/evaluation time out of the first training throughput window.
    log_started = time.perf_counter()
    log_images = 0
    log_loss = 0.0
    log_updates = 0
    accumulated_loss = 0.0
    accumulated_batches = 0
    log_every = int(cfg.get("log_every", 20))
    ckpt_every = int(cfg.get("ckpt_every", 100))
    label_smoothing = float(cfg.get("label_smoothing", 0.0))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    stopped = False
    budget_stopped = False
    validating = val_dataset is not None and bool(cfg.get("validate", True))

    def over_budget() -> bool:
        if budget_sec is None:
            return False
        limit = training_limit_seconds(budget_sec, final_eval_sec if validating else 0.0)
        budget_clock["limit_sec"] = limit
        return run_elapsed() >= limit

    for epoch in range(start_epoch, epochs):
        if over_budget():
            # The previous epoch's evaluation used up the budget.
            budget_stopped = True
            break
        if progressive is not None:
            # Decided per epoch, from the same (time-fitted) position the lr schedule uses.
            position = effective_step(
                step, total_steps, budget_clock["elapsed_sec"], budget_clock["limit_sec"]
            ) / total_steps
            low = position < progressive[1]
            epoch_size = progressive[0] if low else full_image_size
            if cfg.get("synthetic"):
                train_dataset.image_size = epoch_size
            else:
                train_dataset.transform = low_train_transform if low else train_transform
            run_info.setdefault("train_image_size_by_epoch", []).append(epoch_size)
            print(f"epoch {epoch + 1}: training at {epoch_size} px (schedule {100 * position:.1f}%)")
        train_loader = _loader(train_dataset, cfg, True, int(cfg.get("seed", 0)) + epoch)
        for batch_index, (images, labels, _paths) in enumerate(train_loader):
            if epoch == start_epoch and batch_index < resume_batch:
                continue
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = training_loss(
                    model, model(images), labels, label_smoothing, cosine_margin, neighbours
                )
            (loss / grad_accum).backward()
            accumulated_loss += float(loss.detach())
            accumulated_batches += 1
            log_images += images.size(0)
            is_update = (batch_index + 1) % grad_accum == 0 or batch_index + 1 == len(train_loader)
            if not is_update:
                continue
            if cfg.get("clip_grad_norm") is not None:
                torch.nn.utils.clip_grad_norm_(trainable, float(cfg["clip_grad_norm"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            budget_clock["elapsed_sec"] = run_elapsed()
            scheduler.step()
            step += 1
            log_loss += accumulated_loss / max(1, accumulated_batches)
            log_updates += 1
            accumulated_loss = 0.0
            accumulated_batches = 0
            done = max_steps is not None and step >= max_steps
            if over_budget():
                budget_stopped = done = True
            if step % log_every == 0 or done:
                now = time.perf_counter()
                row = {
                    "phase": "train", "epoch": epoch + 1, "step": step,
                    "loss": log_loss / max(1, log_updates), "accuracy": "",
                    "lr": optimizer.param_groups[0]["lr"],
                    "images_per_sec": log_images / max(now - log_started, 1e-9),
                    "max_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
                    "elapsed_sec": run_elapsed(),
                }
                writer.writerow(row)
                metrics_file.flush()
                budget_note = ""
                if budget_sec is not None:
                    used = row["elapsed_sec"] / budget_sec
                    budget_note = (
                        f" | budget {100 * used:.1f}% used, "
                        f"~{step / steps_per_epoch / max(used, 1e-9):.1f} epochs fit"
                    )
                print(
                    f"step {step} loss {row['loss']:.4f} "
                    f"{row['images_per_sec']:.1f} img/s{budget_note}"
                )
                log_started, log_images, log_loss, log_updates = now, 0, 0.0, 0
            if step % ckpt_every == 0:
                save_training_checkpoint(
                    output_dir / "last.pt", model, optimizer, scheduler, cfg,
                    step, epoch, batch_index + 1, best_accuracy, elapsed_sec=run_elapsed(),
                )
            if done:
                stopped = True
                break

        if val_dataset is not None and bool(cfg.get("validate", True)):
            val_loader = _loader(val_dataset, cfg, False, int(cfg.get("seed", 0)))
            result = evaluate(model, val_loader, device, amp_dtype)
            final_eval_sec = result["elapsed_sec"]
            writer.writerow(
                {
                    "phase": "val", "epoch": epoch + 1, "step": step,
                    "loss": result["loss"], "accuracy": result["accuracy"], "lr": "",
                    "images_per_sec": result["images_per_sec"],
                    "max_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
                    "elapsed_sec": run_elapsed(),
                }
            )
            metrics_file.flush()
            print(f"validation top1={100 * result['accuracy']:.2f}% loss={result['loss']:.4f}")
            if result["accuracy"] > best_accuracy:
                best_accuracy = result["accuracy"]
                save_training_checkpoint(
                    output_dir / "best.pt", model, optimizer, scheduler, cfg,
                    step, epoch + 1, 0, best_accuracy, elapsed_sec=run_elapsed(),
                )

        next_epoch = epoch if stopped else epoch + 1
        next_batch = batch_index + 1 if stopped else 0
        save_training_checkpoint(
            output_dir / "last.pt", model, optimizer, scheduler, cfg,
            step, next_epoch, next_batch, best_accuracy, elapsed_sec=run_elapsed(),
        )
        resume_batch = 0
        if stopped:
            break

    if budget_sec is not None:
        run_info["stopped_by_budget"] = budget_stopped
        if budget_stopped:
            print("stopped at the time budget; the learning rate had already decayed to zero")
    finish_run("completed")
    return output_dir / ("best.pt" if (output_dir / "best.pt").is_file() else "last.pt")
