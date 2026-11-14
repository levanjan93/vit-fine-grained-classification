"""Build the ViT-B variants (timm VisionTransformer) used by the CS604 experiments."""

from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """A frozen linear layer plus a small trainable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.dropout(inputs), self.lora_a), self.lora_b)
        return self.base(inputs) + update * self.scaling


ADAPTER_INPUTS = ("residual", "norm")


class AdaptFormerAdapter(nn.Module):
    """AdaptFormer's parallel bottleneck (Chen et al., NeurIPS 2022): scale * up(relu(down(x))).

    ``up`` starts at zero, so the model is the pretrained one until the first update
    (the authors' "lora" init). ``down`` starts Kaiming-uniform, or from the input's
    principal directions after ``calibrate_adapters_pca``.
    """

    def __init__(self, dim: int, width: int, scale: float = 0.1, dropout: float = 0.0,
                 scale_trainable: bool = False):
        super().__init__()
        if width <= 0:
            raise ValueError("adapter_dim must be positive")
        self.down = nn.Linear(dim, width)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(width, dim)
        if scale_trainable:
            self.scale = nn.Parameter(torch.tensor(float(scale)))
        else:
            self.scale = float(scale)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.up(self.dropout(self.act(self.down(inputs)))) * self.scale


def _attach_adaptformer(block: nn.Module, adapter: AdaptFormerAdapter, adapter_input: str) -> None:
    """Add ``adapter`` beside ``block.mlp``: x + mlp(norm2(x)) + adapter(input).

    ``residual`` feeds the adapter the block's residual stream x, as the authors'
    released code does; ``norm`` feeds it norm2(x), the MLP's own input. Hooks keep
    the pretrained parameter names. The adapter output passes through the block's
    ``drop_path2``/``ls2`` with the MLP's, so use ``drop_path_rate: 0`` to match the paper.
    """

    block.adaptformer = adapter
    if adapter_input == "residual":
        def stash(_module, args):
            object.__setattr__(adapter, "_residual", args[0])

        def add(_module, _args, output):
            residual = adapter.__dict__.pop("_residual")
            return output + adapter(residual)

        block.norm2.register_forward_pre_hook(stash)
    elif adapter_input == "norm":
        def add(_module, args, output):
            return output + adapter(args[0])
    else:
        raise ValueError(f"adapter_input must be one of {ADAPTER_INPUTS}, got {adapter_input!r}")
    block.mlp.register_forward_hook(add)


SSF_INITS = ("identity", "normal")


class SSFScaleShift(nn.Module):
    """SSF's per-channel scale and shift (Lian et al., NeurIPS 2022): y = scale * x + shift.

    ``identity`` starts at scale 1, shift 0, so the model is the pretrained one until the
    first update; ``normal`` is the authors' N(1, 0.02) / N(0, 0.02).
    """

    def __init__(self, dim: int, init: str = "identity"):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        if init == "normal":
            nn.init.normal_(self.scale, mean=1.0, std=0.02)
            nn.init.normal_(self.shift, std=0.02)
        elif init != "identity":
            raise ValueError(f"ssf_init must be one of {SSF_INITS}, got {init!r}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # In the activation's dtype: fp32 parameters times bf16 activations would promote
        # every wrapped output (qkv, fc1, ...) to fp32, halving training throughput.
        scale, shift = self.scale.to(inputs.dtype), self.shift.to(inputs.dtype)
        # Channels are last for Linear/LayerNorm tokens, dim 1 for the patch-embedding conv.
        if inputs.ndim == 4:
            scale, shift = scale[:, None, None], shift[:, None, None]
        return torch.addcmul(shift, inputs, scale)


def _ssf_targets(model: nn.Module) -> list[nn.Module]:
    """The operations the authors' code follows with SSF: patch embedding, every block's
    norm1, qkv, proj, norm2, fc1, fc2, and the final norm."""

    targets = [model.patch_embed.proj]
    for block in model.blocks:
        targets += [block.norm1, block.attn.qkv, block.attn.proj, block.norm2,
                    block.mlp.fc1, block.mlp.fc2]
    targets.append(model.norm)
    return targets


def _ssf_channels(module: nn.Module) -> int:
    if isinstance(module, nn.Linear):
        return module.out_features
    if isinstance(module, nn.Conv2d):
        return module.out_channels
    if isinstance(module, nn.LayerNorm):
        return module.normalized_shape[-1]
    raise TypeError(f"SSF cannot follow {type(module).__name__}")


def _attach_ssf(model: nn.Module, init: str) -> None:
    """Add an ``ssf`` child to every target and a hook applying it to the target's output.

    The child keeps the pretrained parameter names; its parameters are ``<target>.ssf.*``.
    """

    for module in _ssf_targets(model):
        module.ssf = SSFScaleShift(_ssf_channels(module), init)
        handle = module.register_forward_hook(lambda mod, _args, output: mod.ssf(output))
        object.__setattr__(module.ssf, "_hook", handle)


def ssf_modules(model: nn.Module) -> list[SSFScaleShift]:
    return [module for module in model.modules() if isinstance(module, SSFScaleShift)]


@torch.no_grad()
def merge_ssf(model: nn.Module) -> int:
    """Fold every SSF scale/shift into the weights it follows and drop it.

    scale * (W x + b) + shift = (scale W) x + (scale b + shift), so the merged model is
    the plain timm ViT with no inference cost. Returns the number of merged operations.
    """

    merged = 0
    for module in list(model.modules()):
        ssf = module._modules.get("ssf")
        if not isinstance(ssf, SSFScaleShift):
            continue
        scale, shift = ssf.scale.to(module.weight.dtype), ssf.shift.to(module.weight.dtype)
        view = (-1,) + (1,) * (module.weight.ndim - 1)
        module.weight.mul_(scale.view(view) if not isinstance(module, nn.LayerNorm) else scale)
        if module.bias is None:
            module.bias = nn.Parameter(torch.zeros_like(shift), requires_grad=False)
        module.bias.mul_(scale).add_(shift)
        ssf._hook.remove()
        del module.ssf
        merged += 1
    return merged


def adaptformer_adapters(model: nn.Module) -> list[AdaptFormerAdapter]:
    return [block.adaptformer for block in getattr(model, "blocks", ()) if hasattr(block, "adaptformer")]


@torch.no_grad()
def calibrate_adapters_pca(model: nn.Module, batches, device: torch.device, amp_dtype=None,
                           eps: float = 1e-6) -> dict:
    """Start each adapter's ``down`` from the whitened principal directions of its input.

    Runs ``batches`` (an iterable of image tensors) through the model, accumulates
    the mean and covariance of every token the adapter sees, and sets row i of
    ``down`` to v_i / sqrt(lambda_i) with bias -v_i . mean / sqrt(lambda_i): each
    hidden unit starts as a zero-mean, unit-variance projection onto one of the
    top-``width`` directions. ``up`` stays zero, so the model's outputs do not change.
    """

    adapters = adaptformer_adapters(model)
    if not adapters:
        raise ValueError("adapter_init: pca needs mode: adaptformer with adapter_dim > 0")
    dim = adapters[0].down.in_features
    if max(adapter.down.out_features for adapter in adapters) > dim:
        raise ValueError(f"adapter_init: pca needs adapter_dim <= {dim}")
    sums = [torch.zeros(dim, dtype=torch.float64, device=device) for _ in adapters]
    outer = [torch.zeros(dim, dim, dtype=torch.float64, device=device) for _ in adapters]
    counts = [0] * len(adapters)

    def collector(index):
        def hook(_module, args):
            tokens = args[0].detach().reshape(-1, dim).double()
            sums[index] += tokens.sum(dim=0)
            outer[index] += tokens.T @ tokens
            counts[index] += tokens.shape[0]
        return hook

    handles = [adapter.register_forward_pre_hook(collector(i)) for i, adapter in enumerate(adapters)]
    was_training = model.training
    model.eval()
    try:
        for images in batches:
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                model(images.to(device, non_blocking=True))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    if not counts[0]:
        raise ValueError("adapter_init: pca saw no calibration images")

    explained = []
    for adapter, total, product, count in zip(adapters, sums, outer, counts):
        width = adapter.down.out_features
        mean = total / count
        covariance = product / count - torch.outer(mean, mean)
        values, vectors = torch.linalg.eigh(covariance)  # ascending
        values, vectors = values.flip(0)[:width], vectors.flip(1)[:, :width]
        weight = vectors.T / values.clamp_min(eps).sqrt()[:, None]
        adapter.down.weight.copy_(weight.to(adapter.down.weight.dtype))
        adapter.down.bias.copy_((-(weight @ mean)).to(adapter.down.bias.dtype))
        explained.append(float(values.sum() / covariance.trace()))
    return {"tokens_per_block": counts[0], "variance_explained": explained}


class CosineClassifier(nn.Module):
    """Cosine-similarity head suitable for class-prototype initialization."""

    def __init__(self, in_features: int, out_features: int, scale: float = 20.0, force_fp32: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.force_fp32 = force_fp32
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.log_scale = nn.Parameter(torch.tensor(math.log(scale)))
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.force_fp32:
            # Match build_prototypes.py: keep backbone AMP, but normalize and
            # compare features outside autocast so nearby class scores don't tie.
            with torch.autocast(device_type=inputs.device.type, enabled=False):
                scale = self.log_scale.float().exp().clamp(max=100.0)
                return scale * F.linear(
                    F.normalize(inputs.float(), dim=-1),
                    F.normalize(self.weight.float(), dim=-1),
                )
        scale = self.log_scale.exp().clamp(max=100.0)
        return scale * F.linear(F.normalize(inputs, dim=-1), F.normalize(self.weight, dim=-1))


class Fp32Linear(nn.Linear):
    """A linear classifier whose logits stay float32 under bf16 autocast.

    timm's plain ``nn.Linear`` head would produce bf16 logits under autocast,
    which ties nearby class scores (as ``cosine_head_fp32`` fixes for the cosine head).
    """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return F.linear(inputs.float(), self.weight.float(), self.bias.float())


def _classifier(model: nn.Module) -> nn.Module:
    classifier = model.get_classifier()
    if not isinstance(classifier, nn.Module):
        raise TypeError("The timm model does not expose a classifier module")
    return classifier


def adapter_widths(adapter_dim, depth: int) -> list[int]:
    """Per-block AdaptFormer widths: one int for every block, or a list with one per block."""

    if isinstance(adapter_dim, (list, tuple)):
        widths = [int(width) for width in adapter_dim]
        if len(widths) != depth:
            raise ValueError(f"adapter_dim list needs {depth} widths (one per block), got {len(widths)}")
        return widths
    return [int(adapter_dim)] * depth


def adapter_dim_from_config(cfg: dict):
    """``adapter_dim`` as an int, or a list of ints (one per block); default 64."""

    value = cfg.get("adapter_dim", 64)
    return [int(width) for width in value] if isinstance(value, (list, tuple)) else int(value)


# ``lora_targets`` names: the block's child module (attn or mlp) holding each Linear.
LORA_TARGETS = {"qkv": "attn", "proj": "attn", "fc1": "mlp", "fc2": "mlp"}


def lora_targets_from_config(cfg: dict) -> list[str]:
    """``lora_targets`` as a list of names from ``LORA_TARGETS``; default ``[qkv]``."""

    value = cfg.get("lora_targets", ["qkv"])
    targets = [str(value)] if isinstance(value, str) else [str(target) for target in value]
    unknown = sorted(set(targets) - set(LORA_TARGETS))
    if unknown or not targets or len(set(targets)) != len(targets):
        raise ValueError(f"lora_targets must be distinct names from {sorted(LORA_TARGETS)}, got {value!r}")
    return targets


def lora_blocks_from_config(cfg: dict) -> list[int] | None:
    """``lora_blocks``: 0-based indices of the blocks that get LoRA; absent/null = every block."""

    value = cfg.get("lora_blocks")
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"lora_blocks must be a list of block indices, got {value!r}")
    blocks = [int(index) for index in value]
    if not blocks or len(set(blocks)) != len(blocks) or min(blocks) < 0:
        raise ValueError(f"lora_blocks must be distinct non-negative block indices, got {value!r}")
    return sorted(blocks)


def _replace_with_lora(
    model: nn.Module, targets, rank: int, alpha: float, dropout: float, blocks=None
) -> int:
    """Wrap the ``targets`` Linears (qkv, proj, fc1, fc2) of ``blocks`` (default all) in LoRA."""

    depth = len(model.blocks)
    chosen = range(depth) if blocks is None else blocks
    if max(chosen) >= depth:
        raise ValueError(f"lora_blocks indices must be below {depth}, got {list(chosen)}")
    replaced = 0
    for index in chosen:
        block = model.blocks[index]
        for target in targets:
            parent = getattr(block, LORA_TARGETS[target])
            linear = getattr(parent, target)
            if not isinstance(linear, nn.Linear):
                raise TypeError(f"Expected Linear {target}, found {type(linear).__name__}")
            setattr(parent, target, LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout))
            replaced += 1
    return replaced


def build_vit(
    checkpoint: str,
    num_classes: int,
    grad_checkpointing: bool = False,
    mode: str = "full",
    head_type: str = "linear",
    train_head: bool = True,
    last_blocks: int = 1,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    lora_targets=("qkv",),
    lora_blocks: list[int] | None = None,
    cosine_head_fp32: bool = False,
    image_size: int | None = None,
    cosine_scale: float = 20.0,
    cosine_scale_trainable: bool = True,
    linear_head_fp32: bool = False,
    drop_path_rate: float = 0.0,
    adapter_dim: int | list[int] = 64,
    adapter_scale: float = 0.1,
    adapter_scale_trainable: bool = False,
    adapter_dropout: float = 0.0,
    adapter_input: str = "residual",
    adapter_train_bias: bool = False,
    adapter_train_norm: bool = False,
    ssf_init: str = "identity",
    adapter_ssf: bool = False,
    adapter_lora: bool = False,
    dynamic_img_size: bool = False,
) -> nn.Module:
    # image_size resizes the position embeddings, e.g. DINOv2 (default 518) to 224.
    size_kwargs = {} if image_size is None else {"img_size": int(image_size)}
    # Accept smaller inputs too (position embeddings resampled per forward pass, an
    # identity at image_size); progressive-resolution training needs it.
    if dynamic_img_size:
        size_kwargs["dynamic_img_size"] = True
    # Stochastic depth: timm ramps the drop rate linearly up to drop_path_rate in the last block.
    if drop_path_rate:
        size_kwargs["drop_path_rate"] = float(drop_path_rate)
    model = timm.create_model(checkpoint, pretrained=True, num_classes=num_classes, **size_kwargs)
    if head_type == "cosine":
        old_head = _classifier(model)
        if not isinstance(old_head, nn.Linear):
            raise TypeError("Cosine head replacement expects a linear timm classifier")
        model.head = CosineClassifier(
            old_head.in_features, num_classes, scale=cosine_scale, force_fp32=cosine_head_fp32
        )
    elif head_type == "linear":
        if linear_head_fp32:
            old_head = _classifier(model)
            model.head = Fp32Linear(old_head.in_features, num_classes)
            model.head.load_state_dict(old_head.state_dict())
    else:
        raise ValueError(f"Unknown head_type: {head_type}")

    if grad_checkpointing:
        model.set_grad_checkpointing(True)

    if mode == "full":
        model.requires_grad_(True)
    elif mode == "head":
        model.requires_grad_(False)
        _classifier(model).requires_grad_(True)
    elif mode == "last_blocks":
        if not 1 <= last_blocks <= len(model.blocks):
            raise ValueError(f"last_blocks must be in [1, {len(model.blocks)}]")
        model.requires_grad_(False)
        for block in model.blocks[-last_blocks:]:
            block.requires_grad_(True)
        model.norm.requires_grad_(True)
        _classifier(model).requires_grad_(train_head)
    elif mode == "lora":
        model.requires_grad_(False)
        _replace_with_lora(model, lora_targets, lora_rank, lora_alpha, lora_dropout, lora_blocks)
        _classifier(model).requires_grad_(train_head)
    elif mode == "adaptformer":
        model.requires_grad_(False)
        widths = adapter_widths(adapter_dim, len(model.blocks))
        if min(widths) < 0:
            raise ValueError("adapter_dim must be positive, or 0 for no adapter in that block")
        for block, width in zip(model.blocks, widths):
            # Width 0: no adapter in this block (all zeros: LoRA and/or biases/norms alone).
            if width == 0:
                continue
            adapter = AdaptFormerAdapter(
                block.mlp.fc1.in_features, width, adapter_scale, adapter_dropout,
                adapter_scale_trainable,
            )
            _attach_adaptformer(block, adapter, adapter_input)
        if adapter_lora:
            # LoRA on the lora_targets of lora_blocks (default every block) beside the adapters
            # (lora_rank/lora_alpha/lora_dropout). Before the bias unfreeze, so the wrapped base
            # biases still train under adapter_train_bias.
            _replace_with_lora(
                model, lora_targets, lora_rank, lora_alpha, lora_dropout, lora_blocks
            )
        _unfreeze_bias_and_norm(model, adapter_train_bias, adapter_train_norm)
        if adapter_ssf:
            # SSF after every op beside the adapters; with adapter_input: norm the adapter reads
            # the scaled/shifted norm2(x). Identity init keeps the PCA calibration unchanged.
            _attach_ssf(model, ssf_init)
        _classifier(model).requires_grad_(train_head)
    elif mode == "ssf":
        model.requires_grad_(False)
        _attach_ssf(model, ssf_init)
        _classifier(model).requires_grad_(train_head)
    else:
        raise ValueError(f"Unknown adaptation mode: {mode}")
    if (adapter_train_bias or adapter_train_norm) and mode != "adaptformer":
        raise ValueError("adapter_train_bias/adapter_train_norm need mode: adaptformer")
    if adapter_ssf and mode != "adaptformer":
        raise ValueError("adapter_ssf needs mode: adaptformer (use mode: ssf for SSF alone)")
    if adapter_lora and mode != "adaptformer":
        raise ValueError("adapter_lora needs mode: adaptformer (use mode: lora for LoRA alone)")
    if lora_blocks is not None and not (mode == "lora" or adapter_lora):
        raise ValueError("lora_blocks needs mode: lora or adapter_lora: true")
    _freeze_scale_unless_trainable(_classifier(model), cosine_scale_trainable)
    return model


def _unfreeze_bias_and_norm(model: nn.Module, bias: bool, norm: bool) -> None:
    """Train pretrained backbone biases (BitFit) and/or LayerNorm weights beside the adapters.

    ``bias``: every Linear, Conv2d (patch embedding) and LayerNorm bias; ``norm``: every
    LayerNorm weight. The adapters and the classifier are left as they are. Both start at
    their pretrained values, so the step-zero gate is unchanged.
    """

    skip = {id(p) for p in _classifier(model).parameters()}
    skip |= {id(p) for adapter in adaptformer_adapters(model) for p in adapter.parameters()}
    for module in model.modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d, nn.LayerNorm)):
            continue
        if bias and module.bias is not None and id(module.bias) not in skip:
            module.bias.requires_grad_(True)
        if norm and isinstance(module, nn.LayerNorm) and module.weight is not None:
            module.weight.requires_grad_(True)


def _freeze_scale_unless_trainable(classifier: nn.Module, trainable: bool) -> None:
    """``cosine_scale_trainable: false`` keeps the cosine temperature at its initial value."""

    if trainable:
        return
    if not isinstance(classifier, CosineClassifier):
        raise ValueError("cosine_scale_trainable: false needs head_type: cosine")
    classifier.log_scale.requires_grad_(False)


class FeatureHead(nn.Module):
    """The classifier alone, for training on cached pre-logits features.

    Built by the same code as the full model's head (``head_type``, ``cosine_scale``,
    ``cosine_head_fp32``, ``cosine_scale_trainable``, ``linear_head_fp32``), and it exposes
    ``get_classifier()``, so ``apply_class_prototypes`` and
    ``optimizer_param_groups`` treat it exactly like the full model in ``mode: head``.
    The state dict keys (``head.*``) match the full model's classifier keys.
    """

    def __init__(self, head: nn.Module):
        super().__init__()
        self.head = head

    def get_classifier(self) -> nn.Module:
        return self.head

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)


def build_feature_head(cfg: dict, in_features: int) -> FeatureHead:
    """A trainable head matching ``build_from_config(cfg)``'s classifier in ``mode: head``."""

    head_type = str(cfg.get("head_type", "linear"))
    num_classes = int(cfg["num_classes"])
    if head_type == "cosine":
        head: nn.Module = CosineClassifier(
            in_features,
            num_classes,
            scale=float(cfg.get("cosine_scale", 20.0)),
            force_fp32=bool(cfg.get("cosine_head_fp32", False)),
        )
    elif head_type == "linear":
        linear = Fp32Linear if bool(cfg.get("linear_head_fp32", False)) else nn.Linear
        head = linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown head_type: {head_type}")
    _freeze_scale_unless_trainable(head, bool(cfg.get("cosine_scale_trainable", True)))
    return FeatureHead(head)


def apply_class_prototypes(model: nn.Module, prototypes: torch.Tensor) -> None:
    """Copy one normalized feature average into each class weight."""

    classifier = _classifier(model)
    weight = getattr(classifier, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError("The classifier has no weight parameter")
    if tuple(weight.shape) != tuple(prototypes.shape):
        raise ValueError(
            f"Prototype shape {tuple(prototypes.shape)} does not match "
            f"classifier weight {tuple(weight.shape)}"
        )
    normalized = F.normalize(prototypes.float(), dim=1).to(
        device=weight.device, dtype=weight.dtype
    )
    with torch.no_grad():
        weight.copy_(normalized)
        bias = getattr(classifier, "bias", None)
        if isinstance(bias, torch.Tensor):
            bias.zero_()


def build_from_config(cfg: dict) -> nn.Module:
    """Build a model from the shared YAML configuration dictionary."""

    return build_vit(
        checkpoint=cfg["timm_checkpoint"],
        num_classes=int(cfg["num_classes"]),
        grad_checkpointing=bool(cfg.get("grad_checkpointing", False)),
        mode=str(cfg.get("mode", "full")),
        head_type=str(cfg.get("head_type", "linear")),
        train_head=bool(cfg.get("train_head", True)),
        last_blocks=int(cfg.get("last_blocks", 1)),
        lora_rank=int(cfg.get("lora_rank", 8)),
        lora_alpha=float(cfg.get("lora_alpha", 16.0)),
        lora_dropout=float(cfg.get("lora_dropout", 0.0)),
        lora_targets=lora_targets_from_config(cfg),
        lora_blocks=lora_blocks_from_config(cfg),
        cosine_head_fp32=bool(cfg.get("cosine_head_fp32", False)),
        image_size=None if cfg.get("image_size") is None else int(cfg["image_size"]),
        cosine_scale=float(cfg.get("cosine_scale", 20.0)),
        cosine_scale_trainable=bool(cfg.get("cosine_scale_trainable", True)),
        linear_head_fp32=bool(cfg.get("linear_head_fp32", False)),
        drop_path_rate=float(cfg.get("drop_path_rate", 0.0)),
        adapter_dim=adapter_dim_from_config(cfg),
        adapter_scale=float(cfg.get("adapter_scale", 0.1)),
        adapter_scale_trainable=bool(cfg.get("adapter_scale_trainable", False)),
        adapter_dropout=float(cfg.get("adapter_dropout", 0.0)),
        adapter_input=str(cfg.get("adapter_input", "residual")),
        adapter_train_bias=bool(cfg.get("adapter_train_bias", False)),
        adapter_train_norm=bool(cfg.get("adapter_train_norm", False)),
        ssf_init=str(cfg.get("ssf_init", "identity")),
        adapter_ssf=bool(cfg.get("adapter_ssf", False)),
        adapter_lora=bool(cfg.get("adapter_lora", False)),
        dynamic_img_size=cfg.get("train_low_image_size") is not None,
    )


def load_head_init(model: nn.Module, blob: dict) -> None:
    """Load a head trained on cached features (``sweep_cached_head.py --save-heads``).

    The file holds ``head``, the ``FeatureHead`` state dict (keys ``head.*``, the
    full model's classifier keys), and ``timm_checkpoint``, the backbone whose
    features it was trained on.
    """

    state = {
        name.removeprefix("head."): tensor for name, tensor in blob["head"].items()
    }
    _classifier(model).load_state_dict(state)


def model_data_config(model: nn.Module) -> dict:
    """timm's preprocessing for ``model``, at the input size it was built for.

    timm keeps the checkpoint's default input size in the data config even when
    ``img_size`` changed it (DINOv2: 518), so take the size from the patch embedding.
    """

    from timm.data import resolve_model_data_config

    config = resolve_model_data_config(model)
    img_size = getattr(getattr(model, "patch_embed", None), "img_size", None)
    if img_size is not None:
        config["input_size"] = (config["input_size"][0], *img_size)
    return config


def use_compact_checkpoints(cfg: dict) -> bool:
    """Compact checkpoints drop frozen backbone weights that timm can rebuild.

    Safe only when the frozen backbone is exactly the pretrained one, so it is the
    default for LoRA/head runs without ``init_checkpoint``.
    """

    frozen_backbone = str(cfg.get("mode", "full")) in ("lora", "head", "adaptformer", "ssf")
    default = frozen_backbone and not cfg.get("init_checkpoint")
    compact = bool(cfg.get("compact_checkpoints", default))
    if compact and not default:
        raise ValueError(
            "compact_checkpoints needs mode lora/head and no init_checkpoint; "
            "set compact_checkpoints: false"
        )
    return compact


def _compact_keys(model: nn.Module) -> set[str]:
    classifier_ids = {id(p) for p in _classifier(model).parameters()}
    return {
        name for name, p in model.named_parameters()
        if p.requires_grad or id(p) in classifier_ids
    } | {name for name, _ in model.named_buffers()}


def base_fingerprint(model: nn.Module) -> str:
    """Hash a sample of frozen backbone tensors to catch a different pretrained base."""

    import hashlib

    keep = _compact_keys(model)
    state = model.state_dict()
    frozen = sorted(name for name in state if name not in keep)
    digest = hashlib.sha256()
    for name in sorted({frozen[0], frozen[len(frozen) // 2], frozen[-1]} if frozen else ()):
        digest.update(name.encode())
        digest.update(state[name].detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def checkpoint_model_state(model: nn.Module, compact: bool) -> dict:
    """The ``model`` entry of a checkpoint, plus its format fields."""

    state = model.state_dict()
    if not compact:
        return {"model": state, "model_format": "full"}
    keep = _compact_keys(model)
    return {
        "model": {name: tensor for name, tensor in state.items() if name in keep},
        "model_format": "compact",
        "base_fingerprint": base_fingerprint(model),
    }


def load_model_state(model: nn.Module, checkpoint: dict, check_fingerprint: bool = True) -> None:
    """Load a full or compact checkpoint into a model built from its config.

    ``check_fingerprint=False`` skips the frozen-backbone check. Only for a known,
    harmless difference: at a non-native ``image_size`` the position embedding is
    resampled when the model is built, and other timm/torch versions (e.g. Colab's)
    resample it with tiny numeric differences, which changes the fingerprint.
    """

    state = checkpoint["model"]
    if checkpoint.get("model_format", "full") == "full":
        model.load_state_dict(state)
        return
    expected = _compact_keys(model)
    if set(state) != expected:
        raise ValueError(
            "Compact checkpoint keys do not match this model's trainable/classifier "
            f"weights ({len(set(state) ^ expected)} differ)"
        )
    if check_fingerprint and checkpoint.get("base_fingerprint") != base_fingerprint(model):
        raise ValueError(
            "Frozen backbone differs from the one this compact checkpoint was trained on "
            "(timm weights changed?)"
        )
    model.load_state_dict(state, strict=False)


NO_DECAY_NAMES = ("cls_token", "pos_embed", "reg_token")


def is_added_parameter(name: str) -> bool:
    """A parameter of a module added for PEFT (LoRA, AdaptFormer, SSF), not a pretrained one."""

    return "lora_" in name or ".adaptformer." in name or ".ssf." in name


def vit_layer_id(name: str, num_blocks: int) -> int:
    """Depth of a timm ViT parameter: 0 embeddings, i + 1 for blocks.i, top norm/head above."""

    if name.startswith(("patch_embed.", *NO_DECAY_NAMES)):
        return 0
    if name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
    return num_blocks + 1


def optimizer_param_groups(
    model: nn.Module,
    lr: float,
    head_lr: float | None = None,
    layer_decay: float | None = None,
    no_decay_norm_bias: bool = False,
    bias_lr: float | None = None,
) -> list[dict]:
    """AdamW parameter groups.

    Without the optional settings this is a single group of every trainable
    parameter at ``lr``. ``head_lr`` gives the trainable classifier (cosine weight
    and temperature) its own rate. ``layer_decay`` (layer-wise LR decay) scales a
    backbone parameter's rate by ``layer_decay ** (top - depth)``: the final norm
    keeps ``lr``, block i gets ``lr * layer_decay ** (blocks - i)``, embeddings the
    smallest. ``no_decay_norm_bias`` sets weight decay 0 on 1-D parameters (norms,
    biases, the cosine temperature) and the cls/position tokens. ``bias_lr`` gives
    the trainable pretrained backbone parameters (``adapter_train_bias`` /
    ``adapter_train_norm``) their own rate, apart from the adapters and the head.

    The backbone group with the highest rate comes first, so
    ``param_groups[0]["lr"]`` is the top backbone rate in the metrics log.
    """

    if head_lr is not None and head_lr <= 0:
        raise ValueError(f"head_lr must be positive, got {head_lr}")
    if layer_decay is not None and not 0 < layer_decay <= 1:
        raise ValueError(f"layer_decay must be in (0, 1], got {layer_decay}")
    if bias_lr is not None and bias_lr <= 0:
        raise ValueError(f"bias_lr must be positive, got {bias_lr}")
    if bias_lr is not None and layer_decay is not None:
        raise ValueError("bias_lr and layer_decay cannot be combined")
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if head_lr is None and layer_decay is None and not no_decay_norm_bias and bias_lr is None:
        return [{"params": [p for _name, p in trainable], "lr": lr}]
    classifier_ids = {id(p) for p in _classifier(model).parameters()}
    if head_lr is not None and not any(id(p) in classifier_ids for _name, p in trainable):
        raise ValueError("head_lr is set but the classifier is frozen; set train_head: true")
    num_blocks = len(getattr(model, "blocks", ()))
    if layer_decay is not None and not num_blocks:
        raise ValueError("layer_decay needs a ViT with blocks")

    def is_pretrained(name: str) -> bool:
        return not is_added_parameter(name)
    if bias_lr is not None and not any(
        id(p) not in classifier_ids and is_pretrained(name) for name, p in trainable
    ):
        raise ValueError("bias_lr is set but no pretrained backbone parameter trains; "
                         "set adapter_train_bias or adapter_train_norm")

    groups: dict[tuple, dict] = {}
    for name, param in trainable:
        is_head = id(param) in classifier_ids
        if is_head:
            depth, rate = num_blocks + 1, lr if head_lr is None else head_lr
        elif bias_lr is not None and is_pretrained(name):
            depth, rate = -1, bias_lr
        else:
            if layer_decay is None:
                depth, rate = 0, lr
            else:
                depth = vit_layer_id(name, num_blocks)
                rate = lr * layer_decay ** (num_blocks + 1 - depth)
        no_decay = no_decay_norm_bias and (param.ndim <= 1 or name in NO_DECAY_NAMES)
        key = (is_head, depth, no_decay)
        if key not in groups:
            groups[key] = {
                "params": [], "lr": rate,
                "name": f"{'head' if is_head else 'pretrained' if depth < 0 else f'layer{depth}'}"
                        f"{'_no_decay' if no_decay else ''}",
            }
            if no_decay:
                groups[key]["weight_decay"] = 0.0
        groups[key]["params"].append(param)
    # Backbone first, highest rate first; head groups last.
    order = sorted(groups, key=lambda key: (key[0], -groups[key]["lr"], key[2]))
    return [groups[key] for key in order]


def parameter_report(model: nn.Module) -> dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    classifier_ids = {id(p) for p in _classifier(model).parameters()}
    classifier_total = sum(p.numel() for p in model.parameters() if id(p) in classifier_ids)
    classifier_trainable = sum(
        p.numel() for p in model.parameters() if id(p) in classifier_ids and p.requires_grad
    )
    # Modules added for PEFT (LoRA, adapters, SSF) are not part of the pretrained backbone,
    # which is the denominator of the course's "under 5%" (head excluded).
    added = sum(p.numel() for name, p in model.named_parameters() if is_added_parameter(name))
    pretrained_backbone = total - classifier_total - added
    backbone_trainable = trainable - classifier_trainable
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": 100.0 * trainable / total if total else 0.0,
        "classifier_params": classifier_total,
        "classifier_trainable_params": classifier_trainable,
        "backbone_trainable_params": backbone_trainable,
        "pretrained_backbone_params": pretrained_backbone,
        "backbone_trainable_pct": (
            100.0 * backbone_trainable / pretrained_backbone if pretrained_backbone else 0.0
        ),
    }
