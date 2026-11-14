"""Prediction-time "look, then zoom in" view for a timm VisionTransformer.

The plain view is classified first. A saliency map on the patch grid then picks the
smallest square window (at least ``min_side`` patches) holding ``mass`` of the map;
that window is cut from a ``factor`` x resolution copy of the same view, resized to
the model's input size and classified again. Studied in ``scripts/eval_zoom.py``
(internal holdout): ``cam`` at mass 0.7 is the chosen setting.

Saliency maps:

- ``attn``: the last block's CLS-to-patch attention, mean over heads.
- ``cam``: cosine between each final patch token and the predicted class's head
  weight (class-specific), minus its per-image mean, clamped at zero.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_MAP = "cam"
DEFAULT_MASS = 0.7
DEFAULT_MIN_SIDE = 8
DEFAULT_FACTOR = 2


class TwoViews:
    """Returns the eval view and the same field of view at a higher resolution."""

    def __init__(self, small, big):
        self.small = small
        self.big = big

    def __call__(self, image):
        return self.small(image), self.big(image)


def two_view_transform(data_cfg: dict, factor: int = DEFAULT_FACTOR):
    from timm.data import create_transform

    size = int(data_cfg["input_size"][-1])
    big_cfg = {**data_cfg, "input_size": (3, size * factor, size * factor)}
    return TwoViews(create_transform(**data_cfg, is_training=False),
                    create_transform(**big_cfg, is_training=False))


class AttentionTap:
    """Recomputes the last block's CLS attention from its input (fused attention hides it)."""

    def __init__(self, model):
        self.attn = model.blocks[-1].attn
        self.prefix = int(getattr(model, "num_prefix_tokens", 1))
        self.inputs = None
        self.handle = self.attn.register_forward_pre_hook(self._stash)

    def _stash(self, _module, args):
        self.inputs = args[0]

    def cls_attention(self) -> torch.Tensor:
        attn, x = self.attn, self.inputs
        batch, tokens, channels = x.shape
        qkv = attn.qkv(x).reshape(batch, tokens, 3, attn.num_heads, channels // attn.num_heads)
        q, k, _v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = attn.q_norm(q), attn.k_norm(k)
        scores = (q[:, :, :1].float() * attn.scale) @ k.float().transpose(-2, -1)
        return scores.softmax(-1).mean(1)[:, 0, self.prefix:]  # (batch, patches)

    def remove(self) -> None:
        self.handle.remove()


def cam_map(model, tokens: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    prefix = int(getattr(model, "num_prefix_tokens", 1))
    patches = F.normalize(tokens[:, prefix:].float(), dim=-1)
    weights = F.normalize(model.get_classifier().weight[predicted].float(), dim=-1)
    cosine = torch.einsum("bnc,bc->bn", patches, weights)
    return (cosine - cosine.mean(1, keepdim=True)).clamp_min(0)


def select_boxes(saliency: torch.Tensor, grid: int, mass: float, min_side: int) -> torch.Tensor:
    """(batch, grid*grid) map -> (batch, 3) integer boxes ``top, left, side`` in patches.

    The smallest square window (side >= min_side) whose best placement holds ``mass``
    of the map; a bounding box of the top patches is dragged to the image edges by a
    few scattered salient patches.
    """

    weights = (saliency / saliency.sum(1, keepdim=True).clamp_min(1e-12)).view(-1, 1, grid, grid)
    batch = weights.shape[0]
    device = saliency.device
    side = torch.full((batch,), grid, device=device, dtype=torch.long)
    top = torch.zeros(batch, device=device, dtype=torch.long)
    left = torch.zeros(batch, device=device, dtype=torch.long)
    done = torch.zeros(batch, device=device, dtype=torch.bool)
    for size in range(min_side, grid):
        sums = F.avg_pool2d(weights, size, stride=1).flatten(1) * size * size
        best, where = sums.max(1)
        width = grid - size + 1
        take = (best >= mass) & ~done
        side[take] = size
        top[take] = where[take] // width
        left[take] = where[take] % width
        done |= take
    return torch.stack([top, left, side], 1)


def center_boxes(boxes: torch.Tensor, grid: int) -> torch.Tensor:
    """Same sizes, moved to the centre (the no-saliency control)."""

    centred = boxes.clone()
    centred[:, 0] = centred[:, 1] = torch.div(grid - boxes[:, 2], 2, rounding_mode="floor")
    return centred


def crop_zoom(big: torch.Tensor, boxes: torch.Tensor, patch_px: int, size: int) -> torch.Tensor:
    views = []
    for image, (top, left, side) in zip(big, boxes.tolist()):
        region = image[:, top * patch_px:(top + side) * patch_px, left * patch_px:(left + side) * patch_px]
        views.append(F.interpolate(region[None], size=(size, size), mode="bicubic",
                                   antialias=True, align_corners=False)[0])
    return torch.stack(views)


def zoom_view(model, tokens: torch.Tensor, logits: torch.Tensor, big: torch.Tensor,
              saliency: str = DEFAULT_MAP, mass: float = DEFAULT_MASS,
              min_side: int = DEFAULT_MIN_SIDE, tap: AttentionTap | None = None):
    """Zoomed input batch and its boxes, from the plain view's tokens and logits."""

    size = int(model.patch_embed.img_size[0])
    patch = int(model.patch_embed.patch_size[0])
    grid = size // patch
    if saliency == "cam":
        weights = cam_map(model, tokens, logits.argmax(1))
    elif saliency == "attn":
        if tap is None:
            raise ValueError("saliency 'attn' needs an AttentionTap registered before the forward pass")
        weights = tap.cls_attention()
    else:
        raise ValueError(f"unknown saliency map {saliency!r}")
    boxes = select_boxes(weights, grid, mass, min_side)
    factor = big.shape[-1] // size
    return crop_zoom(big, boxes, patch * factor, size), boxes
