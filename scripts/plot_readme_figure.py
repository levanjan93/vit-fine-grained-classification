"""Draws docs/adaptformer_progress.png for the README from the logged results in docs/RESULTS.md.

Dots, not bars: the axes start near the data, and a bar would exaggerate the gaps.
"""

from pathlib import Path

import matplotlib.pyplot as plt

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, BLUE_DARK = "#86b6ef", "#1c5cab"

# Internal holdout (50,000 images, 5 per class), 336 px, 3 epochs.
HOLDOUT = [
    ("AdaptFormer baseline", 75.27),
    ("+ normalized input, PCA init,\n   learned scale, CosFace margin", 76.26),
    ("+ trainable biases and LayerNorms", 76.76),
    ("+ LoRA r16 on qkv (4.96% of backbone)", 77.43),
]
FULL_FT_3EP = 78.39

# CodaBench validation top-1 (50,000 hidden-label images), full 2-hour training budget.
CODABENCH = [
    ("336 px, plain prediction", 0.779),
    ("+ crop 0.75 and flip", 0.790),
    ("+ zoom view instead of flip", 0.797),
    ("+ trainable biases and LayerNorms", 0.802),
    ("train at 448 px", 0.809),
    ("+ crop 0.9 for the plain view", 0.814),
    ("+ LoRA r16 on qkv", 0.822),
]


def dot_panel(ax, rows, ref=None, ref_label=None, fmt="{:.2f}", title=""):
    labels = [r[0] for r in rows][::-1]
    values = [r[1] for r in rows][::-1]
    y = range(len(rows))
    # Rows are drawn bottom-up, so the last (final) result sits at y = 0 and is highlighted.
    colors = [BLUE_DARK] + [BLUE] * (len(rows) - 1)
    ax.scatter(values, y, s=90, c=colors, edgecolors=SURFACE, linewidths=2, zorder=3)
    for yi, v in zip(y, values):
        ax.text(v, yi + 0.32, fmt.format(v), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_yticks(list(y), labels, fontsize=9, color=INK)
    if ref is not None:
        ax.axvline(ref, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
        ax.text(ref, len(rows) - 0.45, ref_label, ha="right", va="bottom", fontsize=8.5, color=INK2)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=12)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=INK2, labelsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(rows) - 0.1)


def main() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURFACE,
                               gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.75})
    for ax in (a, b):
        ax.set_facecolor(SURFACE)
    dot_panel(a, HOLDOUT, FULL_FT_3EP, "full fine-tune, same 3 epochs: 78.39 ",
              title="AdaptFormer recipe, internal holdout top-1 (%)")
    a.set_xlim(74.8, 78.8)
    dot_panel(b, CODABENCH, fmt="{:.3f}", title="PEFT track, CodaBench validation top-1")
    b.set_xlim(0.772, 0.828)
    out = Path(__file__).resolve().parents[1] / "docs" / "adaptformer_progress.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
