# Results

## Setup

- **Data:** 10,000 classes, 45 labelled training images per class (450,000 total).
  The CodaBench validation set (50,000 images) and test set (100,000) have hidden labels.
- **Backbone:** `vit_base_patch16_224.augreg2_in21k_ft_in1k` (ViT-B/16, 86M parameters).
- **Budget:** 2 GPU-hours on a 48 GB card per final run. The cosine schedule is fitted to that budget.
- **Parameter-efficient track:** at most 5% of backbone parameters trainable. The classifier head is
  always trained and is not counted.
- **Internal holdout:** the same 5 images per class (50,000 images, split seed 604), never trained
  on. Screens train on the other 40. Every run first checks that the untrained model scores the
  expected accuracy (42.53% at 336 px), so all runs start from the same verified model.
- **Tie rule:** a difference under 0.4 points on the holdout counts as a tie.

## Full fine-tuning recipe (internal holdout, 3 epochs)

| Step | Resolution | Holdout top-1 | Change |
| --- | --- | ---: | ---: |
| Linear-probe head, layer-wise lr decay 0.75, head at the backbone lr (4e-4) | 224 | 69.83 | |
| + head lr 10x (4e-3) | 224 | 71.37 | +1.53 |
| + head initialized from class prototypes, cosine classifier | 224 | 72.98 | +1.62 |
| + train at 336 px | 336 | 78.39 | +5.41 |

Ablations removing one part each:
- **No layer-wise lr decay:** 63.12, which is −6.7 against the 69.83 control.
- **Constant lr instead of the cosine schedule:** 63.99, which is −9.0 against 72.98.
- **Head lr 30x instead of 10x:** 71.97 with the linear-probe head, and 72.71 with the prototype head. Neither beats 10x by the tie rule.
- **Longer training at 224 px:** 6 epochs reached 73.97, and 12 epochs 74.49.

## AdaptFormer recipe (internal holdout, 336 px, 3 epochs)

| Step | Holdout top-1 | Change |
| --- | ---: | ---: |
| AdaptFormer width 192, residual input, Kaiming init, fixed scale 0.1 | 75.27 | |
| + normalized input, PCA init, learned scale, CosFace margin 0.1 | 76.26 | +0.99 |
| + trainable biases and LayerNorm weights (4.27% of backbone) | 76.76 | +0.50 |
| + LoRA r16 on qkv (4.96% of backbone), two-seed mean | 77.43 | +0.67 |
| *Reference: full fine-tune, same 3 epochs* | *78.39* | |

The first change won at every epoch: +1.21, +1.07 and +0.99.

## Equal-parameter controls (two seeds, 3 epochs)

| Arm, all at about 4.95% of the backbone | Seed 0 | Seed 1 | Mean |
| --- | ---: | ---: | ---: |
| AdaptFormer w192 + biases/norms + LoRA r16 qkv | 77.44 | 77.41 | **77.43** |
| LoRA r28 on qkv/proj/fc1/fc2 + biases/norms, no adapters | 77.23 | 77.28 | 77.26 |
| AdaptFormer w224 + biases/norms | 76.81 | 76.87 | 76.84 |

- **Adapters + LoRA vs wider adapters:** adding LoRA beats spending the same parameters on wider
  adapters by 0.59, in both seeds.
- **Adapters + LoRA vs LoRA alone:** a tie (+0.17).
- **Repeatability:** the seeds agree within 0.08 per arm.

## Prediction views (internal holdout, crop 0.75)

| View | Forward passes | AdaptFormer | Full fine-tune |
| --- | ---: | ---: | ---: |
| plain | 1 | 76.94 | 78.80 |
| plain + flip | 2 | 77.10 | 78.97 |
| plain + zoom, window centred (control) | 2 | 77.67 | 79.43 |
| **plain + zoom, saliency window** | 2 | **77.95** | **79.59** |

The zoom view was chosen on even-numbered images and confirmed on odd-numbered ones.

## Training resolution at equal compute (internal holdout)

| Resolution | Full fine-tune | AdaptFormer |
| --- | ---: | ---: |
| 336 | 63.20 | 59.63 |
| 384 | 64.06 | 61.18 |
| 448 | 64.79 | 62.72 |
| 512 | 64.98 | |

These are short screens (10 images per class), so the absolute numbers are lower than the final runs.

## CodaBench (validation top-1)

| Model | Prediction | Score |
| --- | --- | ---: |
| AdaptFormer, 336 px | plain | 0.779 |
| AdaptFormer, 336 px | crop 0.75 + zoom | 0.797 |
| + biases/norms | crop 0.75 + zoom | 0.802 |
| + 448 px | crop 0.9 + zoom | 0.814 |
| **+ LoRA r16 on qkv (4.94%)** | crop 0.9 + zoom | **0.822** |
| Full fine-tune, 224 px | plain | 0.757 |
| Full fine-tune, 384 px | plain | 0.817 |
| Full fine-tune, 448 px | plain | 0.825 |
| Full fine-tune, 448 px | crop 0.9 + zoom | **0.832** |

**Phase 2 test set (full fine-tune, 448 px):**
- 0.8291 with zoom;
- 0.8384 with zoom + the team's kNN re-ranker.

## Things that did not help

- **SSF in place of biases/norms:** a tie, at 21% lower throughput.
- **Regularization:** adapter dropout, and RandAugment + random erasing, were both ties.
- **Depth-graded adapter widths:** a tie.
- **Neighbour label smoothing** and **progressive resizing:** dropped.
- **Spreading the parameter budget** over all blocks vs concentrating it in a few: a tie for both LoRA and AdaptFormer.
