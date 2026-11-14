"""Generate and validate a CodaBench prediction file from a checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cs604.engine.predict import predict_split  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument("--crop-pct", type=float, default=None,
                        help="center-crop fraction; default: the checkpoint's (0.9 for this ViT)")
    parser.add_argument("--tta-flip", action="store_true",
                        help="average the softmax with a horizontally flipped pass (2x the time)")
    parser.add_argument("--zoom", action="store_true",
                        help="add an attention-guided zoomed view (cs604.engine.zoom; 2x the passes)")
    parser.add_argument("--zoom-mass", type=float, default=None,
                        help="share of the saliency map the zoom box holds (default 0.7)")
    parser.add_argument("--skip-base-fingerprint", action="store_true",
                        help="load a compact checkpoint whose frozen-backbone fingerprint differs "
                             "(a 336 px run trained on Colab: position-embedding resampling differs "
                             "slightly between timm/torch versions); recorded in the .inference.json")
    args = parser.parse_args()
    result = predict_split(
        args.checkpoint,
        args.data_root,
        args.split,
        args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        make_zip=not args.no_zip,
        crop_pct=args.crop_pct,
        tta_flip=args.tta_flip,
        zoom=args.zoom,
        zoom_mass=args.zoom_mass,
        check_fingerprint=not args.skip_base_fingerprint,
    )
    print(f"validated submission: {result}")


if __name__ == "__main__":
    main()
