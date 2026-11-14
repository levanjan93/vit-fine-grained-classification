"""Check that the downloaded CS604 splits match the project brief."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cs604.data.dataset import load_split_index  # noqa: E402

DEFAULT_DIR = Path("data")
EXPECTED = {
    "train": {"samples": 450_000, "labeled": True, "per_class": 45},
    "val": {"samples": 50_000, "labeled": False, "per_class": None},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()

    ok = True
    for split, spec in EXPECTED.items():
        list_path = args.data_root / f"{split}.txt"
        if not list_path.is_file():
            print(f"FAIL {split}: missing {list_path}")
            ok = False
            continue
        samples = load_split_index(list_path)
        labeled = all(label is not None for _, label in samples)
        print(f"{split}: {len(samples)} lines, labeled={labeled}")
        if len(samples) != spec["samples"]:
            print(f"  FAIL expected {spec['samples']} samples")
            ok = False
        if labeled != spec["labeled"]:
            print(f"  FAIL expected labeled={spec['labeled']}")
            ok = False
        missing = 0
        checked = 0
        for rel_path, _ in samples[:100]:
            checked += 1
            if not (args.data_root / rel_path).is_file():
                missing += 1
        if missing:
            print(f"  FAIL {missing}/{checked} sampled files are missing on disk")
            ok = False
        if split == "train" and labeled:
            counts = Counter(label for _, label in samples)
            if len(counts) != 10_000:
                print(f"  FAIL expected 10000 classes, found {len(counts)}")
                ok = False
            uneven = sum(1 for n in counts.values() if n != spec["per_class"])
            if uneven:
                print(f"  FAIL {uneven} classes do not have {spec['per_class']} images")
                ok = False

    if not ok:
        sys.exit(1)
    print("dataset checks passed")


if __name__ == "__main__":
    main()
