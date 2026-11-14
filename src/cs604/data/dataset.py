"""Image-list dataset for the CS604 course splits.

train.txt lines: ``train/class_XXXX/<id>.jpg <class_id>``
val.txt / test.txt lines: ``val/image_XXXXXX.jpg`` (path only; labels withheld)
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image


def load_split_index(list_path: Path) -> list[tuple[str, int | None]]:
    samples: list[tuple[str, int | None]] = []
    with list_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            rel_path = parts[0]
            label = int(parts[1]) if len(parts) > 1 else None
            samples.append((rel_path, label))
    return samples


def stratified_split(
    samples: Iterable[tuple[str, int | None]],
    val_per_class: int = 5,
    seed: int = 0,
    train_per_class: int | None = None,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Create a repeatable train/validation split with every class represented.

    ``train_per_class`` is useful for cheap screening runs. It is applied after
    reserving validation examples, so the validation set stays identical.
    """

    by_class: dict[int, list[str]] = defaultdict(list)
    for path, label in samples:
        if label is None:
            raise ValueError("A stratified split requires labels")
        by_class[label].append(path)

    train: list[tuple[str, int]] = []
    val: list[tuple[str, int]] = []
    for label in sorted(by_class):
        paths = sorted(by_class[label])
        rng = random.Random(seed + label)
        rng.shuffle(paths)
        if len(paths) <= val_per_class:
            raise ValueError(
                f"Class {label} has {len(paths)} samples; need more than "
                f"val_per_class={val_per_class}"
            )
        val.extend((path, label) for path in paths[:val_per_class])
        remaining = paths[val_per_class:]
        if train_per_class is not None:
            remaining = remaining[:train_per_class]
        train.extend((path, label) for path in remaining)
    return train, val


def write_split_index(path: Path, samples: Iterable[tuple[str, int | None]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for rel_path, label in samples:
            suffix = "" if label is None else f" {label}"
            handle.write(f"{rel_path}{suffix}\n")


class CS604ImageDataset:
    """Map-style dataset. Returns ``(image, label, relative_path)``.

    ``label`` is ``None`` for the unlabeled validation and test splits.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform=None,
        samples: list[tuple[str, int | None]] | None = None,
        list_path: str | Path | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        if samples is not None:
            self.samples = list(samples)
        else:
            index_path = Path(list_path) if list_path is not None else self.root / f"{split}.txt"
            if not index_path.is_file():
                raise FileNotFoundError(f"Missing split index: {index_path}")
            self.samples = load_split_index(index_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        rel_path, label = self.samples[index]
        image = Image.open(self.root / rel_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        # PyTorch's default collator cannot batch None. Prediction ignores -1.
        return image, -1 if label is None else label, rel_path
