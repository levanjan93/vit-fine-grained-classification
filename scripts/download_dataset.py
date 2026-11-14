"""Download doem1997/cs604-course-data and extract the image archives.

The default location is the sibling folder next to this repository.
Re-running this script resumes an incomplete download and skips archives
that were already extracted.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ID = "doem1997/cs604-course-data"
DEFAULT_DIR = Path("data")
ARCHIVES = ("train.tar.gz", "val.tar.gz")


def download(local_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(local_dir),
    )


def extract(local_dir: Path) -> None:
    for name in ARCHIVES:
        archive = local_dir / name
        if not archive.is_file():
            print(f"skip missing archive: {archive}")
            continue
        marker = local_dir / f".extracted_{name}"
        if marker.is_file():
            print(f"already extracted: {name}")
            continue
        print(f"extracting {archive} ...")
        subprocess.run(
            ["tar", "-xzf", str(archive), "-C", str(local_dir)],
            check=True,
        )
        marker.write_text("ok\n", encoding="utf-8")
        print(f"extracted {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    print(f"downloading {REPO_ID} -> {args.local_dir}")
    download(args.local_dir)
    if not args.skip_extract:
        extract(args.local_dir)
    print("done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
