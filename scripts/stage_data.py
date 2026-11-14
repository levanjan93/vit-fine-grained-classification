"""Put the CS604 archives on a fast local disk and extract them.

Google Drive is the cache. Training reads images from ``--local-dir``
(Colab's ``/content`` disk, or the desktop dataset folder).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

REPO_ID = "doem1997/cs604-course-data"
ALL_ARCHIVES = ("train.tar.gz", "val.tar.gz")

TRAIN_TXT_MIN_LINES = 350_000
VAL_TXT_MIN_LINES = 40_000
# Rough lower bound: compressed train tar + extracted images on /content during extract.
MIN_LOCAL_FREE_GIB_TRAIN = 35.0

STREAM_CHUNK_BYTES = 8 * 1024**2
STREAM_REPORT_BYTES = 2 * 1024**3
STREAM_ATTEMPTS = 3
# Extracted train/ is ~40 GiB (JPEGs barely compress); keep headroom for checkpoint staging.
MIN_LOCAL_FREE_GIB_STREAM = 50.0

ARCHIVE_CHECKS: dict[str, tuple[tuple[str, ...], int | None]] = {
    "train.tar.gz": (("train.txt", "train"), TRAIN_TXT_MIN_LINES),
    "val.tar.gz": (("val.txt", "val"), VAL_TXT_MIN_LINES),
}


def count_lines(path: Path) -> int:
    with path.open(encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def extract_marker(local_dir: Path, archive_name: str) -> Path:
    return local_dir / f".extracted_{archive_name}"


def archive_layout_ok(local_dir: Path, archive_name: str) -> bool:
    spec = ARCHIVE_CHECKS.get(archive_name)
    if spec is None:
        return True
    required, min_lines = spec
    for name in required:
        if not (local_dir / name).exists():
            return False
    list_file = next((local_dir / name for name in required if name.endswith(".txt")), None)
    if list_file is not None and min_lines is not None:
        if count_lines(list_file) < min_lines:
            return False
    return True


def report_disk(path: Path, *, label: str = "disk") -> float:
    usage = shutil.disk_usage(path)
    free_gib = usage.free / (1024**3)
    total_gib = usage.total / (1024**3)
    print(f"{label} {path}: {free_gib:.1f} GiB free of {total_gib:.1f} GiB total", flush=True)
    return free_gib


def resolve_archive_path(root: Path, downloaded: Path, name: str) -> Path:
    candidate = root / name
    if candidate.is_file():
        return candidate
    if downloaded.is_file():
        return downloaded
    raise FileNotFoundError(f"Could not locate {name} under {root} (download returned {downloaded})")


def reclaim_vm_space_for_train(local_dir: Path) -> None:
    """Drop val artifacts from the VM so train extract fits on /content."""
    for name in ("val", "val.txt", "val.tar.gz", ".extracted_val.tar.gz"):
        path = local_dir / name
        if path.is_dir():
            print(f"reclaim VM space: removing {path}", flush=True)
            shutil.rmtree(path)
        elif path.is_file():
            print(f"reclaim VM space: removing {path}", flush=True)
            path.unlink()


def _download_from_hub(name: str, cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"downloading {name} from {REPO_ID} into {cache_dir}", flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=name,
                local_dir=str(cache_dir),
                token=token,
            )
        )
    except GatedRepoError as error:
        raise SystemExit(
            "Hugging Face dataset access denied. Accept terms at "
            f"https://huggingface.co/datasets/{REPO_ID} then set HF_TOKEN "
            "(Colab Secrets or a local token file)."
        ) from error
    except HfHubHTTPError as error:
        raise SystemExit(f"Hugging Face download failed for {name}: {error}") from error
    return resolve_archive_path(cache_dir, downloaded, name)


def _ensure_archive(
    name: str,
    local_dir: Path,
    drive_dir: Path | None,
    *,
    prefer_drive_cache: bool,
) -> Path:
    local = local_dir / name
    if local.is_file():
        return local
    if drive_dir is not None:
        cached = drive_dir / name
        if cached.is_file():
            if prefer_drive_cache:
                print(f"using Drive archive (no local copy): {cached}", flush=True)
                return cached
            print(f"copying {name} from Drive to local disk", flush=True)
            local_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, local)
            return local

    cache_dir = drive_dir if drive_dir is not None else local_dir
    archive = _download_from_hub(name, cache_dir)
    if drive_dir is not None and cache_dir == drive_dir:
        print(f"cached {name} on Drive at {archive}", flush=True)
    elif drive_dir is not None and not (drive_dir / name).is_file() and archive.is_file():
        drive_dir.mkdir(parents=True, exist_ok=True)
        dest = drive_dir / name
        try:
            print(f"caching {name} on Drive for the next session", flush=True)
            shutil.copy2(archive, dest)
        except OSError as error:
            print(f"warning: could not cache {name} on Drive ({error}); continuing", flush=True)
    return archive if archive.is_file() else local_dir / name


def _extract(archive: Path, local_dir: Path) -> None:
    marker = extract_marker(local_dir, archive.name)
    if marker.is_file() and archive_layout_ok(local_dir, archive.name):
        print(f"already extracted: {archive.name}", flush=True)
        return
    if marker.is_file():
        print(
            f"stale or incomplete extract for {archive.name}; removing marker and re-extracting",
            flush=True,
        )
        marker.unlink(missing_ok=True)
    if not archive.is_file():
        raise SystemExit(
            f"Missing archive {archive.name}. Upload to Drive cs604/data/ or allow Hugging Face download."
        )
    print(f"extracting {archive.name} from {archive}", flush=True)
    try:
        subprocess.run(["tar", "-xzf", str(archive), "-C", str(local_dir)], check=True)
    except subprocess.CalledProcessError as error:
        free = report_disk(local_dir, label="disk after failed tar")
        raise SystemExit(
            f"tar failed for {archive.name} (exit {error.returncode}). "
            f"Free space now {free:.1f} GiB — need ~{MIN_LOCAL_FREE_GIB_TRAIN:.0f}+ GiB on /content for train."
        ) from error
    _finish_extract(local_dir, archive.name)


def _finish_extract(local_dir: Path, archive_name: str) -> None:
    marker = extract_marker(local_dir, archive_name)
    if not archive_layout_ok(local_dir, archive_name):
        marker.unlink(missing_ok=True)
        list_path = local_dir / "train.txt" if archive_name == "train.tar.gz" else local_dir / "val.txt"
        lines = count_lines(list_path) if list_path.is_file() else 0
        raise SystemExit(
            f"After extracting {archive_name}, layout check failed under {local_dir}. "
            f"train.txt/val.txt lines seen: {lines}. Archive may be corrupt or disk full."
        )
    marker.write_text("ok\n", encoding="utf-8")


def pipe_into_tar(source_cmd: list[str], local_dir: Path, *, source_input: bytes = b"") -> None:
    """Run ``source_cmd`` and feed its stdout to ``tar -xz`` without touching disk."""
    local_dir.mkdir(parents=True, exist_ok=True)
    source = subprocess.Popen(source_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    tar = subprocess.Popen(["tar", "-xzf", "-", "-C", str(local_dir)], stdin=subprocess.PIPE)
    assert source.stdin is not None and source.stdout is not None and tar.stdin is not None
    source.stdin.write(source_input)
    source.stdin.close()
    copied = 0
    next_report = STREAM_REPORT_BYTES
    try:
        while chunk := source.stdout.read(STREAM_CHUNK_BYTES):
            tar.stdin.write(chunk)
            copied += len(chunk)
            if copied >= next_report:
                print(f"  streamed {copied / 1024**3:.1f} GiB", flush=True)
                next_report += STREAM_REPORT_BYTES
    except BrokenPipeError:
        pass  # tar exited early; its return code below says why
    finally:
        tar.stdin.close()
        source.stdout.close()
    source_code = source.wait()
    tar_code = tar.wait()
    if source_code != 0 or tar_code != 0:
        raise subprocess.CalledProcessError(
            source_code or tar_code,
            f"download exit {source_code}, tar exit {tar_code} after {copied / 1024**3:.1f} GiB",
        )
    print(f"  streamed {copied / 1024**3:.1f} GiB total", flush=True)


def list_file_name(archive_name: str) -> str | None:
    required, _ = ARCHIVE_CHECKS.get(archive_name, ((), None))
    return next((entry for entry in required if entry.endswith(".txt")), None)


def ensure_list_file(archive_name: str, local_dir: Path) -> None:
    """Fetch train.txt / val.txt: they sit beside the archives on the Hub, not inside them."""
    name = list_file_name(archive_name)
    if name is None:
        return
    _, min_lines = ARCHIVE_CHECKS[archive_name]
    path = local_dir / name
    if path.is_file() and (min_lines is None or count_lines(path) >= min_lines):
        return
    path.unlink(missing_ok=True)
    _download_from_hub(name, local_dir)
    print(f"{name}: {count_lines(path)} lines", flush=True)


def images_match_list(local_dir: Path, archive_name: str) -> bool:
    """True when every image named in the list file exists and is non-empty."""
    name = list_file_name(archive_name)
    if name is None or not archive_layout_ok(local_dir, archive_name):
        return False
    with (local_dir / name).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            image = local_dir / parts[0]
            try:
                if image.stat().st_size == 0:
                    return False
            except OSError:
                return False
    return True


def remove_partial_extract(local_dir: Path, archive_name: str) -> None:
    """Drop an unverified extract and HF download cache left by a failed attempt."""
    required, _ = ARCHIVE_CHECKS.get(archive_name, ((), None))
    # The .txt list is a separate repo file, not part of the extract; keep it.
    trees = [local_dir / entry for entry in required if not entry.endswith(".txt")]
    for path in trees + [local_dir / ".cache"]:
        if path.is_dir():
            print(f"removing partial {path}", flush=True)
            shutil.rmtree(path)
        elif path.is_file():
            print(f"removing partial {path}", flush=True)
            path.unlink()


def check_stream_space(local_dir: Path, min_free_gib: float = MIN_LOCAL_FREE_GIB_STREAM) -> None:
    """Fail before downloading, not after tar fills the disk 40 GiB in."""
    local_dir.mkdir(parents=True, exist_ok=True)
    free = report_disk(local_dir, label="VM")
    if free < min_free_gib:
        raise SystemExit(
            f"Only {free:.1f} GiB free on {local_dir}; streaming train needs ~{min_free_gib:.0f}+ GiB. "
            f"Leftovers from earlier attempts are the usual cause: delete {local_dir} "
            "(shutil.rmtree) or use Runtime > Disconnect and delete runtime, then re-run."
        )


def _stream_from_hub(name: str, local_dir: Path) -> None:
    """Download straight into tar so the .tar.gz never lands on disk."""
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(repo_id=REPO_ID, filename=name, repo_type="dataset")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    # Token goes to curl on stdin (-K -), not argv. curl -L drops it on the CDN redirect.
    config = f'header = "Authorization: Bearer {token}"\n'.encode() if token else b""
    curl = ["curl", "-fsSL", "--retry", "5", "--retry-connrefused", "-K", "-", url]

    marker = extract_marker(local_dir, name)
    marker.unlink(missing_ok=True)
    remove_partial_extract(local_dir, name)
    check_stream_space(local_dir)
    for attempt in range(1, STREAM_ATTEMPTS + 1):
        print(f"streaming {name} from {REPO_ID} into {local_dir} (attempt {attempt})", flush=True)
        try:
            pipe_into_tar(curl, local_dir, source_input=config)
            break
        except subprocess.CalledProcessError as error:
            free = report_disk(local_dir, label="disk after failed stream")
            print(f"stream failed: {error.cmd}", flush=True)
            if attempt == STREAM_ATTEMPTS:
                raise SystemExit(
                    f"Streaming {name} failed {STREAM_ATTEMPTS} times ({free:.1f} GiB free). "
                    "curl exit 22 usually means access denied: accept terms at "
                    f"https://huggingface.co/datasets/{REPO_ID} and set HF_TOKEN."
                ) from error
    _finish_extract(local_dir, name)


def validate_training_layout(local_dir: Path) -> None:
    train_txt = local_dir / "train.txt"
    if not train_txt.is_file():
        raise SystemExit(
            f"Missing {train_txt}. Run stage_data with a valid train.tar.gz "
            "(Drive cs604/data/ or Hugging Face download)."
        )
    lines = count_lines(train_txt)
    if lines < TRAIN_TXT_MIN_LINES:
        raise SystemExit(
            f"{train_txt} has only {lines} lines (expected at least {TRAIN_TXT_MIN_LINES}). "
            "Delete stale .extracted_* markers and re-run stage_data."
        )
    if not (local_dir / "train").is_dir():
        raise SystemExit(f"Missing image tree {local_dir / 'train'} after extract.")
    print(f"train split ok: {lines} lines in train.txt", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--drive-dir", type=Path)
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Stage only train.tar.gz (enough for LoRA training; saves VM disk).",
    )
    parser.add_argument(
        "--reclaim-vm-for-train",
        action="store_true",
        help="Remove val artifacts from --local-dir before staging train (Colab disk).",
    )
    parser.add_argument(
        "--vm-only",
        action="store_true",
        help="Keep course tarballs off Drive (download/extract on --local-dir only).",
    )
    parser.add_argument(
        "--delete-local-archive-after-extract",
        action="store_true",
        help="Remove the .tar.gz from --local-dir after a successful extract (saves VM disk).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Pipe the Hugging Face download straight into tar; the .tar.gz never hits disk. "
        "Implies --vm-only.",
    )
    args = parser.parse_args(argv)
    if args.stream:
        args.vm_only = True

    archive_drive_dir = None if args.vm_only else args.drive_dir
    if args.vm_only:
        print(
            "vm-only: course archives will not be stored on Drive "
            "(use when Drive quota is too small for train.tar.gz).",
            flush=True,
        )

    args.local_dir.mkdir(parents=True, exist_ok=True)
    if archive_drive_dir is not None:
        report_disk(archive_drive_dir, label="Drive cache")
    free = report_disk(args.local_dir, label="VM")

    if args.train_only and args.reclaim_vm_for_train:
        reclaim_vm_space_for_train(args.local_dir)
        free = report_disk(args.local_dir, label="VM after reclaim")

    if args.train_only and free < MIN_LOCAL_FREE_GIB_TRAIN:
        print(
            f"warning: only {free:.1f} GiB free on VM; train extract typically needs "
            f"~{MIN_LOCAL_FREE_GIB_TRAIN:.0f}+ GiB on /content",
            flush=True,
        )

    prefer_drive = archive_drive_dir is not None
    archives = ("train.tar.gz",) if args.train_only else ALL_ARCHIVES
    for name in archives:
        ensure_list_file(name, args.local_dir)
        if args.stream and not (args.local_dir / name).is_file():
            marker = extract_marker(args.local_dir, name)
            if marker.is_file() and archive_layout_ok(args.local_dir, name):
                print(f"already extracted: {name}", flush=True)
            elif images_match_list(args.local_dir, name):
                print(f"found complete extract of {name} from an earlier run; skipping download", flush=True)
                marker.write_text("ok\n", encoding="utf-8")
            else:
                _stream_from_hub(name, args.local_dir)
            continue
        archive = _ensure_archive(
            name,
            args.local_dir,
            archive_drive_dir,
            prefer_drive_cache=prefer_drive,
        )
        _extract(archive, args.local_dir)
        local_copy = args.local_dir / name
        if args.delete_local_archive_after_extract and local_copy.is_file():
            local_copy.unlink()
            print(f"removed local {name} after extract (VM space)", flush=True)
        elif (
            local_copy.is_file()
            and prefer_drive
            and archive_drive_dir is not None
            and (archive_drive_dir / name).is_file()
        ):
            local_copy.unlink()
            print(f"removed local copy of {name}; using Drive archive", flush=True)

    validate_training_layout(args.local_dir)
    print("data ready", args.local_dir, flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as error:
        if error.code not in (None, 0):
            message = str(error).strip()
            if message:
                print(message, file=sys.stderr, flush=True)
        raise
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print(f"stage_data failed: {error}", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
