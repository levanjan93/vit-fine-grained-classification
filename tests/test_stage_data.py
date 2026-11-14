from __future__ import annotations

import importlib.util
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


def _load_stage_data():
    path = Path(__file__).resolve().parents[1] / "scripts" / "stage_data.py"
    spec = importlib.util.spec_from_file_location("stage_data", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


stage_data = _load_stage_data()


class StageDataLayoutTests(unittest.TestCase):
    def test_archive_layout_requires_train_txt_and_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(stage_data.archive_layout_ok(root, "train.tar.gz"))
            (root / "train.txt").write_text("a\n", encoding="utf-8")
            self.assertFalse(stage_data.archive_layout_ok(root, "train.tar.gz"))
            (root / "train").mkdir()
            self.assertFalse(stage_data.archive_layout_ok(root, "train.tar.gz"))
            (root / "train.txt").write_text("line\n" * stage_data.TRAIN_TXT_MIN_LINES, encoding="utf-8")
            self.assertTrue(stage_data.archive_layout_ok(root, "train.tar.gz"))

    def test_stale_marker_detected_when_layout_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = stage_data.extract_marker(root, "train.tar.gz")
            marker.write_text("ok\n", encoding="utf-8")
            self.assertFalse(stage_data.archive_layout_ok(root, "train.tar.gz"))


class StreamSpaceTests(unittest.TestCase):
    def test_low_space_fails_before_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as caught:
                stage_data.check_stream_space(Path(tmp), min_free_gib=1e9)
            self.assertIn("free", str(caught.exception))

    def test_enough_space_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage_data.check_stream_space(Path(tmp), min_free_gib=0.0)

    def test_partial_extract_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train" / "0").mkdir(parents=True)
            (root / "train.txt").write_text("a\n", encoding="utf-8")
            (root / ".cache" / "huggingface").mkdir(parents=True)
            (root / "keep.txt").write_text("x", encoding="utf-8")
            stage_data.remove_partial_extract(root, "train.tar.gz")
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["keep.txt", "train.txt"])


class ListFileTests(unittest.TestCase):
    def _write_split(self, root: Path, lines: int) -> None:
        (root / "train" / "class_0000").mkdir(parents=True)
        rows = []
        for index in range(lines):
            rel = f"train/class_0000/{index}.jpg"
            (root / rel).write_bytes(b"jpeg")
            rows.append(f"{rel} 0\n")
        (root / "train.txt").write_text("".join(rows), encoding="utf-8")

    def test_images_match_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_min = stage_data.ARCHIVE_CHECKS["train.tar.gz"]
            stage_data.ARCHIVE_CHECKS["train.tar.gz"] = (old_min[0], 3)
            try:
                self._write_split(root, 3)
                self.assertTrue(stage_data.images_match_list(root, "train.tar.gz"))
                (root / "train" / "class_0000" / "2.jpg").write_bytes(b"")
                self.assertFalse(stage_data.images_match_list(root, "train.tar.gz"))
                (root / "train" / "class_0000" / "2.jpg").unlink()
                self.assertFalse(stage_data.images_match_list(root, "train.tar.gz"))
            finally:
                stage_data.ARCHIVE_CHECKS["train.tar.gz"] = old_min

    def test_ensure_list_file_keeps_full_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train.txt").write_text("x 0\n" * stage_data.TRAIN_TXT_MIN_LINES, encoding="utf-8")
            stage_data.ensure_list_file("train.tar.gz", root)  # no download attempted
            self.assertEqual(stage_data.count_lines(root / "train.txt"), stage_data.TRAIN_TXT_MIN_LINES)


class PipeIntoTarTests(unittest.TestCase):
    def _make_archive(self, root: Path) -> Path:
        src = root / "src"
        (src / "train" / "0").mkdir(parents=True)
        (src / "train" / "0" / "a.jpg").write_bytes(b"jpeg")
        (src / "train.txt").write_text("train/0/a.jpg 0\n", encoding="utf-8")
        archive = root / "train.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(src / "train", arcname="train")
            handle.add(src / "train.txt", arcname="train.txt")
        return archive

    def test_streams_archive_into_local_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = self._make_archive(root)
            out = root / "out"
            # Mimic curl: read config from stdin, write the archive bytes to stdout.
            source = [
                sys.executable,
                "-c",
                "import sys, shutil; sys.stdin.read(); "
                f"shutil.copyfileobj(open({str(archive)!r}, 'rb'), sys.stdout.buffer)",
            ]
            stage_data.pipe_into_tar(source, out, source_input=b"header = x\n")
            self.assertEqual((out / "train" / "0" / "a.jpg").read_bytes(), b"jpeg")
            self.assertTrue((out / "train.txt").is_file())
            self.assertFalse((out / "train.tar.gz").exists())

    def test_failed_download_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = [sys.executable, "-c", "import sys; sys.exit(22)"]
            with self.assertRaises(subprocess.CalledProcessError):
                stage_data.pipe_into_tar(source, Path(tmp) / "out")


if __name__ == "__main__":
    unittest.main()
