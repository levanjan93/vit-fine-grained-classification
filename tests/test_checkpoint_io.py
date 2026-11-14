"""Verified checkpoint staging and numbered resume copies."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cs604.engine import checkpoint_io


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 2)

    def get_classifier(self):
        return self.head

    def forward(self, images):
        return self.head(images)


class CheckpointIOTests(unittest.TestCase):
    def setUp(self):
        output_root = Path(__file__).resolve().parents[2] / "cs604-outputs"
        output_root.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="test-checkpoint-io-", dir=output_root)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.staging = self.root / "staging"
        self.staging.mkdir()
        self.env = patch.dict(
            "os.environ", {"CS604_CHECKPOINT_STAGING": str(self.staging)}, clear=False
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def _save(self, dest: Path, step: int = 1) -> dict:
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        cfg = {"step": step, "mode": "head"}
        return checkpoint_io.save_training_checkpoint(
            dest, model, optimizer, scheduler, cfg, step, 0, step, 0.439
        )

    def test_save_creates_verified_versioned_copy_and_manifest(self):
        output = self.root / "run"
        entry = self._save(output / "last.pt", step=3)
        versioned = output / "checkpoints" / "resume_step_000003.pt"
        self.assertTrue(versioned.is_file())
        self.assertTrue((output / "last.pt").is_file())
        self.assertEqual(entry["sha256"], checkpoint_io.sha256_file(output / "last.pt"))
        manifest = [
            json.loads(line)
            for line in (output / "checkpoint_hashes.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest[0]["versioned"], str(versioned))

    def test_truncated_newest_copy_fails_while_earlier_resume_still_loads(self):
        output = self.root / "run"
        self._save(output / "last.pt", step=1)
        self._save(output / "last.pt", step=2)
        newest = output / "checkpoints" / "resume_step_000002.pt"
        previous = output / "checkpoints" / "resume_step_000001.pt"
        with newest.open("r+b") as handle:
            handle.truncate(handle.tell() // 2)
        with self.assertRaises(Exception):
            torch.load(newest, map_location="cpu", weights_only=False)
        loaded = torch.load(previous, map_location="cpu", weights_only=False)
        self.assertEqual(loaded["step"], 1)

    def test_elapsed_time_is_saved_for_resume(self):
        output = self.root / "run"
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        checkpoint_io.save_training_checkpoint(
            output / "last.pt", model, optimizer, scheduler, {"mode": "head"},
            5, 0, 5, 0.439, elapsed_sec=1234.5,
        )
        loaded = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        self.assertEqual(loaded["elapsed_sec"], 1234.5)


if __name__ == "__main__":
    unittest.main()
