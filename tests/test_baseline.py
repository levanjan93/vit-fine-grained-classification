"""Exercise baseline protection with tiny CPU models, without course images/downloads."""

from __future__ import annotations

import csv
import json
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cs604.engine import loop
from cs604.models.build import CosineClassifier, build_from_config
from cs604.engine.baseline import (
    assess_initial_accuracy, baseline_provenance, check_prototype_split,
    validate_baseline_setup,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 2)

    def get_classifier(self):
        return self.head

    def forward(self, images):
        return self.head(images)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        # Keep even temporary test checkpoints in the project output area.
        output_root = Path(__file__).resolve().parents[2] / "cs604-outputs"
        output_root.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="test-baseline-", dir=output_root)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def run_fixture(self, accuracies, evaluate_only=False, resume=None, epochs=1, expected=0.439):
        output = self.root / "run"
        split = output / "splits" / "internal_val.txt"
        split.parent.mkdir(parents=True, exist_ok=True)
        split.write_text("train/a.jpg 0\ntrain/b.jpg 1\n", encoding="utf-8")
        model = TinyModel()
        initial = {key: value.clone() for key, value in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        cfg = {
            "output_dir": str(output), "timm_checkpoint": "tiny", "num_classes": 2,
            "mode": "head", "batch_size": 2, "epochs": epochs, "lr": 0.01,
            "precision": "fp32", "num_workers": 0, "log_every": 1, "ckpt_every": 1,
            "evaluate_before_training": True, "evaluate_only": evaluate_only,
            "expected_initial_accuracy": expected, "initial_accuracy_tolerance": 0.001,
            "expected_initial_samples": 2, "resume": resume,
        }
        dataset = [(torch.tensor([1., 0.]), 0, "train/a.jpg"),
                   (torch.tensor([0., 1.]), 1, "train/b.jpg")]
        results = [{"accuracy": accuracy, "loss": 1., "images_per_sec": 10., "elapsed_sec": 0.2}
                   for accuracy in accuracies]
        cpu = torch.device("cpu")
        with ExitStack() as stack:
            stack.enter_context(patch.object(loop, "_seed"))
            stack.enter_context(patch.object(loop, "_git_metadata", return_value={"git_commit": "test"}))
            stack.enter_context(patch.object(loop, "build_from_config", return_value=model))
            stack.enter_context(patch.object(loop, "_make_datasets", return_value=(dataset, dataset)))
            loader = stack.enter_context(patch.object(
                loop, "_loader", side_effect=lambda data, cfg, shuffle, seed: DataLoader(data, batch_size=2)
            ))
            evaluation = stack.enter_context(patch.object(loop, "evaluate", side_effect=results))
            stack.enter_context(patch("timm.data.resolve_model_data_config", return_value={}))
            stack.enter_context(patch("timm.data.create_transform", return_value=None))
            stack.enter_context(patch.object(torch.accelerator, "is_available", return_value=False))
            # Redirect only the loop's device selection; keep torch.device a real type
            # inside PyTorch's optimizer/serialization internals.
            torch_proxy = SimpleNamespace(**vars(torch))
            torch_proxy.device = lambda *args, **kwargs: cpu
            torch_proxy.cuda = SimpleNamespace(
                is_available=lambda: True,
                get_device_properties=lambda index: SimpleNamespace(total_memory=16 * 1024**3),
                get_device_name=lambda index: "CPU test",
                max_memory_allocated=lambda: 0,
            )
            stack.enter_context(patch.object(loop, "torch", torch_proxy))
            stack.enter_context(patch.object(torch.optim, "AdamW", return_value=optimizer))
            # Wrap before scheduler construction so LambdaLR sees a normal bound step method.
            original_step = optimizer.step
            calls = []

            def counted_step(self, *args, **kwargs):
                calls.append(1)
                return original_step(*args, **kwargs)

            from types import MethodType
            optimizer.step = MethodType(counted_step, optimizer)
            error = None
            try:
                result = loop.run_training(cfg)
            except ValueError as caught:
                error = caught
                result = None
            return SimpleNamespace(
                output=output, model=model, initial=initial, cfg=cfg, result=result,
                steps=len(calls), evaluation_calls=evaluation.call_count,
                train_loads=sum(bool(call.args[2]) for call in loader.call_args_list), error=error,
            )

    def checkpoint(self, path):
        return torch.load(path, map_location="cpu", weights_only=False)

    def test_evaluation_only_preserves_reloadable_model_without_training(self):
        run = self.run_fixture([0.439], evaluate_only=True)
        self.assertIsNone(run.error)
        self.assertEqual(run.result, run.output / "baseline.pt")
        self.assertEqual(run.steps, 0)
        self.assertEqual(run.train_loads, 0)
        self.assertEqual(run.evaluation_calls, 1)
        saved = self.checkpoint(run.result)
        restored = TinyModel()
        restored.load_state_dict(saved["model"])
        for key, value in restored.state_dict().items():
            torch.testing.assert_close(value, run.initial[key])
        self.assertEqual(saved["step"], 0)
        self.assertEqual(saved["best_accuracy"], 0.439)
        self.assertIn("config", saved)  # The schema consumed by predict_split.
        info = json.loads((run.output / "run.json").read_text())
        self.assertEqual(info["optimizer_steps"], 0)
        self.assertEqual(info["status"], "evaluated")
        with (run.output / "metrics.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["phase"] for row in rows], ["initial_val"])

    def test_new_backbone_without_expected_accuracy_records_step_zero(self):
        # A first evaluate-only run on a new backbone measures its gate value.
        run = self.run_fixture([0.30], evaluate_only=True, expected=None)
        self.assertIsNone(run.error)
        baseline = json.loads((run.output / "baseline.json").read_text())
        self.assertTrue(baseline["passed"])
        self.assertEqual(baseline["initial_internal_accuracy"], 0.30)
        self.assertIsNone(baseline["expected_initial_accuracy"])
        self.assertIsNone(baseline["accuracy_delta"])
        info = json.loads((run.output / "run.json").read_text())
        self.assertEqual(info["status"], "evaluated")
        self.assertEqual(info["initial_internal_accuracy"], 0.30)
        self.assertEqual(self.checkpoint(run.output / "best.pt")["best_accuracy"], 0.30)

    def test_worse_training_cannot_replace_initial_best(self):
        run = self.run_fixture([0.439, 0.40])
        self.assertIsNone(run.error)
        self.assertEqual(run.steps, 1)
        best = self.checkpoint(run.output / "best.pt")
        self.assertEqual(best["step"], 0)
        self.assertEqual(best["best_accuracy"], 0.439)
        for key, value in best["model"].items():
            torch.testing.assert_close(value, run.initial[key])
        last = self.checkpoint(run.output / "last.pt")
        self.assertEqual(last["step"], 1)
        self.assertTrue(any(not torch.equal(last["model"][key], value) for key, value in run.initial.items()))

    def test_improvement_replaces_best_but_never_baseline(self):
        run = self.run_fixture([0.439, 0.46])
        self.assertIsNone(run.error)
        best = self.checkpoint(run.output / "best.pt")
        self.assertEqual(best["step"], 1)
        self.assertEqual(best["best_accuracy"], 0.46)
        baseline = self.checkpoint(run.output / "baseline.pt")
        self.assertEqual(baseline["step"], 0)
        for key, value in baseline["model"].items():
            torch.testing.assert_close(value, run.initial[key])

    def test_wrong_initial_accuracy_stops_before_optimizer_updates(self):
        run = self.run_fixture([0.40])
        self.assertRegex(str(run.error), "Initial accuracy check failed")
        self.assertEqual(run.steps, 0)
        self.assertEqual(run.train_loads, 0)
        self.assertTrue((run.output / "baseline.pt").is_file())
        self.assertFalse((run.output / "best.pt").exists())
        self.assertFalse(json.loads((run.output / "baseline.json").read_text())["passed"])
        self.assertEqual(json.loads((run.output / "run.json").read_text())["status"], "initial_validation_failed")

    def test_existing_baseline_is_not_overwritten(self):
        run = self.run_fixture([0.439], evaluate_only=True)
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            validate_baseline_setup(run.cfg, run.output)

    def test_resuming_earlier_checkpoint_keeps_existing_best_and_initial_log(self):
        first = self.run_fixture([0.439, 0.46])
        baseline_bytes = (first.output / "baseline.pt").read_bytes()
        resumed = self.run_fixture([0.45], resume=str(first.output / "baseline.pt"))
        self.assertIsNone(resumed.error)
        self.assertEqual(resumed.evaluation_calls, 1)  # Only the post-training evaluation.
        self.assertEqual(self.checkpoint(resumed.output / "best.pt")["best_accuracy"], 0.46)
        self.assertEqual((resumed.output / "baseline.pt").read_bytes(), baseline_bytes)
        with (resumed.output / "metrics.csv").open() as handle:
            phases = [row["phase"] for row in csv.DictReader(handle)]
        self.assertEqual(phases.count("initial_val"), 1)
        self.assertEqual(phases.count("val"), 2)

    def test_split_and_prototype_metadata_must_match(self):
        current = self.root / "current.txt"
        reference = self.root / "reference.txt"
        current.write_text("a 0\n")
        reference.write_text("b 0\n")
        cfg = {"evaluate_before_training": True, "baseline_reference_split": str(reference), "split_seed": 604,
               "timm_checkpoint": "vit_base_patch16_224.augreg2_in21k_ft_in1k"}
        with self.assertRaisesRegex(ValueError, "holdout does not match"):
            baseline_provenance(cfg, current, 1)
        with self.assertRaisesRegex(ValueError, "split_seed"):
            check_prototype_split(cfg, {"split_seed": 0, "val_per_class": 5})
        with self.assertRaisesRegex(ValueError, "Expected 2"):
            baseline_provenance({"expected_initial_samples": 2}, current, 1)

    def test_prototypes_must_come_from_the_configured_backbone(self):
        in21k = {"timm_checkpoint": "vit_base_patch16_224.orig_in21k", "split_seed": 604}
        split = {"split_seed": 604, "val_per_class": 5}
        check_prototype_split(in21k, {**split, "timm_checkpoint": "vit_base_patch16_224.orig_in21k"})
        with self.assertRaisesRegex(ValueError, "built with vit_base_patch16_224.augreg_in21k"):
            check_prototype_split(in21k, {**split, "timm_checkpoint": "vit_base_patch16_224.augreg_in21k"})
        # Files without the field predate it and were all built with the IN1k fine-tune.
        with self.assertRaisesRegex(ValueError, "augreg2_in21k_ft_in1k"):
            check_prototype_split(in21k, split)
        check_prototype_split({**in21k, "timm_checkpoint": "vit_base_patch16_224.augreg2_in21k_ft_in1k"}, split)
        # The backbone check applies even to runs without a step-zero gate.
        with self.assertRaisesRegex(ValueError, "built with"):
            check_prototype_split({"timm_checkpoint": "vit_base_patch16_224.orig_in21k"}, split)

    def test_accuracy_tolerance_is_absolute_and_nonfinite_values_fail(self):
        cfg = {"expected_initial_accuracy": 0.439, "initial_accuracy_tolerance": 0.001}
        self.assertTrue(assess_initial_accuracy(0.438, cfg)["passed"])
        self.assertTrue(assess_initial_accuracy(0.440, cfg)["passed"])
        self.assertFalse(assess_initial_accuracy(0.4379, cfg)["passed"])
        self.assertFalse(assess_initial_accuracy(float("nan"), cfg)["passed"])

    def test_validation_cannot_be_disabled_for_protected_runs(self):
        with self.assertRaisesRegex(ValueError, "requires a real"):
            validate_baseline_setup({"evaluate_only": True, "validate": False}, self.root)
        with self.assertRaisesRegex(ValueError, "cannot resume"):
            validate_baseline_setup({"evaluate_only": True, "resume": "saved.pt"}, self.root)

    def test_fp32_head_preserves_nearby_scores_and_gradients_under_autocast(self):
        head = CosineClassifier(2, 2, force_fp32=True)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[1., 0.02], [1., 0.01]]))
        inputs = torch.tensor([[1., 0.]], requires_grad=True)
        reference = head(inputs)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            scores = head(inputs)
        self.assertEqual(scores.dtype, torch.float32)
        torch.testing.assert_close(scores, reference)
        self.assertEqual(scores.argmax(dim=1).item(), 1)
        scores.sum().backward()
        for gradient in (inputs.grad, head.weight.grad, head.log_scale.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
        head.force_fp32 = False
        with torch.autocast("cpu", dtype=torch.bfloat16):
            reduced_scores = head(inputs)
        self.assertEqual(reduced_scores.argmax(dim=1).item(), 0)  # Reduced precision ties the scores.

    def test_resuming_preceding_numbered_checkpoint_after_corrupt_latest(self):
        import os

        staging = self.root / "checkpoint-staging"
        staging.mkdir()
        with patch.dict(os.environ, {"CS604_CHECKPOINT_STAGING": str(staging)}, clear=False):
            first = self.run_fixture([0.439, 0.46, 0.47], epochs=2)
        self.assertEqual(first.steps, 2)
        output = first.output
        corrupt = output / "checkpoints" / "resume_step_000002.pt"
        with corrupt.open("r+b") as handle:
            handle.truncate(max(1, handle.tell() // 2))
        with self.assertRaises(Exception):
            self.checkpoint(corrupt)
        previous = output / "checkpoints" / "resume_step_000001.pt"
        resumed = self.run_fixture([0.48], resume=str(previous), epochs=2)
        self.assertIsNone(resumed.error)
        self.assertGreaterEqual(resumed.steps, 1)
        self.assertEqual(self.checkpoint(resumed.output / "last.pt")["step"], 2)

    def test_checkpoint_config_restores_head_precision(self):
        cfg = {"timm_checkpoint": "tiny", "num_classes": 2, "mode": "head",
               "head_type": "cosine", "cosine_head_fp32": True}
        with patch("cs604.models.build.timm.create_model", return_value=TinyModel()):
            model = build_from_config(cfg)
        self.assertTrue(model.head.force_fp32)
        with self.assertRaisesRegex(ValueError, "cosine_head_fp32"):
            loop._check_architecture_compatibility(cfg, {**cfg, "cosine_head_fp32": False})


if __name__ == "__main__":
    unittest.main()
