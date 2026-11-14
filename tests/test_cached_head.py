"""Feature cache and cached-feature head training (CPU, tiny synthetic data)."""

from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cs604.engine.cached_head import (  # noqa: E402
    CachedFeatures,
    epoch_batches,
    evaluate_features,
    train_cached_head,
    view_schedule,
)
from cs604.engine.feature_cache import CACHE_FORMAT, SeededViewDataset, per_sample_seed  # noqa: E402
from cs604.models.build import (  # noqa: E402
    CosineClassifier,
    Fp32Linear,
    apply_class_prototypes,
    build_feature_head,
    optimizer_param_groups,
)


class RandomTransformImages(Dataset):
    """Each item draws from Python's random (like timm's crop) and torch (flip, jitter)."""

    def __len__(self) -> int:
        return 6

    def __getitem__(self, index: int):
        return torch.tensor([random.random(), float(torch.rand(()))]), index % 2, f"img{index}"


class SeededViewTests(unittest.TestCase):
    def test_a_view_does_not_depend_on_visit_order(self) -> None:
        forward = SeededViewDataset(RandomTransformImages(), view_seed=3)
        first = {i: forward[i][0] for i in range(6)}
        backward = {i: forward[i][0] for i in reversed(range(6))}
        for i in range(6):
            torch.testing.assert_close(first[i], backward[i])
        self.assertEqual(forward[4][2], 4)  # the row index replaces the path

    def test_different_view_seeds_give_different_views(self) -> None:
        one = SeededViewDataset(RandomTransformImages(), view_seed=1)[0][0]
        two = SeededViewDataset(RandomTransformImages(), view_seed=2)[0][0]
        self.assertFalse(torch.equal(one, two))
        self.assertNotEqual(per_sample_seed(1, 5), per_sample_seed(2, 5))
        with self.assertRaises(ValueError):
            per_sample_seed(1, 10_000_000)


class BatchOrderTests(unittest.TestCase):
    def test_cached_batches_match_the_training_loader(self) -> None:
        from cs604.engine.loop import _loader

        class Indexed(Dataset):
            def __len__(self):
                return 23

            def __getitem__(self, index):
                return index, 0, ""

        cfg = {"batch_size": 5, "num_workers": 0}
        for epoch in (0, 3):
            loop_batches = [b[0].tolist() for b in _loader(Indexed(), cfg, True, 7 + epoch)]
            cached = [b.tolist() for b in epoch_batches(23, 5, 7, epoch)]
            self.assertEqual(loop_batches, cached)

    def test_view_schedule_uses_every_view_once_per_cycle(self) -> None:
        schedule = view_schedule(rows=50, views=4, epochs=10, seed=0)
        self.assertEqual(tuple(schedule.shape), (10, 50))
        for cycle in (schedule[0:4], schedule[4:8]):
            self.assertTrue(torch.equal(cycle.sort(dim=0).values, torch.arange(4)[:, None].expand(4, 50)))
        torch.testing.assert_close(schedule, view_schedule(50, 4, 10, 0))


class FeatureHeadTests(unittest.TestCase):
    def test_cosine_scale_and_frozen_temperature(self) -> None:
        head = build_feature_head(
            {"head_type": "cosine", "num_classes": 3, "cosine_scale": 40.0,
             "cosine_scale_trainable": False, "cosine_head_fp32": True}, 4)
        classifier = head.get_classifier()
        self.assertIsInstance(classifier, CosineClassifier)
        self.assertAlmostEqual(float(classifier.log_scale.exp()), 40.0, places=4)
        self.assertFalse(classifier.log_scale.requires_grad)
        params = optimizer_param_groups(head, 0.1, head_lr=0.2)[0]["params"]
        self.assertEqual([p.shape for p in params], [classifier.weight.shape])
        self.assertEqual(set(head.state_dict()), {"head.weight", "head.log_scale"})

    def test_frozen_temperature_needs_a_cosine_head(self) -> None:
        with self.assertRaisesRegex(ValueError, "cosine"):
            build_feature_head({"head_type": "linear", "num_classes": 3, "cosine_scale_trainable": False}, 4)

    def test_build_vit_honours_the_same_keys(self) -> None:
        from unittest.mock import patch

        from cs604.models.build import build_from_config
        from tests.test_pipeline import TinyViT

        cfg = {"timm_checkpoint": "tiny", "num_classes": 3, "mode": "head", "head_type": "cosine",
               "cosine_scale": 40.0, "cosine_scale_trainable": False}
        with patch("cs604.models.build.timm.create_model", return_value=TinyViT(0)):
            model = build_from_config(cfg)
        self.assertAlmostEqual(float(model.head.log_scale.exp()), 40.0, places=4)
        self.assertFalse(model.head.log_scale.requires_grad)
        self.assertTrue(model.head.weight.requires_grad)
        with patch("cs604.models.build.timm.create_model", return_value=TinyViT(0)):
            linear = build_from_config({**cfg, "head_type": "linear", "linear_head_fp32": True,
                                        "cosine_scale_trainable": True})
        self.assertIsInstance(linear.head, Fp32Linear)

    def test_fp32_linear_keeps_float32_logits_under_autocast(self) -> None:
        layer = Fp32Linear(4, 3)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertEqual(layer(torch.randn(2, 4)).dtype, torch.float32)
            self.assertEqual(torch.nn.Linear(4, 3)(torch.randn(2, 4)).dtype, torch.bfloat16)


def write_synthetic_cache(root: Path, views: int = 3, classes: int = 4, per_class: int = 12) -> dict:
    """Clustered features: class means plus noise; the holdout is drawn the same way."""

    generator = torch.Generator().manual_seed(0)
    dim = 8
    means = F.normalize(torch.randn(classes, dim, generator=generator), dim=1) * 3
    train_labels = torch.arange(classes).repeat_interleave(per_class)
    holdout_labels = torch.arange(classes).repeat_interleave(5)

    def sample(labels):
        return means[labels] + 1.5 * torch.randn(len(labels), dim, generator=generator)

    manifest = {"format": CACHE_FORMAT, "timm_checkpoint": "tiny", "views": {}}
    entries = {"holdout": [sample(holdout_labels)], "clean": [sample(train_labels)],
               "aug": [sample(train_labels) for _ in range(views)]}
    for group, arrays in entries.items():
        for number, array in enumerate(arrays):
            dtype = "float32" if group == "holdout" else "float16"
            path = root / group / f"view_{number:02d}"
            path.mkdir(parents=True)
            data = array.numpy().astype(dtype)
            np.save(path / "features.npy", data)
            manifest["views"].setdefault(group, {})[f"view_{number:02d}"] = {
                "shape": list(data.shape), "dtype": dtype}
    np.save(root / "labels_train.npy", train_labels.numpy())
    np.save(root / "labels_holdout.npy", holdout_labels.numpy())
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    clean = torch.from_numpy(np.load(root / "clean" / "view_00" / "features.npy")).float()
    prototypes = torch.stack([F.normalize(clean[train_labels == c], dim=1).mean(0) for c in range(classes)])
    torch.save({"prototypes": prototypes, "timm_checkpoint": "tiny", "split_seed": 604,
                "val_per_class": 5}, root / "prototypes.pt")
    return {"classes": classes, "prototypes": F.normalize(prototypes, dim=1)}


class CachedTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.info = write_synthetic_cache(self.root)
        self.cfg = {
            "timm_checkpoint": "tiny", "num_classes": self.info["classes"], "mode": "head",
            "head_type": "cosine", "cosine_head_fp32": True,
            "prototype_file": str(self.root / "prototypes.pt"), "split_seed": 604, "val_per_class": 5,
            "batch_size": 8, "epochs": 4, "lr": 0.05, "weight_decay": 0.0, "precision": "fp32",
            "seed": 0, "expected_initial_accuracy": None,
        }

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_step_zero_equals_nearest_prototype_accuracy(self) -> None:
        data = CachedFeatures(self.root, ["aug"], torch.device("cpu"))
        self.assertEqual(tuple(data.train.shape), (3, 48, 8))
        self.assertEqual(data.train.dtype, torch.float16)
        head = build_feature_head(self.cfg, 8)
        apply_class_prototypes(head, self.info["prototypes"])
        direct = (F.normalize(data.holdout, dim=1) @ self.info["prototypes"].T).argmax(1)
        expected = int((direct == data.holdout_labels).sum()) / len(direct)
        self.assertEqual(evaluate_features(head, data.holdout, data.holdout_labels, None)["accuracy"], expected)
        result = train_cached_head({**self.cfg, "expected_initial_accuracy": expected}, data, log=lambda _m: None)
        self.assertEqual(result["initial_accuracy"], expected)

    def test_training_fits_and_counts_steps_like_the_loop(self) -> None:
        data = CachedFeatures(self.root, ["aug"], torch.device("cpu"))
        result = train_cached_head(self.cfg, data, log=lambda _m: None)
        self.assertEqual(result["optimizer_steps"], 4 * 6)  # ceil(48 / 8) per epoch
        self.assertEqual(result["warmup_steps"], 1)
        self.assertEqual(len(result["curve"]), 4)
        self.assertLess(result["curve"][-1]["train_loss"], result["curve"][0]["train_loss"])
        self.assertNotAlmostEqual(result["final_scale"], 20.0, places=3)  # temperature trained
        again = train_cached_head(self.cfg, data, log=lambda _m: None)
        self.assertEqual(result["final_accuracy"], again["final_accuracy"])

    def test_max_steps_and_failed_gate(self) -> None:
        data = CachedFeatures(self.root, ["clean"], torch.device("cpu"))
        result = train_cached_head({**self.cfg, "max_steps": 5}, data, log=lambda _m: None)
        self.assertEqual(result["optimizer_steps"], 5)
        self.assertEqual(result["total_steps"], 5)
        with self.assertRaisesRegex(ValueError, "Step-zero gate"):
            train_cached_head({**self.cfg, "expected_initial_accuracy": 0.0001,
                               "initial_accuracy_tolerance": 0.0}, data, log=lambda _m: None)

    def test_rejects_other_backbones_and_modes(self) -> None:
        data = CachedFeatures(self.root, ["clean"], torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "different backbone"):
            train_cached_head({**self.cfg, "timm_checkpoint": "other"}, data)
        with self.assertRaisesRegex(ValueError, "mode: head"):
            train_cached_head({**self.cfg, "mode": "lora"}, data)


class BuildTransformsTests(unittest.TestCase):
    def test_train_crop_scale_and_keys_reach_the_shared_transform(self) -> None:
        import timm

        from cs604.engine.loop import _transform_crop_scale, build_transforms

        model = timm.create_model("vit_base_patch16_224.orig_in21k", pretrained=False, num_classes=3)
        default, evaluation = build_transforms({}, model)
        self.assertEqual(_transform_crop_scale(default), [0.08, 1.0])
        self.assertIsNone(_transform_crop_scale(evaluation))
        narrow, _ = build_transforms({"train_crop_scale": [0.35, 1.0], "hflip": 0.0}, model)
        self.assertEqual(_transform_crop_scale(narrow), [0.35, 1.0])
        self.assertNotIn("RandomHorizontalFlip", repr(narrow))


if __name__ == "__main__":
    unittest.main()
