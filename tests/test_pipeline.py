from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cs604.data.dataset import stratified_split
from cs604.engine.predict import validate_submission
from cs604.models.build import CosineClassifier, LoRALinear, apply_class_prototypes


class SplitTests(unittest.TestCase):
    def test_stratified_split_is_balanced_and_repeatable(self) -> None:
        samples = [
            (f"train/class_{label}/{index}.jpg", label)
            for label in range(3)
            for index in range(8)
        ]
        first = stratified_split(samples, val_per_class=2, train_per_class=3, seed=7)
        second = stratified_split(samples, val_per_class=2, train_per_class=3, seed=7)
        self.assertEqual(first, second)
        train, val = first
        self.assertEqual(len(train), 9)
        self.assertEqual(len(val), 6)
        self.assertEqual({label for _path, label in train}, {0, 1, 2})


class LoRATests(unittest.TestCase):
    def test_only_lora_weights_are_trainable_and_reloadable(self) -> None:
        layer = LoRALinear(nn.Linear(5, 7), rank=2, alpha=4)
        self.assertFalse(layer.base.weight.requires_grad)
        self.assertTrue(layer.lora_a.requires_grad)
        inputs = torch.randn(3, 5)
        original = layer(inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(layer.state_dict(), path)
            reloaded = LoRALinear(nn.Linear(5, 7), rank=2, alpha=4)
            reloaded.load_state_dict(torch.load(path, weights_only=True))
            torch.testing.assert_close(original, reloaded(inputs))
            for key, value in layer.state_dict().items():
                torch.testing.assert_close(value, reloaded.state_dict()[key])


class TinyViT(nn.Module):
    """Just enough of a timm ViT for build_vit: blocks[].attn.qkv, norm, head."""

    def __init__(self, seed: int, width: int = 4, classes: int = 3):
        super().__init__()
        torch.manual_seed(seed)
        self.patch = nn.Linear(width, width)
        self.blocks = nn.ModuleList(
            nn.Module() for _ in range(2)
        )
        for block in self.blocks:
            block.attn = nn.Module()
            block.attn.qkv = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, classes)

    def get_classifier(self):
        return self.head

    def set_grad_checkpointing(self, enabled: bool) -> None:
        pass


class CompactCheckpointTests(unittest.TestCase):
    CFG = {"timm_checkpoint": "tiny", "num_classes": 3, "mode": "lora", "lora_rank": 2,
           "head_type": "cosine", "train_head": False}

    def _build(self, seed: int):
        from unittest.mock import patch

        from cs604.models.build import build_from_config

        with patch("cs604.models.build.timm.create_model", return_value=TinyViT(seed)):
            return build_from_config(self.CFG)

    def test_compact_keeps_only_trained_and_head_weights_and_round_trips(self) -> None:
        from cs604.models.build import checkpoint_model_state, load_model_state

        model = self._build(seed=0)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora" in name or name.startswith("head."):
                    parameter.add_(1.0)
        saved = checkpoint_model_state(model, compact=True)
        self.assertEqual(saved["model_format"], "compact")
        self.assertTrue(all("lora" in key or key.startswith("head.") for key in saved["model"]))
        self.assertIn("head.weight", saved["model"])  # frozen but prototype-initialized
        self.assertNotIn("patch.weight", saved["model"])

        restored = self._build(seed=0)
        load_model_state(restored, saved)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value)

        with self.assertRaisesRegex(ValueError, "Frozen backbone differs"):
            load_model_state(self._build(seed=1), saved)

    def test_full_checkpoints_still_load(self) -> None:
        from cs604.models.build import checkpoint_model_state, load_model_state

        model = self._build(seed=0)
        restored = self._build(seed=1)
        load_model_state(restored, {"model": model.state_dict()})  # pre-compact checkpoints
        torch.testing.assert_close(restored.patch.weight, model.patch.weight)
        self.assertEqual(checkpoint_model_state(model, compact=False)["model_format"], "full")

    def test_compact_default_only_for_frozen_pretrained_backbone(self) -> None:
        from cs604.models.build import use_compact_checkpoints

        self.assertTrue(use_compact_checkpoints({"mode": "lora"}))
        self.assertTrue(use_compact_checkpoints({"mode": "head"}))
        self.assertFalse(use_compact_checkpoints({"mode": "full"}))
        self.assertFalse(use_compact_checkpoints({"mode": "lora", "init_checkpoint": "x.pt"}))
        self.assertFalse(use_compact_checkpoints({"mode": "lora", "compact_checkpoints": False}))
        with self.assertRaisesRegex(ValueError, "compact_checkpoints"):
            use_compact_checkpoints({"mode": "full", "compact_checkpoints": True})


class HeadLearningRateTests(unittest.TestCase):
    def _build(self, train_head: bool):
        from unittest.mock import patch

        from cs604.models.build import build_from_config

        cfg = {**CompactCheckpointTests.CFG, "train_head": train_head}
        with patch("cs604.models.build.timm.create_model", return_value=TinyViT(0)):
            return build_from_config(cfg)

    def test_without_head_lr_one_group_holds_every_trainable_parameter(self) -> None:
        from cs604.models.build import optimizer_param_groups

        model = self._build(train_head=True)
        groups = optimizer_param_groups(model, 5e-4)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["lr"], 5e-4)
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.assertEqual([id(p) for p in groups[0]["params"]], [id(p) for p in trainable])
        optimizer = torch.optim.AdamW(groups, lr=5e-4)
        self.assertEqual([g["lr"] for g in optimizer.param_groups], [5e-4])

    def test_head_lr_puts_weight_and_temperature_in_their_own_group(self) -> None:
        from cs604.models.build import optimizer_param_groups

        model = self._build(train_head=True)
        backbone, head = optimizer_param_groups(model, 5e-4, head_lr=2e-3)
        self.assertEqual((backbone["lr"], head["lr"]), (5e-4, 2e-3))
        head_ids = {id(model.head.weight), id(model.head.log_scale)}
        self.assertEqual({id(p) for p in head["params"]}, head_ids)
        self.assertTrue(backbone["params"])
        self.assertTrue(all(id(p) not in head_ids for p in backbone["params"]))
        self.assertEqual(
            len(backbone["params"]) + len(head["params"]),
            sum(1 for p in model.parameters() if p.requires_grad),
        )

    def test_scheduler_scales_both_groups_and_keeps_their_ratio(self) -> None:
        from cs604.models.build import optimizer_param_groups

        model = self._build(train_head=True)
        optimizer = torch.optim.AdamW(optimizer_param_groups(model, 5e-4, 2e-3), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5)
        scheduler.step()
        backbone_lr, head_lr = (group["lr"] for group in optimizer.param_groups)
        self.assertAlmostEqual(backbone_lr, 2.5e-4)
        self.assertAlmostEqual(head_lr, 1e-3)

    def test_head_lr_with_a_frozen_head_is_an_error(self) -> None:
        from cs604.models.build import optimizer_param_groups

        with self.assertRaisesRegex(ValueError, "train_head"):
            optimizer_param_groups(self._build(train_head=False), 5e-4, head_lr=2e-3)
        with self.assertRaisesRegex(ValueError, "positive"):
            optimizer_param_groups(self._build(train_head=True), 5e-4, head_lr=0.0)


class ImageSizeTests(unittest.TestCase):
    @staticmethod
    def _offline(name, _create=__import__("timm").create_model, **kwargs):
        kwargs["pretrained"] = False  # architecture only; no download
        return _create(name, **kwargs)

    def test_image_size_reaches_timm_only_when_set(self) -> None:
        from unittest.mock import patch

        from cs604.models.build import build_from_config

        cfg = {**CompactCheckpointTests.CFG, "image_size": 224}
        with patch("cs604.models.build.timm.create_model",
                   side_effect=lambda *args, **kwargs: TinyViT(0)) as create:
            build_from_config(cfg)
            self.assertEqual(create.call_args.kwargs["img_size"], 224)
            build_from_config(CompactCheckpointTests.CFG)
            self.assertNotIn("img_size", create.call_args.kwargs)

    def test_dinov2_at_224_gets_a_224_transform(self) -> None:
        from unittest.mock import patch

        from cs604.models.build import build_from_config, model_data_config

        cfg = {"timm_checkpoint": "vit_base_patch14_dinov2.lvd142m", "num_classes": 10,
               "mode": "lora", "head_type": "cosine", "train_head": True, "image_size": 224}
        with patch("cs604.models.build.timm.create_model", side_effect=self._offline):
            model = build_from_config(cfg)
        self.assertEqual(tuple(model.pos_embed.shape), (1, 257, 768))  # 16x16 patches + CLS
        config = model_data_config(model)
        self.assertEqual(config["input_size"], (3, 224, 224))
        self.assertEqual(config["mean"], (0.485, 0.456, 0.406))  # its own preprocessing
        with torch.inference_mode():
            self.assertEqual(tuple(model(torch.zeros(1, 3, 224, 224)).shape), (1, 10))

    def test_vit_b16_data_config_is_unchanged(self) -> None:
        from timm.data import resolve_model_data_config

        from cs604.models.build import model_data_config

        model = self._offline("vit_base_patch16_224.augreg_in21k", num_classes=10)
        self.assertEqual(model_data_config(model), resolve_model_data_config(model))


class TrainCropScaleTests(unittest.TestCase):
    def test_unset_keeps_timm_default(self) -> None:
        from timm.data import create_transform

        from cs604.engine.loop import _transform_crop_scale, train_crop_scale

        self.assertIsNone(train_crop_scale({}))
        self.assertIsNone(train_crop_scale({"train_crop_scale": None}))
        default = create_transform(input_size=(3, 224, 224), is_training=True)
        self.assertEqual(_transform_crop_scale(default), [0.08, 1.0])

    def test_configured_scale_reaches_the_train_transform(self) -> None:
        from timm.data import create_transform

        from cs604.engine.loop import _transform_crop_scale, train_crop_scale

        scale = train_crop_scale({"train_crop_scale": [0.35, 1.0]})
        self.assertEqual(scale, (0.35, 1.0))
        transform = create_transform(input_size=(3, 224, 224), is_training=True, scale=scale)
        self.assertEqual(_transform_crop_scale(transform), [0.35, 1.0])
        # The eval transform has no random crop, so nothing to record there.
        evaluation = create_transform(input_size=(3, 224, 224), is_training=False)
        self.assertIsNone(_transform_crop_scale(evaluation))

    def test_invalid_scales_are_rejected(self) -> None:
        from cs604.engine.loop import train_crop_scale

        for bad in (0.5, [0.5], [0.0, 1.0], [0.6, 0.4], [0.5, 1.2], "0.3,1"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                train_crop_scale({"train_crop_scale": bad})


class NeighbourSmoothingTests(unittest.TestCase):
    LOGITS = torch.randn(5, 12, generator=torch.Generator().manual_seed(7))
    LABELS = torch.tensor([0, 1, 5, 10, 11])  # includes both ends of the id range

    def test_share_zero_is_plain_label_smoothing(self) -> None:
        import torch.nn.functional as F

        from cs604.engine.loop import training_loss

        torch.testing.assert_close(
            training_loss(None, self.LOGITS, self.LABELS, 0.1, neighbours=(2, 0.0)),
            F.cross_entropy(self.LOGITS, self.LABELS, label_smoothing=0.1),
        )

    def test_matches_an_explicit_soft_target(self) -> None:
        from cs604.engine.loop import training_loss

        eps, share, window, classes = 0.1, 0.6, 2, 12
        target = torch.zeros(5, classes)
        for row, label in enumerate(self.LABELS.tolist()):
            near = [c for c in range(label - window, label + window + 1) if c != label and 0 <= c < classes]
            target[row] += eps * (1 - share) / classes
            target[row, near] += eps * share / len(near)
            target[row, label] += 1 - eps
        expected = -(target * self.LOGITS.log_softmax(1)).sum(1).mean()
        torch.testing.assert_close(
            training_loss(None, self.LOGITS, self.LABELS, eps, neighbours=(window, share)), expected
        )

    def test_config_parsing(self) -> None:
        from cs604.engine.loop import neighbour_smoothing

        self.assertIsNone(neighbour_smoothing({"label_smoothing": 0.1}))
        self.assertEqual(
            neighbour_smoothing({"label_smoothing": 0.1, "neighbour_smoothing_window": 5}), (5, 1.0)
        )
        for bad in ({"label_smoothing": 0.0, "neighbour_smoothing_window": 5},
                    {"label_smoothing": 0.1, "neighbour_smoothing_window": 0},
                    {"label_smoothing": 0.1, "neighbour_smoothing_window": 5, "neighbour_smoothing_share": 1.5}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                neighbour_smoothing(bad)


class ProgressiveResolutionTests(unittest.TestCase):
    def test_unset_is_off_and_valid_keys_parse(self) -> None:
        from cs604.engine.loop import progressive_resolution

        self.assertIsNone(progressive_resolution({"image_size": 384}))
        self.assertEqual(
            progressive_resolution({"image_size": 384, "train_low_image_size": 224}), (224, 0.75)
        )
        cfg = {"image_size": 384, "train_low_image_size": 224, "train_low_fraction": 0.5}
        self.assertEqual(progressive_resolution(cfg), (224, 0.5))

    def test_invalid_settings_are_rejected(self) -> None:
        from cs604.engine.loop import progressive_resolution

        for bad in (
            {"image_size": 224, "train_low_image_size": 224},
            {"image_size": 384, "train_low_image_size": 200},
            {"image_size": 384, "train_low_image_size": 224, "train_low_fraction": 1.0},
            {"image_size": 384, "train_low_image_size": 224, "mode": "lora"},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                progressive_resolution(bad)

    def test_dynamic_model_takes_both_sizes_and_low_transform_crops_smaller(self) -> None:
        from unittest.mock import patch

        from cs604.engine.loop import build_transforms
        from cs604.models.build import build_from_config

        cfg = {"timm_checkpoint": "vit_base_patch16_224.augreg2_in21k_ft_in1k", "num_classes": 10,
               "mode": "full", "head_type": "cosine", "image_size": 384,
               "train_low_image_size": 224}
        with patch("cs604.models.build.timm.create_model", side_effect=ImageSizeTests._offline):
            model = build_from_config(cfg)
        self.assertEqual(tuple(model.pos_embed.shape), (1, 577, 768))  # stored at 384
        with torch.inference_mode():
            for size in (224, 384):
                self.assertEqual(tuple(model(torch.zeros(1, 3, size, size)).shape), (1, 10))
        from PIL import Image

        image = Image.new("RGB", (500, 400))
        full, evaluation = build_transforms(cfg, model)
        low, _ = build_transforms(cfg, model, train_image_size=224)
        self.assertEqual(tuple(full(image).shape), (3, 384, 384))
        self.assertEqual(tuple(low(image).shape), (3, 224, 224))
        self.assertEqual(tuple(evaluation(image).shape), (3, 384, 384))


class PrototypeTests(unittest.TestCase):
    def test_prototypes_replace_classifier_weights(self) -> None:
        layer = CosineClassifier(4, 3)
        prototypes = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0]]
        )
        apply_class_prototypes(type("Model", (), {"get_classifier": lambda self: layer})(), prototypes)
        torch.testing.assert_close(layer.weight[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
        torch.testing.assert_close(layer.weight[1], torch.tensor([0.0, 1.0, 0.0, 0.0]))


class SubmissionTests(unittest.TestCase):
    def test_submission_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "val.txt"
            submission = root / "val_team_01.txt"
            index.write_text("val/a.jpg\nval/b.jpg\n", encoding="utf-8")
            submission.write_text("val/a.jpg 0\nval/b.jpg 9999\n", encoding="utf-8")
            validate_submission(submission, index)
            submission.write_text("val/b.jpg 0\nval/a.jpg 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected"):
                validate_submission(submission, index)


class FullFineTuneOptionTests(unittest.TestCase):
    """layer_decay, no_decay_norm_bias, drop_path_rate and head_init_file."""

    CFG = {"timm_checkpoint": "vit_tiny_patch16_224", "num_classes": 5, "mode": "full",
           "head_type": "cosine", "train_head": True}

    def _build(self, **overrides):
        from unittest.mock import patch

        from cs604.models.build import build_from_config

        with patch("cs604.models.build.timm.create_model", side_effect=ImageSizeTests._offline):
            return build_from_config({**self.CFG, **overrides})

    def _rates(self, groups) -> dict[int, tuple[float, float | None]]:
        rates = {}
        for group in groups:
            for param in group["params"]:
                self.assertNotIn(id(param), rates)  # every parameter in exactly one group
                rates[id(param)] = (group["lr"], group.get("weight_decay"))
        return rates

    def test_layer_decay_scales_rates_by_depth(self) -> None:
        from cs604.models.build import optimizer_param_groups

        model = self._build()
        groups = optimizer_param_groups(model, 1e-3, head_lr=1e-2, layer_decay=0.5)
        rates = self._rates(groups)
        self.assertEqual(len(rates), sum(1 for p in model.parameters() if p.requires_grad))
        top = len(model.blocks) + 1  # 13 for 12 blocks
        expected = {
            "pos_embed": 1e-3 * 0.5 ** top, "cls_token": 1e-3 * 0.5 ** top,
            "patch_embed.proj.weight": 1e-3 * 0.5 ** top,
            "blocks.0.attn.qkv.weight": 1e-3 * 0.5 ** 12,
            "blocks.11.mlp.fc2.weight": 1e-3 * 0.5,
            "norm.weight": 1e-3, "head.weight": 1e-2, "head.log_scale": 1e-2,
        }
        params = dict(model.named_parameters())
        for name, rate in expected.items():
            self.assertAlmostEqual(rates[id(params[name])][0], rate, msg=name)
        # The metrics log reads param_groups[0]: the top backbone rate.
        self.assertAlmostEqual(groups[0]["lr"], 1e-3)
        self.assertTrue(groups[-1]["name"].startswith("head"))

    def test_no_decay_covers_norms_biases_tokens_and_temperature(self) -> None:
        from cs604.models.build import optimizer_param_groups

        model = self._build()
        rates = self._rates(optimizer_param_groups(model, 1e-3, no_decay_norm_bias=True))
        for name, param in model.named_parameters():
            no_decay = param.ndim <= 1 or name in ("pos_embed", "cls_token")
            self.assertEqual(rates[id(param)], (1e-3, 0.0 if no_decay else None), msg=name)

    def test_defaults_keep_one_group(self) -> None:
        from cs604.models.build import optimizer_param_groups

        self.assertEqual(len(optimizer_param_groups(self._build(), 1e-3)), 1)
        with self.assertRaisesRegex(ValueError, "layer_decay"):
            optimizer_param_groups(self._build(), 1e-3, layer_decay=1.5)

    def test_drop_path_rate_reaches_timm_only_when_set(self) -> None:
        self.assertEqual(self._build().blocks[-1].drop_path1.__class__.__name__, "Identity")
        model = self._build(drop_path_rate=0.1)
        self.assertAlmostEqual(model.blocks[-1].drop_path1.drop_prob, 0.1)

    def test_head_init_round_trips_a_cached_feature_head(self) -> None:
        from cs604.models.build import build_feature_head, load_head_init

        feature_head = build_feature_head(self.CFG, 192)
        with torch.no_grad():
            feature_head.head.weight.normal_()
            feature_head.head.log_scale.fill_(3.0)
        model = self._build()
        load_head_init(model, {"head": feature_head.state_dict(), "timm_checkpoint": "vit_tiny_patch16_224"})
        torch.testing.assert_close(model.head.weight, feature_head.head.weight)
        torch.testing.assert_close(model.head.log_scale, feature_head.head.log_scale)
        features = torch.randn(2, 192)
        torch.testing.assert_close(model.head(features), feature_head(features))

    def test_only_one_initializer_is_allowed(self) -> None:
        from cs604.engine.baseline import validate_baseline_setup

        cfg = {"evaluate_before_training": True, "validate": True,
               "prototype_file": "p.pt", "head_init_file": "h.pt"}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Choose one"):
                validate_baseline_setup(cfg, Path(directory))


if __name__ == "__main__":
    unittest.main()
