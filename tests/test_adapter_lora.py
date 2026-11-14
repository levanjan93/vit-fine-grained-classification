from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import timm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cs604.engine.loop import check_backbone_budget  # noqa: E402
from cs604.models.build import (  # noqa: E402
    LoRALinear,
    build_from_config,
    checkpoint_model_state,
    load_model_state,
    parameter_report,
)

_create = timm.create_model


def _offline(name, **kwargs):
    kwargs["pretrained"] = False  # architecture only; no download
    torch.manual_seed(0)  # the same "pretrained" weights every build
    return _create(name, **kwargs)


TINY = {
    "timm_checkpoint": "vit_tiny_patch16_224", "num_classes": 7, "image_size": 32,
    "head_type": "cosine", "cosine_head_fp32": True, "train_head": True,
}


def _build(**overrides):
    with patch("cs604.models.build.timm.create_model", side_effect=_offline):
        return build_from_config({**TINY, **overrides})


class AdaptFormerLoRATests(unittest.TestCase):
    OPTIONS = {"mode": "adaptformer", "adapter_dim": 8, "adapter_input": "norm",
               "adapter_scale_trainable": True, "adapter_train_bias": True,
               "adapter_train_norm": True, "adapter_lora": True, "lora_rank": 4}

    def setUp(self) -> None:
        self.images = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(2))

    def test_starts_identical_and_wraps_every_qkv(self) -> None:
        reference = _build(mode="head").eval()
        model = _build(**self.OPTIONS).eval()
        torch.testing.assert_close(model(self.images), reference(self.images))
        self.assertTrue(all(isinstance(block.attn.qkv, LoRALinear) for block in model.blocks))

    def test_lora_adapters_biases_and_norms_train(self) -> None:
        model = _build(**self.OPTIONS)
        names = {name for name, p in model.named_parameters() if p.requires_grad}
        self.assertIn("blocks.0.attn.qkv.lora_a", names)
        self.assertIn("blocks.0.attn.qkv.lora_b", names)
        self.assertIn("blocks.0.attn.qkv.base.bias", names)  # still trained after wrapping
        self.assertNotIn("blocks.0.attn.qkv.base.weight", names)
        self.assertTrue(any(".adaptformer." in n for n in names))
        self.assertIn("blocks.0.norm1.weight", names)

    def test_lora_gets_gradients_after_b_moves(self) -> None:
        model = _build(**self.OPTIONS).train()
        with torch.no_grad():
            model.blocks[0].attn.qkv.lora_b.fill_(0.01)
        model(self.images).sum().backward()
        qkv = model.blocks[0].attn.qkv
        self.assertGreater(qkv.lora_a.grad.abs().sum().item(), 0)
        self.assertGreater(qkv.lora_b.grad.abs().sum().item(), 0)

    def test_compact_checkpoint_round_trips(self) -> None:
        model = _build(**self.OPTIONS).eval()
        with torch.no_grad():
            for block in model.blocks:
                block.attn.qkv.lora_b.normal_(0, 0.1)
        saved = checkpoint_model_state(model, compact=True)
        restored = _build(**self.OPTIONS).eval()
        load_model_state(restored, saved)
        torch.testing.assert_close(restored(self.images), model(self.images))
        with self.assertRaisesRegex(ValueError, "Compact checkpoint keys"):
            load_model_state(_build(**{**self.OPTIONS, "adapter_lora": False}), saved)

    def test_needs_adaptformer(self) -> None:
        with self.assertRaisesRegex(ValueError, "adapter_lora needs mode: adaptformer"):
            _build(mode="ssf", adapter_lora=True)

    def test_vit_b_recipe_with_rank_8_is_under_5pct(self) -> None:
        recipe = {**TINY, "timm_checkpoint": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
                  "num_classes": 10000, "image_size": 336, **self.OPTIONS,
                  "adapter_dim": 192, "lora_rank": 8}
        with patch("cs604.models.build.timm.create_model", side_effect=_offline):
            report = parameter_report(build_from_config(recipe))
        # 3,672,588 (adapters + biases/norms) + 12 x 8 x (768 + 2304) LoRA
        self.assertEqual(report["backbone_trainable_params"], 3_672_588 + 294_912)
        self.assertLess(report["backbone_trainable_pct"], 4.7)
        check_backbone_budget(recipe, report)  # no error


class EqualParameterControlTests(unittest.TestCase):
    """The 4.96% controls for AdaptFormer + LoRA r16: AF width 224 alone, and LoRA alone
    (``adapter_dim: 0``) on qkv/proj/fc1/fc2, both with biases/norms."""

    RECIPE = {**TINY, "timm_checkpoint": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
              "num_classes": 10000, "image_size": 336, "mode": "adaptformer",
              "adapter_input": "norm", "adapter_scale_trainable": True,
              "adapter_train_bias": True, "adapter_train_norm": True}
    LORA_ONLY = {"adapter_dim": 0, "adapter_lora": True,
                 "lora_targets": ["qkv", "proj", "fc1", "fc2"]}

    def setUp(self) -> None:
        self.images = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(2))

    def _vit_b_report(self, **overrides) -> dict:
        recipe = {**self.RECIPE, **overrides}
        with patch("cs604.models.build.timm.create_model", side_effect=_offline):
            report = parameter_report(build_from_config(recipe))
        check_backbone_budget(recipe, report)  # under 5%: no error
        return report

    def test_vit_b_counts_match_af_plus_lora_r16(self) -> None:
        af_lora = self._vit_b_report(adapter_dim=192, adapter_lora=True, lora_rank=16)
        af_224 = self._vit_b_report(adapter_dim=224)
        lora_28 = self._vit_b_report(**self.LORA_ONLY, lora_rank=28)
        self.assertEqual(af_lora["backbone_trainable_params"], 4_262_412)
        # 12 x (2 x 768 x 224 + 224 + 768 + 1 scale) + 122,112 biases/norms
        self.assertEqual(af_224["backbone_trainable_params"], 4_262_796)
        # 28 x 12 x (768 + 2304 + 768 + 768 + 768 + 3072 + 3072 + 768) + 122,112
        self.assertEqual(lora_28["backbone_trainable_params"], 4_250_880)

    def test_lora_only_starts_identical_and_wraps_every_target(self) -> None:
        reference = _build(mode="head").eval()
        model = _build(mode="adaptformer", adapter_train_bias=True, adapter_train_norm=True,
                       lora_rank=4, **self.LORA_ONLY).eval()
        torch.testing.assert_close(model(self.images), reference(self.images))
        for block in model.blocks:
            self.assertFalse(hasattr(block, "adaptformer"))
            for linear in (block.attn.qkv, block.attn.proj, block.mlp.fc1, block.mlp.fc2):
                self.assertIsInstance(linear, LoRALinear)
        names = {name for name, p in model.named_parameters() if p.requires_grad}
        self.assertIn("blocks.0.mlp.fc2.lora_b", names)
        self.assertIn("blocks.0.mlp.fc2.base.bias", names)
        self.assertNotIn("blocks.0.mlp.fc2.base.weight", names)

    def test_default_targets_and_bad_names(self) -> None:
        model = _build(mode="lora", lora_rank=4)
        self.assertIsInstance(model.blocks[0].attn.qkv, LoRALinear)
        self.assertNotIsInstance(model.blocks[0].attn.proj, LoRALinear)
        with self.assertRaisesRegex(ValueError, "lora_targets"):
            _build(mode="lora", lora_targets=["qkv", "fc3"])

    def test_vit_b_spread_controls_match_their_references(self) -> None:
        # Spread vs concentrated: the same budget in blocks 4-11 only (0-based).
        conc = list(range(4, 12))
        lora_28 = self._vit_b_report(**self.LORA_ONLY, lora_rank=28)
        lora_42_b4 = self._vit_b_report(**self.LORA_ONLY, lora_rank=42, lora_blocks=conc)
        self.assertEqual(lora_42_b4["backbone_trainable_params"],
                         lora_28["backbone_trainable_params"])  # 42 x 8 = 28 x 12
        af_192 = self._vit_b_report(adapter_dim=192)
        af_288_b4 = self._vit_b_report(adapter_dim=[0] * 4 + [288] * 8)
        self.assertEqual(af_192["backbone_trainable_params"], 3_672_588)
        # 8 x (2 x 768 x 288 + 288 + 768 + 1) + 122,112: 3,076 fewer (bias/scale terms only)
        self.assertEqual(af_288_b4["backbone_trainable_params"], 3_669_512)

    def test_lora_blocks_wraps_only_chosen_blocks(self) -> None:
        reference = _build(mode="head").eval()
        model = _build(mode="adaptformer", adapter_train_bias=True, adapter_train_norm=True,
                       lora_rank=4, lora_blocks=[2, 3], **self.LORA_ONLY).eval()
        torch.testing.assert_close(model(self.images), reference(self.images))
        for index, block in enumerate(model.blocks):
            self.assertEqual(isinstance(block.mlp.fc1, LoRALinear), index in (2, 3))
        names = {name for name, p in model.named_parameters() if p.requires_grad}
        self.assertIn("blocks.0.mlp.fc2.bias", names)  # biases still train everywhere
        lora_mode = _build(mode="lora", lora_rank=4, lora_blocks=[0])
        self.assertIsInstance(lora_mode.blocks[0].attn.qkv, LoRALinear)
        self.assertNotIsInstance(lora_mode.blocks[1].attn.qkv, LoRALinear)
        with self.assertRaisesRegex(ValueError, "below"):
            _build(mode="lora", lora_blocks=[99])
        with self.assertRaisesRegex(ValueError, "distinct"):
            _build(mode="lora", lora_blocks=[1, 1])
        with self.assertRaisesRegex(ValueError, "lora_blocks needs"):
            _build(mode="adaptformer", adapter_dim=4, lora_blocks=[0])

    def test_pca_init_on_partial_widths(self) -> None:
        from cs604.models.build import calibrate_adapters_pca
        model = _build(mode="adaptformer", adapter_dim=[0] * 6 + [8] * 6,
                       adapter_input="norm").eval()
        before = model(self.images)
        calibrate_adapters_pca(model, [self.images], torch.device("cpu"))
        torch.testing.assert_close(model(self.images), before)  # up stays zero
        self.assertFalse(hasattr(model.blocks[0], "adaptformer"))

    def test_pca_init_needs_adapters(self) -> None:
        from cs604.models.build import calibrate_adapters_pca
        model = _build(mode="adaptformer", adapter_train_bias=True, **self.LORA_ONLY)
        with self.assertRaisesRegex(ValueError, "adapter_dim > 0"):
            calibrate_adapters_pca(model, [self.images], torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
