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
    build_from_config,
    checkpoint_model_state,
    load_model_state,
    merge_ssf,
    optimizer_param_groups,
    parameter_report,
    ssf_modules,
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


def _randomize_ssf(model, seed=1):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for ssf in ssf_modules(model):
            ssf.scale.copy_(1 + 0.3 * torch.randn(ssf.scale.shape, generator=generator))
            ssf.shift.copy_(0.3 * torch.randn(ssf.shift.shape, generator=generator))


class SSFTests(unittest.TestCase):
    def setUp(self) -> None:
        self.images = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(2))

    def test_identity_init_starts_identical_to_the_frozen_model(self) -> None:
        reference = _build(mode="head").eval()
        model = _build(mode="ssf").eval()
        torch.testing.assert_close(model(self.images), reference(self.images))

    def test_normal_init_moves_the_outputs(self) -> None:
        reference = _build(mode="head").eval()
        model = _build(mode="ssf", ssf_init="normal").eval()
        self.assertFalse(torch.allclose(model(self.images), reference(self.images)))
        with self.assertRaisesRegex(ValueError, "ssf_init"):
            _build(mode="ssf", ssf_init="zeros")

    def test_only_ssf_and_head_train_and_every_target_is_wrapped(self) -> None:
        model = _build(mode="ssf")
        names = [name for name, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(all(".ssf." in name or name.startswith("head.") for name in names))
        # patch embedding + 6 per block + final norm
        self.assertEqual(len(ssf_modules(model)), 1 + 6 * len(model.blocks) + 1)
        self.assertIn("blocks.0.attn.qkv.ssf.scale", names)
        self.assertIn("patch_embed.proj.ssf.shift", names)
        report = parameter_report(model)
        self.assertEqual(report["pretrained_backbone_params"],
                         parameter_report(_build(mode="head"))["pretrained_backbone_params"])

    def test_scale_and_shift_get_gradients(self) -> None:
        model = _build(mode="ssf").train()
        model(self.images).sum().backward()
        ssf = model.blocks[0].mlp.fc1.ssf
        self.assertGreater(ssf.scale.grad.abs().sum().item(), 0)
        self.assertGreater(ssf.shift.grad.abs().sum().item(), 0)

    def test_keeps_bf16_activations_under_autocast(self) -> None:
        model = _build(mode="ssf")
        seen = []
        model.blocks[0].mlp.fc1.register_forward_hook(lambda _m, _a, out: seen.append(out.dtype))
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):  # CPU: no bf16 backward
            model(self.images)
        self.assertEqual(seen, [torch.bfloat16])
        self.assertEqual(model.blocks[0].mlp.fc1.ssf.scale.dtype, torch.float32)

    def test_merge_folds_into_the_weights_with_the_same_outputs(self) -> None:
        model = _build(mode="ssf").eval()
        _randomize_ssf(model)
        expected = model(self.images)
        self.assertEqual(merge_ssf(model), 1 + 6 * len(model.blocks) + 1)
        self.assertFalse(ssf_modules(model))
        self.assertFalse(any(".ssf." in name for name, _ in model.named_parameters()))
        torch.testing.assert_close(model(self.images), expected, rtol=1e-4, atol=1e-4)

    def test_compact_checkpoint_round_trips(self) -> None:
        model = _build(mode="ssf").eval()
        _randomize_ssf(model)
        saved = checkpoint_model_state(model, compact=True)
        self.assertTrue(all(".ssf." in key or key.startswith("head.") for key in saved["model"]))
        restored = _build(mode="ssf").eval()
        load_model_state(restored, saved)
        torch.testing.assert_close(restored(self.images), model(self.images))

    def test_no_weight_decay_on_scale_and_shift(self) -> None:
        model = _build(mode="ssf")
        groups = optimizer_param_groups(model, 3e-3, head_lr=4e-3, no_decay_norm_bias=True)
        backbone = [g for g in groups if not g["name"].startswith("head")]
        self.assertEqual([g["name"] for g in backbone], ["layer0_no_decay"])
        self.assertEqual(backbone[0]["lr"], 3e-3)
        self.assertEqual(backbone[0]["weight_decay"], 0.0)

    def test_vit_b_count_is_far_under_5pct(self) -> None:
        recipe = {**TINY, "timm_checkpoint": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
                  "num_classes": 10000, "image_size": 336, "mode": "ssf"}
        with patch("cs604.models.build.timm.create_model", side_effect=_offline):
            report = parameter_report(build_from_config(recipe))
        # 2 x (768 patch + 12 x (768 + 2304 + 768 + 768 + 3072 + 768) + 768 final norm)
        self.assertEqual(report["backbone_trainable_params"], 205_824)
        self.assertLess(report["backbone_trainable_pct"], 0.25)
        check_backbone_budget(recipe, report)  # no error
        with self.assertRaisesRegex(ValueError, "under 5%"):
            check_backbone_budget(recipe, {**report, "backbone_trainable_pct": 5.0})


class AdaptFormerSSFTests(unittest.TestCase):
    OPTIONS = {"mode": "adaptformer", "adapter_dim": 8, "adapter_input": "norm",
               "adapter_scale_trainable": True, "adapter_ssf": True}

    def setUp(self) -> None:
        self.images = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(2))

    def test_starts_identical_and_only_adapters_ssf_head_train(self) -> None:
        reference = _build(mode="head").eval()
        model = _build(**self.OPTIONS).eval()
        torch.testing.assert_close(model(self.images), reference(self.images))
        names = [name for name, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(all(".ssf." in n or ".adaptformer." in n or n.startswith("head.")
                            for n in names))
        self.assertTrue(any(".ssf." in n for n in names))
        self.assertTrue(any(".adaptformer." in n for n in names))

    def test_adapter_reads_the_scaled_norm2_output(self) -> None:
        model = _build(**self.OPTIONS).eval()
        seen = []
        model.blocks[0].adaptformer.register_forward_pre_hook(lambda _m, args: seen.append(args[0]))
        with torch.no_grad():
            model.blocks[0].norm2.ssf.scale.fill_(2.0)
            model(self.images)
        norm2 = []
        model.blocks[0].norm2.register_forward_hook(lambda _m, _a, out: norm2.append(out))
        with torch.no_grad():
            model(self.images)
        torch.testing.assert_close(seen[0], norm2[0])

    def test_compact_checkpoint_round_trips(self) -> None:
        model = _build(**self.OPTIONS).eval()
        _randomize_ssf(model)
        saved = checkpoint_model_state(model, compact=True)
        restored = _build(**self.OPTIONS).eval()
        load_model_state(restored, saved)
        torch.testing.assert_close(restored(self.images), model(self.images))
        with self.assertRaisesRegex(ValueError, "Compact checkpoint keys"):
            load_model_state(_build(**{**self.OPTIONS, "adapter_ssf": False}), saved)

    def test_needs_adaptformer(self) -> None:
        with self.assertRaisesRegex(ValueError, "adapter_ssf needs mode: adaptformer"):
            _build(mode="lora", adapter_ssf=True)

    def test_vit_b_recipe_with_ssf_is_under_5pct(self) -> None:
        recipe = {**TINY, "timm_checkpoint": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
                  "num_classes": 10000, "image_size": 336, "mode": "adaptformer",
                  "adapter_dim": 192, "adapter_input": "norm", "adapter_scale_trainable": True}
        with patch("cs604.models.build.timm.create_model", side_effect=_offline):
            plain = parameter_report(build_from_config(recipe))
            both = parameter_report(build_from_config({**recipe, "adapter_ssf": True}))
        self.assertEqual(plain["backbone_trainable_params"], 3_550_476)
        self.assertEqual(both["backbone_trainable_params"], 3_550_476 + 205_824)
        self.assertLess(both["backbone_trainable_pct"], 4.4)
        check_backbone_budget(recipe, both)  # no error


if __name__ == "__main__":
    unittest.main()
