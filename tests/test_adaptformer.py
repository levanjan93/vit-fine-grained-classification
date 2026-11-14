from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import timm
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cs604.engine.loop import check_backbone_budget, training_loss  # noqa: E402
from cs604.models.build import (  # noqa: E402
    adaptformer_adapters,
    build_from_config,
    calibrate_adapters_pca,
    checkpoint_model_state,
    load_model_state,
    optimizer_param_groups,
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


def _randomize_up(model, seed=1):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for adapter in adaptformer_adapters(model):
            adapter.up.weight.copy_(torch.randn(adapter.up.weight.shape, generator=generator))


class AdaptFormerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.images = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(2))

    def test_starts_identical_to_the_frozen_model(self) -> None:
        reference = _build(mode="head").eval()
        for adapter_input in ("residual", "norm"):
            model = _build(mode="adaptformer", adapter_dim=8, adapter_input=adapter_input).eval()
            torch.testing.assert_close(model(self.images), reference(self.images))

    def test_only_adapters_and_head_train(self) -> None:
        model = _build(mode="adaptformer", adapter_dim=8)
        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(trainable)
        self.assertTrue(all(".adaptformer." in name or name.startswith("head.") for name in trainable))
        self.assertEqual(len(adaptformer_adapters(model)), len(model.blocks))
        report = parameter_report(model)
        dim, depth = model.embed_dim, len(model.blocks)
        self.assertEqual(report["backbone_trainable_params"], depth * (2 * dim * 8 + 8 + dim))
        reference = parameter_report(_build(mode="head"))
        self.assertEqual(report["pretrained_backbone_params"],
                         reference["total_params"] - reference["classifier_params"])

    def test_vit_b_width_64_at_336_is_the_paper_size(self) -> None:
        model = _build(timm_checkpoint="vit_base_patch16_224", num_classes=10, image_size=336,
                       mode="adaptformer", adapter_dim=64)
        report = parameter_report(model)
        self.assertEqual(report["backbone_trainable_params"], 12 * (1537 * 64 + 768))  # 1,189,632
        self.assertAlmostEqual(report["backbone_trainable_pct"], 1.38, places=2)

    def test_input_variants_differ_once_trained_and_residual_sees_the_residual(self) -> None:
        outputs = {}
        for adapter_input in ("residual", "norm"):
            model = _build(mode="adaptformer", adapter_dim=8, adapter_input=adapter_input).eval()
            _randomize_up(model)
            seen = []
            adapter = adaptformer_adapters(model)[0]
            adapter.register_forward_pre_hook(lambda _m, args: seen.append(args[0]))
            block_inputs = []
            model.blocks[0].norm2.register_forward_pre_hook(lambda _m, args: block_inputs.append(args[0]))
            outputs[adapter_input] = model(self.images)
            if adapter_input == "residual":
                torch.testing.assert_close(seen[0], block_inputs[0])
            else:
                torch.testing.assert_close(seen[0], model.blocks[0].norm2(block_inputs[0]))
        self.assertFalse(torch.allclose(outputs["residual"], outputs["norm"]))

    def test_down_learns_after_up_moves(self) -> None:
        model = _build(mode="adaptformer", adapter_dim=8)
        loss = F.cross_entropy(model(self.images), torch.tensor([0, 1]))
        loss.backward()
        adapter = adaptformer_adapters(model)[0]
        self.assertEqual(float(adapter.down.weight.grad.abs().sum()), 0.0)  # up is zero
        self.assertGreater(float(adapter.up.weight.grad.abs().sum()), 0.0)
        model.zero_grad()
        _randomize_up(model)
        F.cross_entropy(model(self.images), torch.tensor([0, 1])).backward()
        self.assertGreater(float(adapter.down.weight.grad.abs().sum()), 0.0)

    def test_compact_checkpoint_round_trips(self) -> None:
        model = _build(mode="adaptformer", adapter_dim=8, adapter_scale_trainable=True).eval()
        _randomize_up(model)
        saved = checkpoint_model_state(model, compact=True)
        self.assertTrue(all(".adaptformer." in key or key.startswith("head.") for key in saved["model"]))
        self.assertIn("blocks.0.adaptformer.scale", saved["model"])
        restored = _build(mode="adaptformer", adapter_dim=8, adapter_scale_trainable=True).eval()
        load_model_state(restored, saved)
        torch.testing.assert_close(restored(self.images), model(self.images))

    def test_adapters_get_lr_and_head_gets_head_lr(self) -> None:
        model = _build(mode="adaptformer", adapter_dim=8)
        groups = optimizer_param_groups(model, 1e-3, head_lr=3e-3, no_decay_norm_bias=True)
        adapters = {id(p) for a in adaptformer_adapters(model) for p in a.parameters()}
        for group in groups:
            ids = {id(p) for p in group["params"]}
            self.assertEqual(group["lr"], 1e-3 if ids <= adapters else 3e-3)

    def test_pca_init_whitens_the_input_and_keeps_outputs(self) -> None:
        model = _build(mode="adaptformer", adapter_dim=8).eval()
        before = model(self.images)
        batches = [torch.randn(4, 3, 32, 32, generator=torch.Generator().manual_seed(i)) for i in range(3)]
        result = calibrate_adapters_pca(model, batches, torch.device("cpu"))
        torch.testing.assert_close(model(self.images), before)
        self.assertEqual(len(result["variance_explained"]), len(model.blocks))
        tokens = []
        adapter = adaptformer_adapters(model)[1]
        handle = adapter.register_forward_pre_hook(lambda _m, args: tokens.append(args[0]))
        for images in batches:
            model(images)
        handle.remove()
        hidden = adapter.down(torch.cat(tokens).reshape(-1, model.embed_dim).double().float())
        torch.testing.assert_close(hidden.mean(0), torch.zeros(8), atol=1e-3, rtol=0)
        torch.testing.assert_close(hidden.var(0, unbiased=False), torch.ones(8), atol=1e-2, rtol=0)

    def test_unknown_adapter_input_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "adapter_input"):
            _build(mode="adaptformer", adapter_input="serial")


class PerBlockWidthTests(unittest.TestCase):
    """``adapter_dim`` as a list: one AdaptFormer width per block."""

    GRADED = [4] * 4 + [8] * 4 + [12] * 4  # vit_tiny has 12 blocks

    def test_list_sets_each_block_width_and_starts_identical(self) -> None:
        images = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(3))
        reference = _build(mode="head").eval()
        model = _build(mode="adaptformer", adapter_dim=self.GRADED).eval()
        self.assertEqual([a.down.out_features for a in adaptformer_adapters(model)], self.GRADED)
        torch.testing.assert_close(model(images), reference(images))
        uniform = _build(mode="adaptformer", adapter_dim=8)
        self.assertEqual(parameter_report(model)["backbone_trainable_params"],
                         parameter_report(uniform)["backbone_trainable_params"])

    def test_vit_b_graded_widths_match_the_uniform_192_budget(self) -> None:
        recipe = {"timm_checkpoint": "vit_base_patch16_224", "num_classes": 10, "image_size": 336,
                  "mode": "adaptformer", "adapter_scale_trainable": True}
        graded = parameter_report(_build(**recipe, adapter_dim=[96] * 4 + [192] * 4 + [288] * 4))
        self.assertEqual(graded["backbone_trainable_params"], 3_550_476)

    def test_wrong_number_of_widths_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "12 widths"):
            _build(mode="adaptformer", adapter_dim=[8] * 11)

    def test_pca_init_whitens_each_width(self) -> None:
        model = _build(mode="adaptformer", adapter_dim=self.GRADED).eval()
        batches = [torch.randn(4, 3, 32, 32, generator=torch.Generator().manual_seed(i)) for i in range(3)]
        calibrate_adapters_pca(model, batches, torch.device("cpu"))
        for index in (0, 11):
            tokens = []
            adapter = adaptformer_adapters(model)[index]
            handle = adapter.register_forward_pre_hook(lambda _m, args: tokens.append(args[0]))
            for images in batches:
                model(images)
            handle.remove()
            hidden = adapter.down(torch.cat(tokens).reshape(-1, model.embed_dim))
            width = self.GRADED[index]
            torch.testing.assert_close(hidden.mean(0), torch.zeros(width), atol=1e-3, rtol=0)
            torch.testing.assert_close(hidden.var(0, unbiased=False), torch.ones(width), atol=1e-2, rtol=0)


class BiasNormTests(unittest.TestCase):
    """``adapter_train_bias`` / ``adapter_train_norm``: pretrained biases and LayerNorm weights."""

    def setUp(self) -> None:
        self.images = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(2))

    def test_vit_b_counts_and_the_5pct_check(self) -> None:
        recipe = {"timm_checkpoint": "vit_base_patch16_224", "num_classes": 10, "image_size": 336,
                  "mode": "adaptformer", "adapter_dim": 192, "adapter_scale_trainable": True}
        reports, trainable = {}, {}
        for arm, extra in {"off": {}, "bias": {"adapter_train_bias": True},
                           "both": {"adapter_train_bias": True, "adapter_train_norm": True}}.items():
            model = _build(**recipe, **extra)
            reports[arm] = parameter_report(model)
            trainable[arm] = {name: p.numel() for name, p in model.named_parameters()
                              if p.requires_grad and ".adaptformer." not in name
                              and not name.startswith("head.")}
        self.assertEqual(reports["off"]["backbone_trainable_params"], 3_550_476)
        self.assertEqual(trainable["off"], {})
        norm_names = {name for name in trainable["both"] if ".norm" in name or name.startswith("norm.")}
        linear_bias = sum(n for name, n in trainable["bias"].items() if name not in norm_names)
        norm_bias = sum(n for name, n in trainable["both"].items()
                        if name in norm_names and name.endswith(".bias"))
        norm_weight = sum(n for name, n in trainable["both"].items()
                          if name in norm_names and name.endswith(".weight"))
        self.assertEqual((linear_bias, norm_bias, norm_weight), (83_712, 19_200, 19_200))
        self.assertTrue(all(name.endswith(".bias") for name in trainable["bias"]))
        self.assertEqual(reports["bias"]["backbone_trainable_params"], 3_550_476 + 102_912)
        both = reports["both"]
        self.assertEqual(both["backbone_trainable_params"], 3_672_588)
        # Pretrained parameters: the denominator does not change, the numerator counts them.
        self.assertEqual(both["pretrained_backbone_params"], reports["off"]["pretrained_backbone_params"])
        self.assertAlmostEqual(both["backbone_trainable_pct"], 4.27, places=2)
        check_backbone_budget(recipe, both)  # under 5%: no error
        # Width 226 alone is 4.86%; its biases and norms take it to 5.0004%.
        wide = {**recipe, "adapter_dim": 226}
        check_backbone_budget(wide, parameter_report(_build(**wide)))
        with self.assertRaisesRegex(ValueError, "under 5%"):
            check_backbone_budget(wide, parameter_report(
                _build(**wide, adapter_train_bias=True, adapter_train_norm=True)))

    def test_starts_identical_and_only_bias_norm_adapters_head_train(self) -> None:
        reference = _build(mode="head").eval()
        model = _build(mode="adaptformer", adapter_dim=8, adapter_input="norm",
                       adapter_train_bias=True, adapter_train_norm=True).eval()
        torch.testing.assert_close(model(self.images), reference(self.images))
        for name, p in model.named_parameters():
            expected = (".adaptformer." in name or name.startswith("head.")
                        or name.endswith(".bias") or (".norm" in name or name.startswith("norm.")))
            self.assertEqual(p.requires_grad, expected, name)
        F.cross_entropy(model.train()(self.images), torch.tensor([0, 1])).backward()
        self.assertGreater(float(model.blocks[0].attn.qkv.bias.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.blocks[0].norm1.weight.grad.abs().sum()), 0.0)

    def test_no_weight_decay_and_adapter_lr(self) -> None:
        model = _build(mode="adaptformer", adapter_dim=8, adapter_train_bias=True, adapter_train_norm=True)
        pretrained = {id(p) for name, p in model.named_parameters()
                      if p.requires_grad and ".adaptformer." not in name and not name.startswith("head.")}
        groups = optimizer_param_groups(model, 1e-3, head_lr=4e-3, no_decay_norm_bias=True)
        seen = set()
        for group in groups:
            ids = {id(p) for p in group["params"]} & pretrained
            if ids:
                self.assertEqual(group["lr"], 1e-3)
                self.assertEqual(group.get("weight_decay"), 0.0)
                seen |= ids
        self.assertEqual(seen, pretrained)

    def test_bias_lr_gives_only_the_pretrained_parameters_their_own_rate(self) -> None:
        model = _build(mode="adaptformer", adapter_dim=8, adapter_train_bias=True, adapter_train_norm=True)
        pretrained = {id(p) for name, p in model.named_parameters()
                      if p.requires_grad and ".adaptformer." not in name and not name.startswith("head.")}
        adapters = {id(p) for a in adaptformer_adapters(model) for p in a.parameters()}
        head = {id(p) for p in model.head.parameters()}
        groups = optimizer_param_groups(model, 1e-3, head_lr=4e-3, no_decay_norm_bias=True, bias_lr=3e-3)
        rates = {}
        for group in groups:
            for p in group["params"]:
                rates[id(p)] = group["lr"]
        self.assertEqual(set(rates), pretrained | adapters | head)
        self.assertTrue(all(rates[i] == 3e-3 for i in pretrained))
        self.assertTrue(all(rates[i] == 1e-3 for i in adapters))
        self.assertTrue(all(rates[i] == 4e-3 for i in head))
        self.assertEqual(groups[0]["lr"], 3e-3)
        with self.assertRaisesRegex(ValueError, "bias_lr"):
            optimizer_param_groups(_build(mode="adaptformer", adapter_dim=8), 1e-3, bias_lr=3e-3)

    def test_compact_checkpoint_keeps_trained_biases(self) -> None:
        options = {"mode": "adaptformer", "adapter_dim": 8, "adapter_train_bias": True,
                   "adapter_train_norm": True}
        model = _build(**options).eval()
        _randomize_up(model)
        with torch.no_grad():
            model.blocks[0].mlp.fc1.bias.add_(0.5)
            model.blocks[1].norm2.weight.mul_(1.5)
        saved = checkpoint_model_state(model, compact=True)
        self.assertIn("blocks.0.mlp.fc1.bias", saved["model"])
        self.assertIn("blocks.1.norm2.weight", saved["model"])
        restored = _build(**options).eval()
        load_model_state(restored, saved)
        torch.testing.assert_close(restored(self.images), model(self.images))
        with self.assertRaisesRegex(ValueError, "Compact checkpoint keys"):
            load_model_state(_build(mode="adaptformer", adapter_dim=8), saved)

    def test_needs_adaptformer(self) -> None:
        with self.assertRaisesRegex(ValueError, "adapter_train_bias"):
            _build(mode="lora", adapter_train_bias=True)


class MarginLossTests(unittest.TestCase):
    def test_zero_margin_is_cross_entropy_and_a_margin_raises_the_loss(self) -> None:
        model = _build(mode="head")
        logits = torch.randn(4, 7, generator=torch.Generator().manual_seed(3))
        labels = torch.tensor([0, 1, 2, 3])
        torch.testing.assert_close(training_loss(model, logits, labels, 0.1),
                                   F.cross_entropy(logits, labels, label_smoothing=0.1))
        scale = float(model.head.log_scale.detach().exp())
        shifted = logits.clone()
        shifted[torch.arange(4), labels] -= 0.2 * scale
        torch.testing.assert_close(training_loss(model, logits, labels, 0.0, cosine_margin=0.2),
                                   F.cross_entropy(shifted, labels))
        self.assertGreater(float(training_loss(model, logits, labels, 0.0, 0.2)),
                           float(training_loss(model, logits, labels, 0.0, 0.0)))


if __name__ == "__main__":
    unittest.main()
