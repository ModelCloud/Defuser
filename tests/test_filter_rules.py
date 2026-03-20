# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Experts
from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.models.phi3.modeling_phi3 import Phi3MLP

from defuser import convert_model
from defuser.modeling.replace_modules import ReplacementModuleBase, apply_replacements
from defuser.utils.common import compile_module_name_filter, matches_module_name_filter


class _DummyLayer(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module


class _WrappedModel(nn.Module):
    def __init__(self, model_type: str, modules: list[nn.Module]):
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type)
        self.layers = nn.ModuleList([_DummyLayer(module) for module in modules])


def _tiny_phi3_mlp() -> Phi3MLP:
    config = Phi3Config(
        hidden_size=64,
        intermediate_size=128,
        hidden_act="silu",
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
    )
    return Phi3MLP(config).eval()


def _tiny_deepseek_v2_experts() -> DeepseekV2Experts:
    config = DeepseekV2Config(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        num_experts_per_tok=2,
        n_routed_experts=4,
        num_local_experts=4,
        vocab_size=128,
    )
    return DeepseekV2Experts(config).eval()


class FilterDummyOriginal(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class FilterDummyReplacement(ReplacementModuleBase):
    @classmethod
    def original_module_class(cls) -> str:
        return "FilterDummyOriginal"

    @classmethod
    def from_original(cls, original: torch.nn.Module, config):
        replacement = cls(original)
        replacement.config = config
        return replacement

    def _materialize_weights(self) -> None:
        return


def test_filter_rules_default_to_positive_and_negative_has_priority():
    positive = compile_module_name_filter(["layers\\.1\\.module$"])
    assert matches_module_name_filter("layers.1.module", positive) is True
    assert matches_module_name_filter("layers.0.module", positive) is False

    overridden = compile_module_name_filter(
        ["+:layers\\.0\\.module$", "-:layers\\.0\\.module$"]
    )
    assert matches_module_name_filter("layers.0.module", overridden) is False


def test_filter_rules_use_pcre_syntax():
    module_filter = compile_module_name_filter([r"+:^layers\.\K0\.module$"])
    assert matches_module_name_filter("layers.0.module", module_filter) is True
    assert matches_module_name_filter("layers.1.module", module_filter) is False


def test_filter_rules_match_full_hf_style_module_paths():
    module_filter = compile_module_name_filter([r"+:^model\.layers\.0\.mlp\.experts$"])

    assert matches_module_name_filter("model.layers.0.mlp.experts", module_filter) is True
    assert matches_module_name_filter("model.layers.1.mlp.experts", module_filter) is False


def test_filter_rules_require_positive_match_when_filter_is_present():
    module_filter = compile_module_name_filter([r"-:^model\.layers\.1\.mlp\.experts$"])

    assert matches_module_name_filter("model.layers.0.mlp.experts", module_filter) is False
    assert matches_module_name_filter("model.layers.1.mlp.experts", module_filter) is False
    assert matches_module_name_filter("model.layers.0.mlp.experts", []) is False


def test_filter_rules_reject_invalid_filter_inputs():
    with pytest.raises(TypeError, match="filter must be a sequence of regex strings"):
        compile_module_name_filter(r"+:^layers\.0$")

    with pytest.raises(TypeError, match="filter rules must be strings"):
        compile_module_name_filter([r"+:^layers\.0$", 1])


def test_convert_model_filter_limits_dense_runtime_patch():
    model = _WrappedModel("phi3", [_tiny_phi3_mlp(), _tiny_phi3_mlp()])

    converted = convert_model(model, filter=[r"+:^layers\.\K0\.module$"])

    assert converted is True
    assert hasattr(model.layers[0].module, "gate_proj")
    assert not hasattr(model.layers[0].module, "gate_up_proj")
    assert hasattr(model.layers[1].module, "gate_up_proj")
    assert not hasattr(model.layers[1].module, "gate_proj")


def test_convert_model_filter_negative_overrides_positive():
    model = _WrappedModel("phi3", [_tiny_phi3_mlp(), _tiny_phi3_mlp()])

    converted = convert_model(
        model,
        filter=[
            r"+:^layers\.0\.module$",
            r"-:^layers\.0\.module$",
        ],
    )

    assert converted is True
    assert hasattr(model.layers[0].module, "gate_up_proj")
    assert not hasattr(model.layers[0].module, "gate_proj")
    assert hasattr(model.layers[1].module, "gate_up_proj")
    assert not hasattr(model.layers[1].module, "gate_proj")


def test_convert_model_filter_combines_with_max_layers():
    model = _WrappedModel("phi3", [_tiny_phi3_mlp(), _tiny_phi3_mlp()])

    converted = convert_model(
        model,
        max_layers=1,
        filter=[r"+:^layers\.1\.module$"],
    )

    assert converted is True
    assert hasattr(model.layers[0].module, "gate_up_proj")
    assert not hasattr(model.layers[0].module, "gate_proj")
    assert hasattr(model.layers[1].module, "gate_up_proj")
    assert not hasattr(model.layers[1].module, "gate_proj")


def test_convert_model_filter_applies_to_moe_tensor_defusion():
    matched = _WrappedModel("deepseek_v2", [_tiny_deepseek_v2_experts()])
    skipped = _WrappedModel("deepseek_v2", [_tiny_deepseek_v2_experts()])

    assert convert_model(matched, filter=[r"+:^layers\.0\.module$"]) is True
    assert hasattr(matched.layers[0].module, "0")
    assert not hasattr(matched.layers[0].module, "gate_up_proj")

    assert convert_model(skipped, filter=[r"+:^layers\.1\.module$"]) is True
    assert not hasattr(skipped.layers[0].module, "0")
    assert hasattr(skipped.layers[0].module, "gate_up_proj")


def test_apply_replacements_filter_applies_to_custom_replacements():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()
            self.layers = nn.ModuleList([FilterDummyOriginal()])

    skipped = DummyModel()
    apply_replacements(skipped, auto_detect_moe=False, filter_rules=[r"+:^layers\.1$"])
    assert isinstance(skipped.layers[0], FilterDummyOriginal)

    matched = DummyModel()
    apply_replacements(matched, auto_detect_moe=False, filter_rules=[r"+:^layers\.0$"])
    assert isinstance(matched.layers[0], FilterDummyReplacement)
