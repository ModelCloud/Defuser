# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from types import SimpleNamespace

import torch
from torch import nn
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeConfig, Qwen2MoeForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig, Qwen3NextForCausalLM
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration

from defuser import convert_model, replace_fused_blocks
from defuser.modeling.replace_modules import ReplacementModuleBase, apply_replacements, materialize_model




def _tiny_moe_config(config_cls):
    return config_cls(
        num_hidden_layers=1,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=128,
    )


def _tiny_qwen3_omni_config():
    return Qwen3OmniMoeConfig(
        enable_audio_output=False,
        thinker_config={
            "text_config": {
                "num_hidden_layers": 1,
                "hidden_size": 64,
                "intermediate_size": 128,
                "moe_intermediate_size": 32,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "vocab_size": 128,
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
            },
            "vision_config": {
                "depth": 1,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_heads": 4,
                "out_hidden_size": 64,
                "num_position_embeddings": 64,
                "deepstack_visual_indexes": [0],
            },
            "audio_config": {
                "num_mel_bins": 16,
                "encoder_layers": 1,
                "encoder_attention_heads": 4,
                "encoder_ffn_dim": 128,
                "d_model": 64,
                "output_dim": 64,
                "max_source_positions": 32,
                "n_window": 4,
                "n_window_infer": 4,
                "conv_chunksize": 16,
                "downsample_hidden_size": 32,
            },
        },
    )


def _tiny_qwen3_5_moe_config():
    return Qwen3_5MoeConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 16,
            "intermediate_size": 128,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "out_hidden_size": 64,
            "num_position_embeddings": 64,
        },
    )


def _tiny_mixtral_config():
    return MixtralConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_local_experts=4,
        num_experts_per_tok=2,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def _assert_unfused_expert_module(experts):
    assert hasattr(experts, "0")
    expert0 = getattr(experts, "0")
    assert hasattr(expert0, "gate_proj")
    assert hasattr(expert0, "up_proj")
    assert hasattr(expert0, "down_proj")


def test_qwen2_moe():
    model_type = "qwen2_moe"
    replace_fused_blocks(model_type)

    model = Qwen2MoeForCausalLM(_tiny_moe_config(Qwen2MoeConfig))
    assert model.config.model_type == model_type

    converted = convert_model(model, max_layers=1)
    assert not converted

    _assert_unfused_expert_module(model.model.layers[0].mlp.experts)


def test_qwen3_moe():
    model_type = "qwen3_moe"
    replace_fused_blocks(model_type)

    model = Qwen3MoeForCausalLM(_tiny_moe_config(Qwen3MoeConfig))
    assert model.config.model_type == model_type

    converted = convert_model(model, max_layers=1)
    assert not converted

    _assert_unfused_expert_module(model.model.layers[0].mlp.experts)


def test_qwen3_next():
    model = Qwen3NextForCausalLM(_tiny_moe_config(Qwen3NextConfig))
    assert model.config.model_type == "qwen3_next"

    converted = convert_model(model, max_layers=1)
    assert converted

    _assert_unfused_expert_module(model.model.layers[0].mlp.experts)


def test_qwen3_omni():
    model = Qwen3OmniMoeForConditionalGeneration(_tiny_qwen3_omni_config())
    assert model.config.model_type == "qwen3_omni_moe"

    converted = convert_model(model, max_layers=1)
    assert converted

    _assert_unfused_expert_module(model.thinker.model.layers[0].mlp.experts)


def test_qwen3_omni_runtime_patch_adds_text_forward_and_generate_defaults():
    recorded = {}

    class DummyQwen3Omni(nn.Module):
        __module__ = "transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe"

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="qwen3_omni_moe")
            self.thinker = lambda *args, **kwargs: ("thinker", args, kwargs)

        def generate(self, *args, return_audio=None, **kwargs):
            recorded["args"] = args
            recorded["kwargs"] = kwargs
            recorded["return_audio"] = return_audio
            return "generated"

    model = DummyQwen3Omni()
    convert_model(model)

    assert model.forward("hello", top_p=0.9) == ("thinker", ("hello",), {"top_p": 0.9})
    assert model.generate(torch.tensor([[1, 2]])) == "generated"
    assert recorded["return_audio"] is False


def test_apply_replacements_runs_custom_replacements():
    class DummyOriginal(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

    class DummyReplacement(ReplacementModuleBase):
        @classmethod
        def original_module_class(cls) -> str:
            return "DummyOriginal"

        @classmethod
        def from_original(cls, original: torch.nn.Module, config):
            replacement = cls(original)
            replacement.config = config
            return replacement

        def _materialize_weights(self) -> None:
            return

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()
            self.layers = nn.ModuleList([DummyOriginal()])

    model = DummyModel()
    apply_replacements(model)

    assert isinstance(model.layers[0], DummyReplacement)


def test_qwen3_5_moe():
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

    model = Qwen3_5MoeForConditionalGeneration(_tiny_qwen3_5_moe_config())
    assert model.config.model_type == "qwen3_5_moe"

    original_moe_block = model.model.language_model.layers[0].mlp
    assert isinstance(original_moe_block, Qwen3_5MoeSparseMoeBlock)

    hidden_dim = original_moe_block.experts.gate_up_proj.shape[-1]
    intermediate_dim = original_moe_block.experts.gate_up_proj.shape[1] // 2

    expected_gate = original_moe_block.experts.gate_up_proj[0, :intermediate_dim, :hidden_dim].contiguous().clone()
    expected_up = original_moe_block.experts.gate_up_proj[0, intermediate_dim:, :hidden_dim].contiguous().clone()
    expected_down = original_moe_block.experts.down_proj[0, :hidden_dim, :intermediate_dim].contiguous().clone()

    converted = convert_model(model, cleanup_original=False, max_layers=1)
    assert converted

    moe_block = model.model.language_model.layers[0].mlp
    experts = moe_block.experts

    _assert_unfused_expert_module(experts)
    expert0 = getattr(experts, "0")

    materialize_model(model.model.language_model.layers[0])

    torch.testing.assert_close(expert0.gate_proj.weight, expected_gate)
    torch.testing.assert_close(expert0.up_proj.weight, expected_up)
    torch.testing.assert_close(expert0.down_proj.weight, expected_down)


def test_mixtral():
    from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

    model = MixtralForCausalLM(_tiny_mixtral_config())
    assert model.config.model_type == "mixtral"

    original_moe_block = model.model.layers[0].mlp
    assert isinstance(original_moe_block, MixtralSparseMoeBlock)

    hidden_dim = original_moe_block.experts.gate_up_proj.shape[-1]
    intermediate_dim = original_moe_block.experts.gate_up_proj.shape[1] // 2

    expected_gate = original_moe_block.experts.gate_up_proj[0, :intermediate_dim, :hidden_dim].contiguous().clone()
    expected_up = original_moe_block.experts.gate_up_proj[0, intermediate_dim:, :hidden_dim].contiguous().clone()
    expected_down = original_moe_block.experts.down_proj[0, :hidden_dim, :intermediate_dim].contiguous().clone()

    converted = convert_model(model, cleanup_original=False, max_layers=1)
    assert converted

    moe_block = model.model.layers[0].mlp
    experts = moe_block.experts

    _assert_unfused_expert_module(experts)
    expert0 = getattr(experts, "0")

    materialize_model(model.model.layers[0])

    torch.testing.assert_close(expert0.gate_proj.weight, expected_gate)
    torch.testing.assert_close(expert0.up_proj.weight, expected_up)
    torch.testing.assert_close(expert0.down_proj.weight, expected_down)
