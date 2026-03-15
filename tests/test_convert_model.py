# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeConfig, Qwen2MoeForCausalLM
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig, Qwen3NextForCausalLM

from defuser import convert_model
from defuser.modeling.replace_modules import materialize_model


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


def _assert_unfused_expert_module(experts):
    assert hasattr(experts, "0")
    expert0 = getattr(experts, "0")
    assert hasattr(expert0, "gate_proj")
    assert hasattr(expert0, "up_proj")
    assert hasattr(expert0, "down_proj")


def test_qwen2_moe():
    model = Qwen2MoeForCausalLM(_tiny_moe_config(Qwen2MoeConfig))
    assert model.config.model_type == "qwen2_moe"

    converted = convert_model(model, max_layers=1)
    assert converted

    _assert_unfused_expert_module(model.model.layers[0].mlp.experts)


def test_qwen3_moe():
    model_id = "Qwen/Qwen3-30B-A3B"
    config = AutoConfig.from_pretrained(model_id)
    config.num_hidden_layers = 1
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        ignore_mismatched_sizes=True,
    )

    assert model.config.model_type == "qwen3_moe"

    converted = convert_model(model, max_layers=1)
    assert converted

    _assert_unfused_expert_module(model.model.layers[0].mlp.experts)


def test_qwen3_next():
    model = Qwen3NextForCausalLM(_tiny_moe_config(Qwen3NextConfig))
    assert model.config.model_type == "qwen3_next"

    converted = convert_model(model, max_layers=1)
    assert converted

    _assert_unfused_expert_module(model.model.layers[0].mlp.experts)


def test_qwen3_5_moe():
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

    config = AutoConfig.from_pretrained("/monster/data/model/Qwen3.5-35B-A3B")
    config.text_config.num_hidden_layers = 1
    model = AutoModelForImageTextToText.from_pretrained(
        "/monster/data/model/Qwen3.5-35B-A3B",
        config=config,
        ignore_mismatched_sizes=True,
    )
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

    model_path = "/monster/data/model/Mixtral-8x7B-Instruct-v0.1" # "mistralai/Mixtral-8x7B-Instruct-v0.1"
    config = AutoConfig.from_pretrained(model_path)
    config.num_hidden_layers = 1
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        ignore_mismatched_sizes=True,
    )
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
