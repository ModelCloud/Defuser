# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from torch import nn
from transformers.core_model_loading import WeightConverter
from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeConfig, Glm4MoeForCausalLM, Glm4MoeMoE
from transformers.models.glm4v.configuration_glm4v import Glm4vConfig
from transformers.models.glm4v.modeling_glm4v import Glm4vForConditionalGeneration
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM, MixtralSparseMoeBlock
from transformers.models.qwen2_moe.modeling_qwen2_moe import (
    Qwen2MoeConfig,
    Qwen2MoeForCausalLM,
    Qwen2MoeSparseMoeBlock,
)
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
    Qwen3MoeSparseMoeBlock,
)
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextConfig,
    Qwen3NextForCausalLM,
    Qwen3NextSparseMoeBlock,
)
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssConfig, GptOssForCausalLM
from transformers.models.llama4.modeling_llama4 import Llama4Config, Llama4ForConditionalGeneration
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeThinkerTextSparseMoeBlock,
)

import defuser.defuser as defuser_api
import defuser.utils.hf as hf_utils
from defuser import convert_model, replace_fused_blocks
from defuser.checkpoint_ops import OwnedChunk, SplitFusedExpertGateUpProj
from defuser.model_registry import MODEL_CONFIG
from defuser.modeling.replace_modules import ReplacementModuleBase, apply_replacements, materialize_model
from defuser.modeling.unfused_moe.glm4_moe import LinearGlm4MoeMoE
from defuser.modeling.unfused_moe.mixtral import LinearMixtralSparseMoeBlock
from defuser.modeling.unfused_moe.qwen2_moe import LinearQwen2MoeSparseMoeBlock
from defuser.modeling.unfused_moe.qwen3_moe import LinearQwen3MoeSparseMoeBlock
from defuser.modeling.unfused_moe.qwen3_next import LinearQwen3NextSparseMoeBlock
from defuser.modeling.unfused_moe.qwen3_omni_moe import LinearQwen3OmniMoeThinkerTextSparseMoeBlock
from defuser.utils.common import MIN_SUPPORTED_TRANSFORMERS_VERSION




def _tiny_moe_config(config_cls):
    return config_cls(
        num_hidden_layers=1,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
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


def _tiny_glm4_moe_config():
    return Glm4MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        moe_intermediate_size=32,
        n_shared_experts=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        num_local_experts=4,
        num_experts_per_tok=2,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        head_dim=16,
        first_k_dense_replace=-1,  # Ensure that the first layer is Glm4MoeMoE.
    )


def _tiny_glm4v_config():
    return Glm4vConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "intermediate_size": 128,
            "hidden_act": "silu",
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        vision_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_channels": 3,
            "image_size": 16,
            "patch_size": 4,
            "out_hidden_size": 64,
        },
    )


def _tiny_gpt_oss_config():
    return GptOssConfig(
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


def _tiny_llama4_config():
    return Llama4Config(
        text_config={
            "vocab_size": 128,
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "intermediate_size": 128,
            "hidden_act": "silu",
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        vision_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_channels": 3,
            "image_size": 16,
            "patch_size": 4,
            "out_hidden_size": 64,
        },
    )


def _write_single_safetensors_checkpoint(path, state_dict: dict[str, torch.Tensor], config) -> None:
    config.save_pretrained(path)
    save_file({name: tensor.detach().cpu().contiguous() for name, tensor in state_dict.items()}, str(path / "model.safetensors"))


def _build_legacy_mixtral_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    legacy_state = {}
    for name, tensor in state_dict.items():
        if name.endswith(".mlp.gate.weight"):
            legacy_state[name.replace(".mlp.", ".block_sparse_moe.")] = tensor
            continue

        if name.endswith(".mlp.experts.gate_up_proj"):
            prefix = name[: -len(".mlp.experts.gate_up_proj")] + ".block_sparse_moe.experts"
            split_size = tensor.shape[1] // 2
            for expert_idx in range(tensor.shape[0]):
                legacy_state[f"{prefix}.{expert_idx}.w1.weight"] = tensor[expert_idx, :split_size].contiguous()
                legacy_state[f"{prefix}.{expert_idx}.w3.weight"] = tensor[expert_idx, split_size:].contiguous()
            continue

        if name.endswith(".mlp.experts.down_proj"):
            prefix = name[: -len(".mlp.experts.down_proj")] + ".block_sparse_moe.experts"
            for expert_idx in range(tensor.shape[0]):
                legacy_state[f"{prefix}.{expert_idx}.w2.weight"] = tensor[expert_idx].contiguous()
            continue

        legacy_state[name] = tensor

    return legacy_state


def _create_original_mixtral_source_model(config):
    from transformers.models.mixtral import modeling_mixtral as mixtral_modeling

    current_moe_block = mixtral_modeling.MixtralSparseMoeBlock
    mixtral_modeling.MixtralSparseMoeBlock = MixtralSparseMoeBlock
    try:
        return MixtralForCausalLM(config).eval()
    finally:
        mixtral_modeling.MixtralSparseMoeBlock = current_moe_block


def _assert_unfused_expert_module(experts):
    assert hasattr(experts, "0")
    expert0 = getattr(experts, "0")
    assert hasattr(expert0, "gate_proj")
    assert hasattr(expert0, "up_proj")
    assert hasattr(expert0, "down_proj")


def _seed_floating_tensors(module: nn.Module, seed: int = 0) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for tensor in list(module.parameters()) + list(module.buffers()):
            if tensor.is_floating_point():
                tensor.copy_(torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype, device=tensor.device))


def _copy_sparse_moe_weights(original_block: nn.Module, defused_block: nn.Module) -> None:
    with torch.no_grad():
        if hasattr(original_block.gate, "weight"):
            defused_block.gate.weight.copy_(original_block.gate.weight)
        if hasattr(original_block.gate, "e_score_correction_bias"):
            defused_block.gate.e_score_correction_bias.copy_(original_block.gate.e_score_correction_bias)
        if hasattr(original_block, "shared_expert"):
            defused_block.shared_expert.load_state_dict(original_block.shared_expert.state_dict())
        if hasattr(original_block, "shared_expert_gate"):
            defused_block.shared_expert_gate.load_state_dict(original_block.shared_expert_gate.state_dict())
        if hasattr(original_block, "shared_experts"):
            defused_block.shared_experts.load_state_dict(original_block.shared_experts.state_dict())

        # Split each fused expert gate/up projection into the two linears used by the defused block.
        for expert_idx, expert in enumerate(defused_block.experts):
            fused_gate_up = original_block.experts.gate_up_proj[expert_idx]
            split_size = fused_gate_up.shape[0] // 2
            expert.gate_proj.weight.copy_(fused_gate_up[:split_size])
            expert.up_proj.weight.copy_(fused_gate_up[split_size:])
            expert.down_proj.weight.copy_(original_block.experts.down_proj[expert_idx])


def _assert_sparse_moe_defused_matches_fused_math(
    original_block: nn.Module,
    defused_block: nn.Module,
    hidden_states: torch.Tensor,
    *,
    atol: float | None = None,
    rtol: float | None = None,
) -> None:
    _seed_floating_tensors(original_block)
    _copy_sparse_moe_weights(original_block, defused_block)

    expected = original_block.eval()(hidden_states)
    actual = defused_block.eval()(hidden_states)

    # The defused replacement must preserve the exact MoE matmul path of the fused block.
    assert_close_kwargs = {}
    if atol is not None:
        assert_close_kwargs["atol"] = atol
    if rtol is not None:
        assert_close_kwargs["rtol"] = rtol
    torch.testing.assert_close(actual, expected, **assert_close_kwargs)


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
    model_type = "qwen3_next"
    replace_fused_blocks(model_type)

    model = Qwen3NextForCausalLM(_tiny_moe_config(Qwen3NextConfig))
    assert model.config.model_type == model_type

    converted = convert_model(model, max_layers=1)
    assert not converted

    _assert_unfused_expert_module(model.model.layers[0].mlp.experts)


def test_qwen3_omni():
    model_type = "qwen3_omni_moe"
    replace_fused_blocks(model_type)

    model = Qwen3OmniMoeForConditionalGeneration(_tiny_qwen3_omni_config())
    assert model.config.model_type == model_type

    converted = convert_model(model, max_layers=1)
    assert not converted

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


def test_replace_fused_blocks_returns_false_for_unregistered_model():
    assert replace_fused_blocks("unsupported_model_type") is False


def test_model_registry_requires_transformers_5_3_or_newer():
    assert {cfg["min_transformers_version"] for cfg in MODEL_CONFIG.values()} == {MIN_SUPPORTED_TRANSFORMERS_VERSION}


def test_replace_fused_blocks_warns_on_unsupported_transformers(monkeypatch):
    warnings = []

    monkeypatch.setattr(defuser_api.transformers, "__version__", "5.2.9")
    monkeypatch.setattr(defuser_api.logger, "warning", warnings.append)

    assert defuser_api.replace_fused_blocks("mixtral") is False
    assert len(warnings) == 1
    assert "replace_fused_blocks()" in warnings[0]
    assert f"transformers>={MIN_SUPPORTED_TRANSFORMERS_VERSION}" in warnings[0]


def test_convert_model_warns_on_unsupported_transformers(monkeypatch):
    warnings = []

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="mixtral")

    monkeypatch.setattr(defuser_api.transformers, "__version__", "5.2.9")
    monkeypatch.setattr(defuser_api.logger, "warning", warnings.append)

    assert defuser_api.convert_model(DummyModel()) is False
    assert len(warnings) == 1
    assert "convert_model()" in warnings[0]
    assert f"transformers>={MIN_SUPPORTED_TRANSFORMERS_VERSION}" in warnings[0]


def test_pre_check_config_warns_on_unsupported_transformers(monkeypatch):
    warnings = []

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="mixtral")

    monkeypatch.setattr(hf_utils.transformers, "__version__", "5.2.9")
    monkeypatch.setattr(hf_utils.logger, "warning", warnings.append)

    assert hf_utils.pre_check_config(DummyModel()) is False
    assert len(warnings) == 1
    assert "pre_check_config()" in warnings[0]
    assert f"transformers>={MIN_SUPPORTED_TRANSFORMERS_VERSION}" in warnings[0]


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
    model_type = "mixtral"
    replace_fused_blocks(model_type)

    model = MixtralForCausalLM(_tiny_mixtral_config())
    assert model.config.model_type == model_type

    original_moe_block = model.model.layers[0].mlp
    assert isinstance(original_moe_block, LinearMixtralSparseMoeBlock)

    converted = convert_model(model, cleanup_original=False, max_layers=1)
    assert not converted

    _assert_unfused_expert_module(model.model.layers[0].mlp.experts)


def test_mixtral_checkpoint_mapping_splits_fused_experts():
    from defuser.defuser import get_checkpoint_conversion_mapping

    mapping = get_checkpoint_conversion_mapping("mixtral")
    gate_up_converter = next(
        item
        for item in mapping
        if isinstance(item, WeightConverter) and item.source_patterns == [".experts.gate_up_proj"]
    )

    assert isinstance(gate_up_converter.operations[0], SplitFusedExpertGateUpProj)
    assert gate_up_converter.target_patterns == [
        ".experts.0.gate_proj.weight",
        ".experts.0.up_proj.weight",
    ]

    fused_gate_up = torch.arange(4 * 6 * 8, dtype=torch.float32).reshape(4, 6, 8)
    split = gate_up_converter.operations[0].convert(
        {".experts.gate_up_proj": fused_gate_up},
        gate_up_converter.source_patterns,
        gate_up_converter.target_patterns,
    )

    torch.testing.assert_close(split[".experts.0.gate_proj.weight"], fused_gate_up[0, :3])
    torch.testing.assert_close(split[".experts.0.up_proj.weight"], fused_gate_up[0, 3:])
    torch.testing.assert_close(split[".experts.3.gate_proj.weight"], fused_gate_up[3, :3])
    torch.testing.assert_close(split[".experts.3.up_proj.weight"], fused_gate_up[3, 3:])


def test_mixtral_from_pretrained_loads_fused_checkpoint_into_defused_model(tmp_path):
    config = _tiny_mixtral_config()
    source_model = _create_original_mixtral_source_model(config)
    source_state = source_model.state_dict()

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        expected_logits = source_model(input_ids=input_ids).logits

    _write_single_safetensors_checkpoint(tmp_path, source_state, config)

    replace_fused_blocks("mixtral")

    loaded = MixtralForCausalLM.from_pretrained(tmp_path).eval()
    assert isinstance(loaded.model.layers[0].mlp, LinearMixtralSparseMoeBlock)

    expert0 = loaded.model.layers[0].mlp.experts[0]
    fused_gate_up = source_state["model.layers.0.mlp.experts.gate_up_proj"][0]
    fused_down = source_state["model.layers.0.mlp.experts.down_proj"][0]

    torch.testing.assert_close(expert0.gate_proj.weight, fused_gate_up[: config.intermediate_size])
    torch.testing.assert_close(expert0.up_proj.weight, fused_gate_up[config.intermediate_size :])
    torch.testing.assert_close(expert0.down_proj.weight, fused_down)

    with torch.no_grad():
        actual_logits = loaded(input_ids=input_ids).logits

    torch.testing.assert_close(actual_logits, expected_logits)


def test_mixtral_from_pretrained_loads_legacy_serialized_checkpoint(tmp_path):
    config = _tiny_mixtral_config()
    source_model = _create_original_mixtral_source_model(config)
    source_state = source_model.state_dict()

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        expected_logits = source_model(input_ids=input_ids).logits

    _write_single_safetensors_checkpoint(tmp_path, _build_legacy_mixtral_state_dict(source_state), config)

    replace_fused_blocks("mixtral")

    loaded = MixtralForCausalLM.from_pretrained(tmp_path).eval()
    assert isinstance(loaded.model.layers[0].mlp, LinearMixtralSparseMoeBlock)

    with torch.no_grad():
        actual_logits = loaded(input_ids=input_ids).logits

    torch.testing.assert_close(actual_logits, expected_logits)


def test_glm4_moe():
    model_type = "glm4_moe"
    replace_fused_blocks(model_type)

    model = Glm4MoeForCausalLM(_tiny_glm4_moe_config())
    assert model.config.model_type == model_type
    original_moe_block = model.model.layers[0].mlp
    assert isinstance(original_moe_block, LinearGlm4MoeMoE)

    converted = convert_model(model, cleanup_original=False, max_layers=1)
    assert not converted

    _assert_unfused_expert_module(model.model.layers[0].mlp.experts)


def test_glm4v():
    model_type = "glm4v"
    replace_fused_blocks(model_type)

    from defuser.modeling.glm4v import LinearGlm4vTextMLP

    model = Glm4vForConditionalGeneration(_tiny_glm4v_config())
    assert model.config.model_type == model_type

    mlp = model.model.language_model.layers[0].mlp
    assert isinstance(mlp, LinearGlm4vTextMLP)
    assert hasattr(mlp, "gate_proj")
    assert hasattr(mlp, "up_proj")
    assert hasattr(mlp, "down_proj")
    assert not hasattr(mlp, "gate_up_proj")

    converted = convert_model(model, cleanup_original=False, max_layers=1)
    assert not converted


def test_mixtral_defused_forward_matches_fused_math():
    config = _tiny_mixtral_config()
    hidden_states = torch.randn(2, 3, config.hidden_size, dtype=torch.float32)

    _assert_sparse_moe_defused_matches_fused_math(
        MixtralSparseMoeBlock(config),
        LinearMixtralSparseMoeBlock(config),
        hidden_states,
    )


def test_qwen2_moe_defused_forward_matches_fused_math():
    config = _tiny_moe_config(Qwen2MoeConfig)
    hidden_states = torch.randn(2, 3, config.hidden_size, dtype=torch.float32)

    _assert_sparse_moe_defused_matches_fused_math(
        Qwen2MoeSparseMoeBlock(config),
        LinearQwen2MoeSparseMoeBlock(config),
        hidden_states,
    )


def test_qwen3_moe_defused_forward_matches_fused_math():
    config = _tiny_moe_config(Qwen3MoeConfig)
    hidden_states = torch.randn(2, 3, config.hidden_size, dtype=torch.float32)

    _assert_sparse_moe_defused_matches_fused_math(
        Qwen3MoeSparseMoeBlock(config),
        LinearQwen3MoeSparseMoeBlock(config),
        hidden_states,
    )


def test_qwen3_next_defused_forward_matches_fused_math():
    config = _tiny_moe_config(Qwen3NextConfig)
    hidden_states = torch.randn(2, 3, config.hidden_size, dtype=torch.float32)

    _assert_sparse_moe_defused_matches_fused_math(
        Qwen3NextSparseMoeBlock(config),
        LinearQwen3NextSparseMoeBlock(config),
        hidden_states,
    )


def test_qwen3_omni_defused_forward_matches_fused_math():
    config = _tiny_qwen3_omni_config().thinker_config.text_config
    hidden_states = torch.randn(2, 3, config.hidden_size, dtype=torch.float32)

    _assert_sparse_moe_defused_matches_fused_math(
        Qwen3OmniMoeThinkerTextSparseMoeBlock(config),
        LinearQwen3OmniMoeThinkerTextSparseMoeBlock(config),
        hidden_states,
    )


def test_glm4_moe_defused_forward_matches_fused_math():
    config = _tiny_glm4_moe_config()
    hidden_states = torch.randn(2, 3, config.hidden_size, dtype=torch.float32)

    _assert_sparse_moe_defused_matches_fused_math(
        Glm4MoeMoE(config),
        LinearGlm4MoeMoE(config),
        hidden_states,
        # GLM4 MoE now shows tiny fp32 roundoff drift in 5.3.0 because the fused gate/up matmul
        # is compared against two split linears. Keep the tolerance narrow enough to catch real regressions.
        atol=3e-4,
        rtol=1e-4,
    )


def test_defused_models_preserve_output_router_logits_capture():
    cases = [
        (
            "mixtral",
            lambda: MixtralForCausalLM(_tiny_mixtral_config()),
        ),
        (
            "qwen2_moe",
            lambda: Qwen2MoeForCausalLM(_tiny_moe_config(Qwen2MoeConfig)),
        ),
        (
            "qwen3_moe",
            lambda: Qwen3MoeForCausalLM(_tiny_moe_config(Qwen3MoeConfig)),
        ),
    ]

    for model_type, build_model in cases:
        replace_fused_blocks(model_type)
        model = build_model().eval()
        outputs = model(input_ids=torch.tensor([[1, 2, 3]]), output_router_logits=True)

        # Router logits are captured through upstream hooks, so defused blocks must keep the same router module type.
        assert outputs.router_logits is not None
        assert len(outputs.router_logits) == 1
        assert outputs.router_logits[0].shape == (3, model.config.num_experts)


def test_glm4v_checkpoint_mapping_splits_gate_up_proj():
    from defuser.defuser import get_checkpoint_conversion_mapping

    mapping = get_checkpoint_conversion_mapping("glm4v")
    converter = next(
        item
        for item in mapping
        if isinstance(item, WeightConverter) and item.source_patterns == ["mlp.gate_up_proj.weight"]
    )
    assert isinstance(converter.operations[0], OwnedChunk)

    assert converter.target_patterns == [
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
    ]

    fused = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    split = converter.operations[0].convert(
        {"mlp.gate_up_proj.weight": fused},
        converter.source_patterns,
        converter.target_patterns,
    )

    torch.testing.assert_close(split["mlp.gate_proj.weight"], fused[:3])
    torch.testing.assert_close(split["mlp.up_proj.weight"], fused[3:])
    assert split["mlp.gate_proj.weight"].data_ptr() != split["mlp.up_proj.weight"].data_ptr()


def test_glm4v_split_forward_matches_fused_math():
    from defuser.modeling.glm4v import LinearGlm4vTextMLP

    config = SimpleNamespace(hidden_size=8, intermediate_size=6, hidden_act="silu")
    fused_gate_up = torch.randn(2 * config.intermediate_size, config.hidden_size, dtype=torch.float32)
    down_proj = torch.randn(config.hidden_size, config.intermediate_size, dtype=torch.float32)
    hidden_states = torch.randn(3, config.hidden_size, dtype=torch.float32)

    mlp = LinearGlm4vTextMLP(config)
    with torch.no_grad():
        mlp.gate_proj.weight.copy_(fused_gate_up[: config.intermediate_size])
        mlp.up_proj.weight.copy_(fused_gate_up[config.intermediate_size :])
        mlp.down_proj.weight.copy_(down_proj)

    fused_gate, fused_up = (hidden_states @ fused_gate_up.transpose(0, 1)).chunk(2, dim=-1)
    expected = (torch.nn.functional.silu(fused_gate) * fused_up) @ down_proj.transpose(0, 1)

    # The split module should exactly reproduce the original fused MLP math.
    torch.testing.assert_close(mlp(hidden_states), expected)

def test_gpt_oss():
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP

    model = GptOssForCausalLM(_tiny_gpt_oss_config())
    assert model.config.model_type == "gpt_oss"

    original_moe_block = model.model.layers[0].mlp
    assert isinstance(original_moe_block, GptOssMLP)

    experts = original_moe_block.experts

    gate_up = experts.gate_up_proj
    down = experts.down_proj

    expert_dim = model.config.intermediate_size

    expected_gate = gate_up[0, :, :expert_dim].clone().T
    expected_up = gate_up[0, :, expert_dim:].clone().T
    expected_down = down[0].clone().T

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


def test_gpt_oss_split_forward_matches_fused_math():
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

    model = GptOssForCausalLM(_tiny_gpt_oss_config())
    fused_experts = model.model.layers[0].mlp.experts
    assert isinstance(fused_experts, GptOssExperts)

    hidden_states = torch.randn(5, model.config.hidden_size, dtype=torch.float32)
    top_k_index = torch.zeros((hidden_states.size(0), 1), dtype=torch.long)
    top_k_weights = torch.ones((hidden_states.size(0), 1), dtype=hidden_states.dtype)

    with torch.no_grad():
        expected = fused_experts(hidden_states, top_k_index, top_k_weights)

    converted = convert_model(model, cleanup_original=False, max_layers=1)
    assert converted

    split_experts = model.model.layers[0].mlp.experts
    _assert_unfused_expert_module(split_experts)
    materialize_model(model.model.layers[0])
    with torch.no_grad():
        actual = split_experts(hidden_states, top_k_index, top_k_weights)

    # The split experts path should exactly reproduce the original fused experts math.
    torch.testing.assert_close(actual, expected)

def test_llama4():
    from transformers.models.llama4.modeling_llama4 import Llama4TextMoe

    model = Llama4ForConditionalGeneration(_tiny_llama4_config())
    assert model.config.model_type == "llama4"

    original_moe_block = model.language_model.model.layers[0].feed_forward
    assert isinstance(original_moe_block, Llama4TextMoe)

    experts = original_moe_block.experts

    gate_up = experts.gate_up_proj
    down = experts.down_proj

    expert_dim = model.config.text_config.intermediate_size

    expected_gate = gate_up[0, :, :expert_dim].clone().T
    expected_up = gate_up[0, :, expert_dim:].clone().T
    expected_down = down[0].clone().T

    converted = convert_model(model, cleanup_original=False, max_layers=1)
    assert converted

    moe_block = model.language_model.model.layers[0].feed_forward
    experts = moe_block.experts

    _assert_unfused_expert_module(experts)
    expert0 = getattr(experts, "0")

    materialize_model(model.language_model.model.layers[0])

    torch.testing.assert_close(expert0.gate_proj.weight, expected_gate)
    torch.testing.assert_close(expert0.up_proj.weight, expected_up)
    torch.testing.assert_close(expert0.down_proj.weight, expected_down)


def test_llama4_experts_forward_matches_fused_math():
    model = Llama4ForConditionalGeneration(_tiny_llama4_config())
    fused_experts = model.language_model.model.layers[0].feed_forward.experts

    hidden_states = torch.randn(fused_experts.num_experts * 5, model.config.text_config.hidden_size, dtype=torch.float32)
    with torch.no_grad():
        expected = fused_experts(hidden_states)

    converted = convert_model(model, cleanup_original=False, max_layers=1)
    assert converted

    split_experts = model.language_model.model.layers[0].feed_forward.experts
    _assert_unfused_expert_module(split_experts)
    materialize_model(model.language_model.model.layers[0])
    with torch.no_grad():
        actual = split_experts(hidden_states)

    # The batched-input generic path should preserve the original llama4 experts math.
    torch.testing.assert_close(actual, expected)


def test_llama4_split_forward_matches_fused_math():
    from transformers.models.llama4.modeling_llama4 import Llama4TextMLP

    config = _tiny_llama4_config().text_config
    fused_gate_up = torch.randn(2 * config.intermediate_size, config.hidden_size, dtype=torch.float32)
    down_proj = torch.randn(config.hidden_size, config.intermediate_size, dtype=torch.float32)
    hidden_states = torch.randn(3, config.hidden_size, dtype=torch.float32)

    mlp = Llama4TextMLP(config)
    with torch.no_grad():
        mlp.gate_proj.weight.copy_(fused_gate_up[: config.intermediate_size])
        mlp.up_proj.weight.copy_(fused_gate_up[config.intermediate_size:])
        mlp.down_proj.weight.copy_(down_proj)

    fused_gate, fused_up = (hidden_states @ fused_gate_up.transpose(0, 1)).chunk(2, dim=-1)
    expected = (torch.nn.functional.silu(fused_gate) * fused_up) @ down_proj.transpose(0, 1)

    # The split module should exactly reproduce the original fused MLP math.
    torch.testing.assert_close(mlp(hidden_states), expected)
