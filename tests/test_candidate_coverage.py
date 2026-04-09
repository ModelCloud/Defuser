# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from defuser import convert_model
from defuser.model_registry import MODEL_CONFIG


class _DummyLayer(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module


class _DummyModel(nn.Module):
    def __init__(self, model_type: str, module: nn.Module):
        super().__init__()
        self.config = getattr(module, "config", None)
        if self.config is None:
            self.config = SimpleNamespace()
        self.config.model_type = model_type
        self.layers = nn.ModuleList([_DummyLayer(module)])


def _load(module_path: str, attr_name: str):
    return getattr(import_module(module_path), attr_name)


def _normalize_attr_value(obj, attr: str, value):
    current = getattr(obj, attr)
    if isinstance(current, list) and not isinstance(value, list):
        width = len(current) or 1
        return [value for _ in range(width)]
    return value


def _build_config(case: dict):
    config = _load(case["config_module"], case["config_name"])()
    sub_attr = case.get("sub_attr")
    if sub_attr is not None:
        config = getattr(config, sub_attr)
    for attr, value in {
        "hidden_size": 64,
        "intermediate_size": 128,
        "moe_intermediate_size": 32,
        "expert_ffn_hidden_size": 32,
        "num_local_experts": 4,
        "num_experts": 4,
        "n_routed_experts": 4,
        "moe_num_experts": 4,
        "num_experts_per_tok": 2,
        "hidden_act": "silu",
        "mlp_hidden_act": "silu",
        "activation": "silu",
        "activation_function": "silu",
        "use_bias": False,
        "mlp_bias": False,
        "add_bias_linear": False,
        "n_group": 1,
        "topk_group": 1,
        "n_shared_experts": 1,
        "num_shared_experts": 1,
        "routed_scaling_factor": 1.0,
        "norm_topk_prob": False,
        "use_expert_bias": False,
        "moe_latent_size": None,
        "zero_expert_num": 1,
        "adapter_rank": 8,
        "hybrid_layer_ids": [0],
        "num_mem_blocks": 1,
        "dropout_rate": 0.0,
    }.items():
        if hasattr(config, attr):
            setattr(config, attr, _normalize_attr_value(config, attr, value))
    for attr, value in case.get("config_updates", {}).items():
        if hasattr(config, attr):
            setattr(config, attr, _normalize_attr_value(config, attr, value))
    return config


def _build_module(case: dict) -> nn.Module:
    module_cls = _load(case["module_path"], case["class_name"])
    kind = case.get("kind", "config")
    if kind == "parallel":
        return module_cls(case["num_experts"], case["input_size"], case["output_size"]).eval()
    if kind == "zamba2":
        return module_cls(_build_config(case), num_fwd_mem_blocks=1, block_id=0).eval()
    config = _build_config(case)
    if case["model_type"] == "ernie4_5_vl_moe" and isinstance(getattr(config, "moe_intermediate_size", None), list):
        return module_cls(config, intermediate_size=config.moe_intermediate_size[0]).eval()
    return module_cls(config).eval()


def _wrapped_model(case: dict) -> tuple[_DummyModel, nn.Module]:
    module = _build_module(case)
    generator = torch.Generator(device="cpu").manual_seed(0)
    with torch.no_grad():
        for tensor in list(module.parameters()) + list(module.buffers()):
            if tensor.is_floating_point():
                tensor.copy_(torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype, device=tensor.device))
    return _DummyModel(case["model_type"], module), module


def _patched_module(model: _DummyModel) -> nn.Module:
    return model.layers[0].module


def _assert_expert_container(module: nn.Module, attrs: tuple[str, ...]) -> None:
    assert hasattr(module, "0")
    expert0 = getattr(module, "0")
    for attr in attrs:
        assert hasattr(expert0, attr)


def _standard_hidden(case: dict) -> torch.Tensor:
    return torch.randn(case.get("hidden_shape", (5, case["input_dim"])), dtype=torch.float32)


STANDARD_MOE_CASES = [
    {
        "model_type": "deepseek_v2",
        "module_path": "transformers.models.deepseek_v2.modeling_deepseek_v2",
        "class_name": "DeepseekV2Experts",
        "config_module": "transformers.models.deepseek_v2.configuration_deepseek_v2",
        "config_name": "DeepseekV2Config",
        "config_updates": {"hidden_size": 64, "moe_intermediate_size": 32, "num_local_experts": 4, "hidden_act": "silu"},
        "input_dim": 64,
        "route_indices": [[0], [1], [2], [3], [0]],
        "expert_attrs": ("gate_proj", "up_proj", "down_proj"),
    },
    {
        "model_type": "deepseek_v3",
        "module_path": "transformers.models.deepseek_v3.modeling_deepseek_v3",
        "class_name": "DeepseekV3NaiveMoe",
        "config_module": "transformers.models.deepseek_v3.configuration_deepseek_v3",
        "config_name": "DeepseekV3Config",
        "config_updates": {"hidden_size": 64, "moe_intermediate_size": 32, "num_local_experts": 4, "hidden_act": "silu"},
        "input_dim": 64,
        "route_indices": [[0], [1], [2], [3], [0]],
        "expert_attrs": ("gate_proj", "up_proj", "down_proj"),
    },
    {
        "model_type": "dots1",
        "module_path": "transformers.models.dots1.modeling_dots1",
        "class_name": "Dots1NaiveMoe",
        "config_module": "transformers.models.dots1.configuration_dots1",
        "config_name": "Dots1Config",
        "config_updates": {"hidden_size": 64, "moe_intermediate_size": 32, "num_local_experts": 4, "hidden_act": "silu"},
        "input_dim": 64,
        "route_indices": [[0], [1], [2], [3], [0]],
        "expert_attrs": ("gate_proj", "up_proj", "down_proj"),
    },
    {
        "model_type": "ernie4_5_moe",
        "module_path": "transformers.models.ernie4_5_moe.modeling_ernie4_5_moe",
        "class_name": "Ernie4_5_MoeExperts",
        "config_module": "transformers.models.ernie4_5_moe.configuration_ernie4_5_moe",
        "config_name": "Ernie4_5_MoeConfig",
        "config_updates": {"hidden_size": 64, "moe_intermediate_size": 32, "num_local_experts": 4, "hidden_act": "silu", "use_bias": False},
        "input_dim": 64,
        "route_indices": [[0], [1], [2], [3], [0]],
        "expert_attrs": ("gate_proj", "up_proj", "down_proj"),
    },
    {
        "model_type": "ernie4_5_vl_moe",
        "module_path": "transformers.models.ernie4_5_vl_moe.modeling_ernie4_5_vl_moe",
        "class_name": "Ernie4_5_VLMoeMoeExperts",
        "config_module": "transformers.models.ernie4_5_vl_moe.configuration_ernie4_5_vl_moe",
        "config_name": "Ernie4_5_VLMoeConfig",
        "sub_attr": "text_config",
        "config_updates": {"hidden_size": 64, "moe_intermediate_size": 32, "num_local_experts": 4, "hidden_act": "silu", "use_bias": False},
        "input_dim": 64,
        "route_indices": [[0], [1], [2], [3], [0]],
        "expert_attrs": ("gate_proj", "up_proj", "down_proj"),
    },
    {
        "model_type": "exaone_moe",
        "module_path": "transformers.models.exaone_moe.modeling_exaone_moe",
        "class_name": "ExaoneMoeExperts",
        "config_module": "transformers.models.exaone_moe.configuration_exaone_moe",
        "config_name": "ExaoneMoeConfig",
        "config_updates": {"hidden_size": 64, "moe_intermediate_size": 32, "num_local_experts": 4, "hidden_act": "silu"},
        "input_dim": 64,
        "route_indices": [[0], [1], [2], [3], [0]],
        "expert_attrs": ("gate_proj", "up_proj", "down_proj"),
    },
    {
        "model_type": "flex_olmo",
        "module_path": "transformers.models.flex_olmo.modeling_flex_olmo",
        "class_name": "FlexOlmoExperts",
        "config_module": "transformers.models.flex_olmo.configuration_flex_olmo",
        "config_name": "FlexOlmoConfig",
        "config_updates": {"hidden_size": 64, "moe_intermediate_size": 32, "num_local_experts": 4, "hidden_act": "silu"},
        "input_dim": 64,
        "route_indices": [[0], [1], [2], [3], [0]],
        "expert_attrs": ("gate_proj", "up_proj", "down_proj"),
    },
    {
        "model_type": "glm4_moe_lite",
        "module_path": "transformers.models.glm4_moe_lite.modeling_glm4_moe_lite",
        "class_name": "Glm4MoeLiteNaiveMoe",
        "config_module": "transformers.models.glm4_moe_lite.configuration_glm4_moe_lite",
        "config_name": "Glm4MoeLiteConfig",
        "config_updates": {"hidden_size": 64, "moe_intermediate_size": 32, "num_local_experts": 4, "hidden_act": "silu"},
        "input_dim": 64,
        "route_indices": [[0], [1], [2], [3], [0]],
        "expert_attrs": ("gate_proj", "up_proj", "down_proj"),
    },
    {
        "model_type": "glm4v_moe",
        "module_path": "transformers.models.glm4v_moe.modeling_glm4v_moe",
        "class_name": "Glm4vMoeTextNaiveMoe",
        "config_module": "transformers.models.glm4v_moe.configuration_glm4v_moe",
        "config_name": "Glm4vMoeConfig",
        "sub_attr": "text_config",
        "config_updates": {"hidden_size": 64, "moe_intermediate_size": 32, "num_local_experts": 4, "hidden_act": "silu"},
        "input_dim": 64,
        "route_indices": [[0], [1], [2], [3], [0]],
        "expert_attrs": ("gate_proj", "up_proj", "down_proj"),
    },
    {
        "model_type": "glm_moe_dsa",
        "module_path": "transformers.models.glm_moe_dsa.modeling_glm_moe_dsa",
        "class_name": "GlmMoeDsaNaiveMoe",
        "config_module": "transformers.models.glm_moe_dsa.configuration_glm_moe_dsa",
        "config_name": "GlmMoeDsaConfig",
        "config_updates": {"hidden_size": 64, "moe_intermediate_size": 32, "num_local_experts": 4, "hidden_act": "silu"},
        "input_dim": 64,
        "route_indices": [[0], [1], [2], [3], [0]],
        "expert_attrs": ("gate_proj", "up_proj", "down_proj"),
    },
    {
        "model_type": "hunyuan_v1_moe",
        "module_path": "transformers.models.hunyuan_v1_moe.modeling_hunyuan_v1_moe",
        "class_name": "HunYuanMoEV1Experts",
        "config_module": "transformers.models.hunyuan_v1_moe.configuration_hunyuan_v1_moe",
        "config_name": "HunYuanMoEV1Config",
        "config_updates": {"hidden_size": 64, "moe_intermediate_size": 32, "num_local_experts": 4, "hidden_act": "silu"},
        "input_dim": 64,
        "route_indices": [[0], [1], [2], [3], [0]],
        "expert_attrs": ("gate_proj", "up_proj", "down_proj"),
    },
    {
        "model_type": "jamba",
        "module_path": "transformers.models.jamba.modeling_jamba",
        "class_name": "JambaExperts",
        "config_module": "transformers.models.jamba.configuration_jamba",
        "config_name": "JambaConfig",
        "config_updates": {"hidden_size": 64, "moe_intermediate_size": 32, "num_experts": 4, "num_local_experts": 4, "hidden_act": "silu"},
        "input_dim": 64,
        "route_indices": [[0], [1], [2], [3], [0]],
        "expert_attrs": ("gate_proj", "up_proj", "down_proj"),
    },
    {
        "model_type": "lfm2_moe",
        "module_path": "transformers.models.lfm2_moe.modeling_lfm2_moe",
        "class_name": "Lfm2MoeExperts",
        "config_module": "transformers.models.lfm2_moe.configuration_lfm2_moe",
        "config_name": "Lfm2MoeConfig",
        "config_updates": {"hidden_size": 64, "moe_intermediate_size": 32, "num_experts": 4, "hidden_act": "silu"},
        "input_dim": 64,
        "route_indices": [[0], [1], [2], [3], [0]],
        "expert_attrs": ("gate_proj", "up_proj", "down_proj"),
    },
    {
        "model_type": "minimax",
        "module_path": "transformers.models.minimax.modeling_minimax",
        "class_name": "MiniMaxExperts",
        "config_module": "transformers.models.minimax.configuration_minimax",
        "config_name": "MiniMaxConfig",
        "config_updates": {"hidden_size": 64, "intermediate_size": 128, "num_local_experts": 4, "hidden_act": "silu"},
        "input_dim": 64,
        "route_indices": [[0], [1], [2], [3], [0]],
        "expert_attrs": ("gate_proj", "up_proj", "down_proj"),
    },
    {
        "model_type": "minimax_m2",
        "module_path": "transformers.models.minimax_m2.modeling_minimax_m2",
        "class_name": "MiniMaxM2Experts",
        "config_module": "transformers.models.minimax_m2.configuration_minimax_m2",
        "config_name": "MiniMaxM2Config",
        "config_updates": {"hidden_size": 64, "intermediate_size": 128, "num_local_experts": 4, "hidden_act": "silu"},
        "input_dim": 64,
        "route_indices": [[0], [1], [2], [3], [0]],
        "expert_attrs": ("gate_proj", "up_proj", "down_proj"),
    },
    {
        "model_type": "nemotron_h",
        "module_path": "transformers.models.nemotron_h.modeling_nemotron_h",
        "class_name": "NemotronHExperts",
        "config_module": "transformers.models.nemotron_h.configuration_nemotron_h",
        "config_name": "NemotronHConfig",
        "config_updates": {"hidden_size": 64, "moe_intermediate_size": 96, "n_routed_experts": 4, "mlp_hidden_act": "silu", "moe_latent_size": None},
        "input_dim": 64,
        "route_indices": [[0], [1], [2], [3], [0]],
        "expert_attrs": ("up_proj", "down_proj"),
    },
    {
        "model_type": "olmoe",
        "module_path": "transformers.models.olmoe.modeling_olmoe",
        "class_name": "OlmoeExperts",
        "config_module": "transformers.models.olmoe.configuration_olmoe",
        "config_name": "OlmoeConfig",
        "config_updates": {"hidden_size": 64, "moe_intermediate_size": 32, "num_local_experts": 4, "hidden_act": "silu"},
        "input_dim": 64,
        "route_indices": [[0], [1], [2], [3], [0]],
        "expert_attrs": ("gate_proj", "up_proj", "down_proj"),
    },
    {
        "model_type": "qwen3_vl_moe",
        "module_path": "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe",
        "class_name": "Qwen3VLMoeTextExperts",
        "config_module": "transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe",
        "config_name": "Qwen3VLMoeConfig",
        "sub_attr": "text_config",
        "config_updates": {"hidden_size": 64, "moe_intermediate_size": 32, "num_experts": 4, "num_local_experts": 4, "hidden_act": "silu"},
        "input_dim": 64,
        "route_indices": [[0], [1], [2], [3], [0]],
        "expert_attrs": ("gate_proj", "up_proj", "down_proj"),
    },
    {
        "model_type": "solar_open",
        "module_path": "transformers.models.solar_open.modeling_solar_open",
        "class_name": "SolarOpenNaiveMoe",
        "config_module": "transformers.models.solar_open.configuration_solar_open",
        "config_name": "SolarOpenConfig",
        "config_updates": {"hidden_size": 64, "moe_intermediate_size": 32, "num_local_experts": 4, "hidden_act": "silu"},
        "input_dim": 64,
        "route_indices": [[0], [1], [2], [3], [0]],
        "expert_attrs": ("gate_proj", "up_proj", "down_proj"),
    },
    {
        "model_type": "dbrx",
        "module_path": "transformers.models.dbrx.modeling_dbrx",
        "class_name": "DbrxExperts",
        "config_module": "transformers.models.dbrx.configuration_dbrx",
        "config_name": "DbrxFFNConfig",
        "config_updates": {"hidden_size": 32, "ffn_hidden_size": 64, "moe_num_experts": 4},
        "input_dim": 64,
        "hidden_shape": (2, 3, 64),
        "route_indices": [[0], [1], [2], [3], [0], [1]],
        "expert_attrs": ("gate_proj", "up_proj", "down_proj"),
        "atol": 2e-4,
        "rtol": 2e-5,
    },
    {
        "model_type": "longcat_flash",
        "module_path": "transformers.models.longcat_flash.modeling_longcat_flash",
        "class_name": "LongcatFlashExperts",
        "config_module": "transformers.models.longcat_flash.configuration_longcat_flash",
        "config_name": "LongcatFlashConfig",
        "config_updates": {"hidden_size": 64, "expert_ffn_hidden_size": 32, "n_routed_experts": 4, "zero_expert_num": 1, "hidden_act": "silu"},
        "input_dim": 64,
        "route_indices": [[0, 4], [4, 0], [2, 1], [3, 4], [1, 2]],
        "route_weights": [[0.6, 0.4], [0.7, 0.3], [0.5, 0.5], [0.8, 0.2], [0.9, 0.1]],
        "expert_attrs": ("gate_proj", "up_proj", "down_proj"),
        "identity_expert_index": 4,
    },
]


PARALLEL_CASES = [
    {
        "model_type": "granitemoe",
        "module_path": "transformers.models.granitemoe.modeling_granitemoe",
        "class_name": "GraniteMoeParallelExperts",
        "kind": "parallel",
        "num_experts": 4,
        "input_size": 64,
        "output_size": 96,
    },
    {
        "model_type": "granitemoehybrid",
        "module_path": "transformers.models.granitemoehybrid.modeling_granitemoehybrid",
        "class_name": "GraniteMoeHybridParallelExperts",
        "kind": "parallel",
        "num_experts": 4,
        "input_size": 64,
        "output_size": 96,
    },
    {
        "model_type": "granitemoeshared",
        "module_path": "transformers.models.granitemoeshared.modeling_granitemoeshared",
        "class_name": "GraniteMoeSharedParallelExperts",
        "kind": "parallel",
        "num_experts": 4,
        "input_size": 64,
        "output_size": 96,
    },
    {
        "model_type": "jetmoe",
        "module_path": "transformers.models.jetmoe.modeling_jetmoe",
        "class_name": "JetMoeParallelExperts",
        "kind": "parallel",
        "num_experts": 4,
        "input_size": 64,
        "output_size": 96,
    },
]


DENSE_CASES = [
    {
        "label": "dia",
        "model_type": "dia",
        "module_path": "transformers.models.dia.modeling_dia",
        "class_name": "DiaMLP",
        "config_module": "transformers.models.dia.configuration_dia",
        "config_name": "DiaConfig",
        "sub_attr": "decoder_config",
        "config_updates": {"hidden_size": 64, "intermediate_size": 128, "hidden_act": "silu"},
        "hidden_size": 64,
    },
    {
        "label": "glm",
        "model_type": "glm",
        "module_path": "transformers.models.glm.modeling_glm",
        "class_name": "GlmMLP",
        "config_module": "transformers.models.glm.configuration_glm",
        "config_name": "GlmConfig",
        "config_updates": {"hidden_size": 64, "intermediate_size": 128, "hidden_act": "silu"},
        "hidden_size": 64,
    },
    {
        "label": "glm4",
        "model_type": "glm4",
        "module_path": "transformers.models.glm4.modeling_glm4",
        "class_name": "Glm4MLP",
        "config_module": "transformers.models.glm4.configuration_glm4",
        "config_name": "Glm4Config",
        "config_updates": {"hidden_size": 64, "intermediate_size": 128, "hidden_act": "silu"},
        "hidden_size": 64,
    },
    {
        "label": "glm_image",
        "model_type": "glm_image",
        "module_path": "transformers.models.glm_image.modeling_glm_image",
        "class_name": "GlmImageTextMLP",
        "config_module": "transformers.models.glm_image.configuration_glm_image",
        "config_name": "GlmImageConfig",
        "sub_attr": "text_config",
        "config_updates": {"hidden_size": 64, "intermediate_size": 128, "hidden_act": "silu"},
        "hidden_size": 64,
    },
    {
        "label": "glm_ocr",
        "model_type": "glm_ocr",
        "module_path": "transformers.models.glm_ocr.modeling_glm_ocr",
        "class_name": "GlmOcrTextMLP",
        "config_module": "transformers.models.glm_ocr.configuration_glm_ocr",
        "config_name": "GlmOcrConfig",
        "sub_attr": "text_config",
        "config_updates": {"hidden_size": 64, "intermediate_size": 128, "hidden_act": "silu"},
        "hidden_size": 64,
    },
    {
        "label": "phi3",
        "model_type": "phi3",
        "module_path": "transformers.models.phi3.modeling_phi3",
        "class_name": "Phi3MLP",
        "config_module": "transformers.models.phi3.configuration_phi3",
        "config_name": "Phi3Config",
        "config_updates": {"hidden_size": 64, "intermediate_size": 128, "hidden_act": "silu"},
        "hidden_size": 64,
    },
    {
        "label": "phi4_multimodal_text",
        "model_type": "phi4_multimodal",
        "module_path": "transformers.models.phi4_multimodal.modeling_phi4_multimodal",
        "class_name": "Phi4MultimodalMLP",
        "config_module": "transformers.models.phi4_multimodal.configuration_phi4_multimodal",
        "config_name": "Phi4MultimodalConfig",
        "config_updates": {"hidden_size": 64, "intermediate_size": 128, "hidden_act": "silu"},
        "hidden_size": 64,
    },
    {
        "label": "phi4_multimodal_audio",
        "model_type": "phi4_multimodal",
        "module_path": "transformers.models.phi4_multimodal.modeling_phi4_multimodal",
        "class_name": "Phi4MultimodalAudioMLP",
        "config_module": "transformers.models.phi4_multimodal.configuration_phi4_multimodal",
        "config_name": "Phi4MultimodalAudioConfig",
        "config_updates": {"hidden_size": 64, "intermediate_size": 128, "activation": "silu", "dropout_rate": 0.0},
        "hidden_size": 64,
    },
    {
        "label": "zamba2",
        "model_type": "zamba2",
        "module_path": "transformers.models.zamba2.modeling_zamba2",
        "class_name": "Zamba2MLP",
        "config_module": "transformers.models.zamba2.configuration_zamba2",
        "config_name": "Zamba2Config",
        "config_updates": {
            "hidden_size": 64,
            "intermediate_size": 128,
            "hidden_act": "silu",
            "adapter_rank": 8,
            "hybrid_layer_ids": [0],
            "num_mem_blocks": 1,
            "add_bias_linear": False,
        },
        "hidden_size": 64,
        "kind": "zamba2",
    },
]


ALL_CANDIDATE_MODEL_TYPES = {
    "dbrx",
    "deepseek_v2",
    "deepseek_v3",
    "dia",
    "dots1",
    "ernie4_5_moe",
    "ernie4_5_vl_moe",
    "exaone_moe",
    "flex_olmo",
    "glm",
    "glm4",
    "glm4_moe",
    "glm4_moe_lite",
    "glm4v",
    "glm4v_moe",
    "glm_image",
    "glm_moe_dsa",
    "glm_ocr",
    "gpt_oss",
    "granitemoe",
    "granitemoehybrid",
    "granitemoeshared",
    "hunyuan_v1_moe",
    "jamba",
    "jetmoe",
    "lfm2_moe",
    "llama4",
    "longcat_flash",
    "minimax",
    "minimax_m2",
    "mixtral",
    "nemotron_h",
    "olmoe",
    "phi3",
    "phi4_multimodal",
    "phimoe",
    "qwen2_moe",
    "qwen3_5_moe",
    "qwen3_moe",
    "qwen3_next",
    "qwen3_omni_moe",
    "qwen3_vl_moe",
    "solar_open",
    "zamba2",
}


def test_model_registry_covers_all_scanned_candidates():
    assert ALL_CANDIDATE_MODEL_TYPES.issubset(MODEL_CONFIG)


@pytest.mark.parametrize("case", STANDARD_MOE_CASES, ids=[case["model_type"] for case in STANDARD_MOE_CASES])
def test_standard_moe_candidates_convert_and_preserve_forward(case):
    torch.manual_seed(0)
    model, original_module = _wrapped_model(case)
    hidden_states = _standard_hidden(case)
    top_k_index = torch.tensor(case["route_indices"], dtype=torch.long)
    route_weights = case.get("route_weights")
    if route_weights is None:
        top_k_weights = torch.ones(top_k_index.shape, dtype=hidden_states.dtype)
    else:
        top_k_weights = torch.tensor(route_weights, dtype=hidden_states.dtype)

    with torch.no_grad():
        expected = original_module(hidden_states.clone(), top_k_index, top_k_weights)

    converted = convert_model(model)
    assert converted is True

    patched = _patched_module(model)
    _assert_expert_container(patched, case["expert_attrs"])

    with torch.no_grad():
        actual = patched(hidden_states.clone(), top_k_index, top_k_weights)

    torch.testing.assert_close(
        actual,
        expected,
        atol=case.get("atol", 1e-5),
        rtol=case.get("rtol", 1.3e-6),
    )

    if case.get("identity_expert_index") is not None:
        assert hasattr(getattr(patched, str(case["identity_expert_index"])), "identity")


@pytest.mark.parametrize("case", PARALLEL_CASES, ids=[case["model_type"] for case in PARALLEL_CASES])
def test_parallel_expert_candidates_convert_and_preserve_forward(case):
    torch.manual_seed(0)
    model, original_module = _wrapped_model(case)
    expert_size = [2, 1, 0, 3]
    inputs = torch.randn(sum(expert_size), case["input_size"], dtype=torch.float32)

    with torch.no_grad():
        expected = original_module(inputs.clone(), expert_size)

    converted = convert_model(model)
    assert converted is True

    patched = _patched_module(model)
    _assert_expert_container(patched, ("linear",))
    assert not hasattr(patched, "weight")

    with torch.no_grad():
        actual = patched(inputs.clone(), expert_size)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("case", DENSE_CASES, ids=[case["label"] for case in DENSE_CASES])
def test_dense_candidates_convert_and_preserve_forward(case):
    torch.manual_seed(0)
    model, original_module = _wrapped_model(case)
    hidden_states = torch.randn(3, case["hidden_size"], dtype=torch.float32)

    with torch.no_grad():
        if case["label"] == "zamba2":
            expected = original_module(hidden_states.clone(), layer_idx=0)
        else:
            expected = original_module(hidden_states.clone())

    converted = convert_model(model)
    assert converted is True

    patched = _patched_module(model)
    assert hasattr(patched, "gate_proj")
    assert hasattr(patched, "up_proj")
    assert not hasattr(patched, "gate_up_proj")

    with torch.no_grad():
        if case["label"] == "zamba2":
            actual = patched(hidden_states.clone(), layer_idx=0)
        else:
            actual = patched(hidden_states.clone())

    torch.testing.assert_close(actual, expected)


def test_runtime_model_patches_respect_max_layers():
    module0 = _build_module(next(case for case in DENSE_CASES if case["label"] == "phi3"))
    module1 = _build_module(next(case for case in DENSE_CASES if case["label"] == "phi3"))

    class TwoLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="phi3")
            self.layers = nn.ModuleList([_DummyLayer(module0), _DummyLayer(module1)])

    model = TwoLayerModel()
    converted = convert_model(model, max_layers=1)

    assert converted is True
    assert hasattr(model.layers[0].module, "gate_proj")
    assert not hasattr(model.layers[0].module, "gate_up_proj")
    assert hasattr(model.layers[1].module, "gate_up_proj")
    assert not hasattr(model.layers[1].module, "gate_proj")
