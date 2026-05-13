# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from importlib import import_module

import pytest
import torch

from defuser import convert_model, replace_fused_blocks
from defuser.model_registry import MODEL_CONFIG


def _load(module_path: str, attr_name: str):
    """Import one symbol lazily so the model case table stays readable."""
    return getattr(import_module(module_path), attr_name)


def _normalize_attr_value(obj, attr: str, value):
    """Preserve list-valued config fields when shrinking configs for meta tests."""
    current = getattr(obj, attr)
    if isinstance(current, list) and not isinstance(value, list):
        width = len(current) or 1
        return [value for _ in range(width)]
    return value


def _set_if_has(obj, **kwargs) -> None:
    """Apply overrides only to attributes exposed by the current config node."""
    for attr, value in kwargs.items():
        if hasattr(obj, attr):
            setattr(obj, attr, _normalize_attr_value(obj, attr, value))


def _mutate_common_config_tree(config, visited: set[int] | None = None) -> None:
    """Shrink nested configs recursively so meta-model construction stays lightweight."""
    if config is None or isinstance(config, (int, float, str, bool, list, tuple, dict)):
        return

    if visited is None:
        visited = set()
    if id(config) in visited:
        return
    visited.add(id(config))

    _set_if_has(
        config,
        vocab_size=128,
        hidden_size=64,
        hidden_dim=64,
        d_model=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        expert_ffn_hidden_size=32,
        ffn_hidden_size=128,
        num_hidden_layers=2,
        num_layers=2,
        n_layers=2,
        decoder_layers=2,
        encoder_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        num_heads=4,
        encoder_attention_heads=4,
        num_local_experts=4,
        num_experts=4,
        moe_num_experts=4,
        n_routed_experts=4,
        num_experts_per_tok=2,
        top_k=2,
        hidden_act="silu",
        activation="silu",
        activation_function="silu",
        mlp_hidden_act="silu",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        image_size=16,
        patch_size=4,
        num_channels=3,
        out_hidden_size=64,
        depth=1,
        num_position_embeddings=64,
        n_shared_experts=1,
        num_shared_experts=1,
        n_group=1,
        topk_group=1,
        use_bias=False,
        mlp_bias=False,
        add_bias_linear=False,
        zero_expert_num=1,
        adapter_rank=8,
        hybrid_layer_ids=[0],
        num_mem_blocks=1,
        dropout_rate=0.0,
        conv_chunksize=16,
        n_window=4,
        n_window_infer=4,
        max_source_positions=32,
        num_mel_bins=16,
        encoder_ffn_dim=128,
        output_dim=64,
        downsample_hidden_size=32,
        max_position_embeddings=64,
        initializer_range=0.02,
    )

    for attr in (
        "text_config",
        "vision_config",
        "audio_config",
        "decoder_config",
        "thinker_config",
        "talker_config",
        "language_config",
        "llm_config",
        "attn_config",
        "ffn_config",
        "code_predictor_config",
    ):
        _mutate_common_config_tree(getattr(config, attr, None), visited)

    thinker = getattr(config, "thinker_config", None)
    if thinker is not None:
        for attr in ("text_config", "vision_config", "audio_config"):
            _mutate_common_config_tree(getattr(thinker, attr, None), visited)

    talker = getattr(config, "talker_config", None)
    if talker is not None:
        for attr in ("text_config", "speech_config", "code_predictor_config"):
            _mutate_common_config_tree(getattr(talker, attr, None), visited)


def _build_model_config(case: dict):
    """Construct a small config tree for one registered public model type."""
    config = _load(case["config_module"], case["config_class"])()
    _mutate_common_config_tree(config)

    model_type = case["model_type"]
    if model_type == "dbrx":
        config.max_seq_len = 64
        config.resid_pdrop = 0.0
        config.emb_pdrop = 0.0
        config.attn_config.kv_n_heads = 1
        config.attn_config.clip_qkv = None
        config.attn_config.rope_theta = 10000.0
        config.ffn_config.ffn_hidden_size = 128
        config.ffn_config.moe_num_experts = 4
        config.ffn_config.moe_top_k = 2
        config.ffn_config.ffn_act_fn = {"name": "silu"}
    elif model_type == "deepseek_v3":
        config.first_k_dense_replace = 0
    elif model_type == "ernie4_5_vl_moe":
        config.text_config.moe_intermediate_size = [32, 32]
        config.text_config.rope_parameters = {
            "rope_theta": 10000.0,
            "rope_type": "default",
            "mrope_section": [2, 2, 4],
        }
    elif model_type == "glm4_moe":
        config.first_k_dense_replace = -1
    elif model_type == "glm_moe_dsa":
        config.mlp_layer_types = ["sparse"] * config.num_hidden_layers
    elif model_type == "granitemoehybrid":
        # Keep this meta-structure test on the attention path. The mamba path
        # lazy-loads optional hub kernels during construction, which is outside
        # the Defuser behavior being validated here.
        config.layer_types = ["attention", "attention"]
        config.shared_intermediate_size = 64
        config.mamba_n_heads = 8
    elif model_type == "jamba":
        # Keep this meta-structure test on the attention path. The mamba path
        # lazy-loads optional hub kernels during construction, which is outside
        # the Defuser behavior being validated here.
        config.attn_layer_period = 1
        config.attn_layer_offset = 0
        config.expert_layer_period = 1
        config.expert_layer_offset = 0
    elif model_type == "lfm2_moe":
        config.layer_types = ["full_attention", "short_conv"]
        config.num_dense_layers = 0
    elif model_type == "laguna":
        config.layer_types = ["full_attention"] * config.num_hidden_layers
        config.mlp_layer_types = ["dense"] + ["sparse"] * (config.num_hidden_layers - 1)
        config.num_attention_heads_per_layer = [config.num_attention_heads] * config.num_hidden_layers
    elif model_type == "nemotron_h":
        # Keep this meta-structure test on MoE blocks. Mamba blocks lazy-load
        # optional hub kernels during construction, which is outside the
        # Defuser behavior being validated here.
        config.layers_block_type = ["moe"] * config.num_hidden_layers
    elif model_type == "qwen3_omni_moe":
        config.enable_audio_output = True
        config.talker_config.spatial_merge_size = 2
        config.talker_config.thinker_hidden_size = 64
        config.talker_config.text_config.shared_expert_intermediate_size = 32
        code_predictor = getattr(config.talker_config, "code_predictor_config", None)
        _set_if_has(
            code_predictor,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=16,
        )
    elif model_type == "zamba2":
        # Zamba2 always constructs Mamba layers, even for hybrid blocks. Keep
        # the optional hub kernels disabled while preserving hybrid MLP targets.
        config.use_mamba_kernels = False
        config.layers_block_type = ["hybrid"] * config.num_hidden_layers
        config.hybrid_layer_ids = list(range(config.num_hidden_layers))

    return config


def _find_module_hits(model, class_paths: tuple[str, ...]) -> list[tuple[str, str]]:
    """Collect modules whose class path matches one of the expected targets."""
    hits = []
    wanted = set(class_paths)
    for name, module in model.named_modules():
        class_path = f"{module.__class__.__module__}.{module.__class__.__name__}"
        if class_path in wanted:
            hits.append((name, class_path))
    return hits


def _assert_meta_parameters(module) -> None:
    """Assert that one module keeps all of its parameters on meta."""
    for _, param in module.named_parameters(recurse=True):
        assert param.is_meta


def _assert_all_model_parameters_meta(model) -> None:
    """Assert that conversion does not materialize weights during meta tests."""
    for _, param in model.named_parameters():
        assert param.is_meta


def _validate_defused_module(case: dict, module) -> None:
    """Run the case-specific structural checks on one converted module."""
    kind = case["validator"]

    if kind == "experts":
        assert hasattr(module, "0")
        expert0 = getattr(module, "0")
        assert hasattr(expert0, "gate_proj")
        assert hasattr(expert0, "up_proj")
        assert hasattr(expert0, "down_proj")
        assert not hasattr(module, "gate_up_proj")
        _assert_meta_parameters(expert0)
        return

    if kind == "nongated_experts":
        assert hasattr(module, "0")
        expert0 = getattr(module, "0")
        assert hasattr(expert0, "up_proj")
        assert hasattr(expert0, "down_proj")
        assert not hasattr(expert0, "gate_proj")
        _assert_meta_parameters(expert0)
        return

    if kind == "parallel":
        assert hasattr(module, "0")
        assert hasattr(getattr(module, "0"), "linear")
        assert not hasattr(module, "weight")
        _assert_meta_parameters(getattr(module, "0"))
        return

    if kind == "dense_split":
        assert hasattr(module, "gate_proj")
        assert hasattr(module, "up_proj")
        assert hasattr(module, "down_proj")
        assert not hasattr(module, "gate_up_proj")
        _assert_meta_parameters(module)
        return

    if kind == "longcat":
        assert hasattr(module, "0")
        assert hasattr(getattr(module, "0"), "gate_proj")
        assert not hasattr(module, "gate_up_proj")
        assert any(hasattr(getattr(module, str(idx)), "identity") for idx in range(module.num_experts))
        _assert_meta_parameters(getattr(module, "0"))
        return

    if kind == "sparse_block":
        expert0 = module.experts[0]
        assert hasattr(expert0, "gate_proj")
        assert hasattr(expert0, "up_proj")
        assert hasattr(expert0, "down_proj")
        assert not hasattr(module.experts, "gate_up_proj")
        _assert_meta_parameters(expert0)
        return

    raise AssertionError(f"Unsupported validator kind: {kind}")


META_MODEL_CASES = [
    {
        "model_type": "dbrx",
        "mode": "convert",
        "model_module": "transformers.models.dbrx.modeling_dbrx",
        "model_class": "DbrxForCausalLM",
        "config_module": "transformers.models.dbrx.configuration_dbrx",
        "config_class": "DbrxConfig",
        "target_class_paths": ("transformers.models.dbrx.modeling_dbrx.DbrxExperts",),
        "validator": "experts",
    },
    {
        "model_type": "deepseek_v2",
        "mode": "convert",
        "model_module": "transformers.models.deepseek_v2.modeling_deepseek_v2",
        "model_class": "DeepseekV2ForCausalLM",
        "config_module": "transformers.models.deepseek_v2.configuration_deepseek_v2",
        "config_class": "DeepseekV2Config",
        "target_class_paths": ("transformers.models.deepseek_v2.modeling_deepseek_v2.DeepseekV2Experts",),
        "validator": "experts",
    },
    {
        "model_type": "deepseek_v3",
        "mode": "convert",
        "model_module": "transformers.models.deepseek_v3.modeling_deepseek_v3",
        "model_class": "DeepseekV3ForCausalLM",
        "config_module": "transformers.models.deepseek_v3.configuration_deepseek_v3",
        "config_class": "DeepseekV3Config",
        "target_class_paths": ("transformers.models.deepseek_v3.modeling_deepseek_v3.DeepseekV3NaiveMoe",),
        "validator": "experts",
        "min_targets": 2,
    },
    {
        "model_type": "deepseek_v4",
        "mode": "convert",
        "model_module": "transformers.models.deepseek_v4.modeling_deepseek_v4",
        "model_class": "DeepseekV4ForCausalLM",
        "config_module": "transformers.models.deepseek_v4.configuration_deepseek_v4",
        "config_class": "DeepseekV4Config",
        "target_class_paths": ("transformers.models.deepseek_v4.modeling_deepseek_v4.DeepseekV4Experts",),
        "validator": "experts",
        "min_targets": 2,
    },
    {
        "model_type": "dia",
        "mode": "convert",
        "model_module": "transformers.models.dia.modeling_dia",
        "model_class": "DiaForConditionalGeneration",
        "config_module": "transformers.models.dia.configuration_dia",
        "config_class": "DiaConfig",
        "target_class_paths": ("transformers.models.dia.modeling_dia.DiaMLP",),
        "validator": "dense_split",
    },
    {
        "model_type": "dots1",
        "mode": "convert",
        "model_module": "transformers.models.dots1.modeling_dots1",
        "model_class": "Dots1ForCausalLM",
        "config_module": "transformers.models.dots1.configuration_dots1",
        "config_class": "Dots1Config",
        "target_class_paths": ("transformers.models.dots1.modeling_dots1.Dots1NaiveMoe",),
        "validator": "experts",
    },
    {
        "model_type": "ernie4_5_moe",
        "mode": "convert",
        "model_module": "transformers.models.ernie4_5_moe.modeling_ernie4_5_moe",
        "model_class": "Ernie4_5_MoeForCausalLM",
        "config_module": "transformers.models.ernie4_5_moe.configuration_ernie4_5_moe",
        "config_class": "Ernie4_5_MoeConfig",
        "target_class_paths": ("transformers.models.ernie4_5_moe.modeling_ernie4_5_moe.Ernie4_5_MoeExperts",),
        "validator": "experts",
    },
    {
        "model_type": "ernie4_5_vl_moe",
        "mode": "convert",
        "model_module": "transformers.models.ernie4_5_vl_moe.modeling_ernie4_5_vl_moe",
        "model_class": "Ernie4_5_VLMoeForConditionalGeneration",
        "config_module": "transformers.models.ernie4_5_vl_moe.configuration_ernie4_5_vl_moe",
        "config_class": "Ernie4_5_VLMoeConfig",
        "target_class_paths": (
            "transformers.models.ernie4_5_vl_moe.modeling_ernie4_5_vl_moe.Ernie4_5_VLMoeMoeExperts",
        ),
        "validator": "experts",
    },
    {
        "model_type": "exaone_moe",
        "mode": "convert",
        "model_module": "transformers.models.exaone_moe.modeling_exaone_moe",
        "model_class": "ExaoneMoeForCausalLM",
        "config_module": "transformers.models.exaone_moe.configuration_exaone_moe",
        "config_class": "ExaoneMoeConfig",
        "target_class_paths": ("transformers.models.exaone_moe.modeling_exaone_moe.ExaoneMoeExperts",),
        "validator": "experts",
    },
    {
        "model_type": "flex_olmo",
        "mode": "convert",
        "model_module": "transformers.models.flex_olmo.modeling_flex_olmo",
        "model_class": "FlexOlmoForCausalLM",
        "config_module": "transformers.models.flex_olmo.configuration_flex_olmo",
        "config_class": "FlexOlmoConfig",
        "target_class_paths": ("transformers.models.flex_olmo.modeling_flex_olmo.FlexOlmoExperts",),
        "validator": "experts",
    },
    {
        "model_type": "glm",
        "mode": "convert",
        "model_module": "transformers.models.glm.modeling_glm",
        "model_class": "GlmForCausalLM",
        "config_module": "transformers.models.glm.configuration_glm",
        "config_class": "GlmConfig",
        "target_class_paths": ("transformers.models.glm.modeling_glm.GlmMLP",),
        "validator": "dense_split",
    },
    {
        "model_type": "glm4",
        "mode": "convert",
        "model_module": "transformers.models.glm4.modeling_glm4",
        "model_class": "Glm4ForCausalLM",
        "config_module": "transformers.models.glm4.configuration_glm4",
        "config_class": "Glm4Config",
        "target_class_paths": ("transformers.models.glm4.modeling_glm4.Glm4MLP",),
        "validator": "dense_split",
    },
    {
        "model_type": "glm4_moe",
        "mode": "replace",
        "model_module": "transformers.models.glm4_moe.modeling_glm4_moe",
        "model_class": "Glm4MoeForCausalLM",
        "config_module": "transformers.models.glm4_moe.configuration_glm4_moe",
        "config_class": "Glm4MoeConfig",
        "target_class_paths": ("defuser.modeling.unfused_moe.glm4_moe.LinearGlm4MoeMoE",),
        "validator": "sparse_block",
    },
    {
        "model_type": "glm4_moe_lite",
        "mode": "replace",
        "model_module": "transformers.models.glm4_moe_lite.modeling_glm4_moe_lite",
        "model_class": "Glm4MoeLiteForCausalLM",
        "config_module": "transformers.models.glm4_moe_lite.configuration_glm4_moe_lite",
        "config_class": "Glm4MoeLiteConfig",
        "target_class_paths": ("defuser.modeling.unfused_moe.glm4_moe_lite.LinearGlm4MoeLiteMoE",),
        "validator": "sparse_block",
    },
    {
        "model_type": "glm4v",
        "mode": "replace",
        "model_module": "transformers.models.glm4v.modeling_glm4v",
        "model_class": "Glm4vForConditionalGeneration",
        "config_module": "transformers.models.glm4v.configuration_glm4v",
        "config_class": "Glm4vConfig",
        "target_class_paths": ("defuser.modeling.glm4v.LinearGlm4vTextMLP",),
        "validator": "dense_split",
    },
    {
        "model_type": "glm4v_moe",
        "mode": "convert",
        "model_module": "transformers.models.glm4v_moe.modeling_glm4v_moe",
        "model_class": "Glm4vMoeForConditionalGeneration",
        "config_module": "transformers.models.glm4v_moe.configuration_glm4v_moe",
        "config_class": "Glm4vMoeConfig",
        "target_class_paths": ("transformers.models.glm4v_moe.modeling_glm4v_moe.Glm4vMoeTextNaiveMoe",),
        "validator": "experts",
    },
    {
        "model_type": "glm_image",
        "mode": "convert",
        "model_module": "transformers.models.glm_image.modeling_glm_image",
        "model_class": "GlmImageForConditionalGeneration",
        "config_module": "transformers.models.glm_image.configuration_glm_image",
        "config_class": "GlmImageConfig",
        "target_class_paths": ("transformers.models.glm_image.modeling_glm_image.GlmImageTextMLP",),
        "validator": "dense_split",
    },
    {
        "model_type": "glm_moe_dsa",
        "mode": "convert",
        "model_module": "transformers.models.glm_moe_dsa.modeling_glm_moe_dsa",
        "model_class": "GlmMoeDsaForCausalLM",
        "config_module": "transformers.models.glm_moe_dsa.configuration_glm_moe_dsa",
        "config_class": "GlmMoeDsaConfig",
        "target_class_paths": ("transformers.models.glm_moe_dsa.modeling_glm_moe_dsa.GlmMoeDsaNaiveMoe",),
        "validator": "experts",
        "min_targets": 2,
    },
    {
        "model_type": "glm_ocr",
        "mode": "convert",
        "model_module": "transformers.models.glm_ocr.modeling_glm_ocr",
        "model_class": "GlmOcrForConditionalGeneration",
        "config_module": "transformers.models.glm_ocr.configuration_glm_ocr",
        "config_class": "GlmOcrConfig",
        "target_class_paths": ("transformers.models.glm_ocr.modeling_glm_ocr.GlmOcrTextMLP",),
        "validator": "dense_split",
    },
    {
        "model_type": "gpt_oss",
        "mode": "convert",
        "model_module": "transformers.models.gpt_oss.modeling_gpt_oss",
        "model_class": "GptOssForCausalLM",
        "config_module": "transformers.models.gpt_oss.configuration_gpt_oss",
        "config_class": "GptOssConfig",
        "target_class_paths": ("transformers.models.gpt_oss.modeling_gpt_oss.GptOssExperts",),
        "validator": "experts",
        "min_targets": 2,
    },
    {
        "model_type": "granitemoe",
        "mode": "convert",
        "model_module": "transformers.models.granitemoe.modeling_granitemoe",
        "model_class": "GraniteMoeForCausalLM",
        "config_module": "transformers.models.granitemoe.configuration_granitemoe",
        "config_class": "GraniteMoeConfig",
        "target_class_paths": ("transformers.models.granitemoe.modeling_granitemoe.GraniteMoeParallelExperts",),
        "validator": "parallel",
        "min_targets": 4,
    },
    {
        "model_type": "granitemoehybrid",
        "mode": "convert",
        "model_module": "transformers.models.granitemoehybrid.modeling_granitemoehybrid",
        "model_class": "GraniteMoeHybridForCausalLM",
        "config_module": "transformers.models.granitemoehybrid.configuration_granitemoehybrid",
        "config_class": "GraniteMoeHybridConfig",
        "target_class_paths": (
            "transformers.models.granitemoehybrid.modeling_granitemoehybrid.GraniteMoeHybridParallelExperts",
        ),
        "validator": "parallel",
        "min_targets": 4,
    },
    {
        "model_type": "granitemoeshared",
        "mode": "convert",
        "model_module": "transformers.models.granitemoeshared.modeling_granitemoeshared",
        "model_class": "GraniteMoeSharedForCausalLM",
        "config_module": "transformers.models.granitemoeshared.configuration_granitemoeshared",
        "config_class": "GraniteMoeSharedConfig",
        "target_class_paths": (
            "transformers.models.granitemoeshared.modeling_granitemoeshared.GraniteMoeSharedParallelExperts",
        ),
        "validator": "parallel",
        "min_targets": 4,
    },
    {
        "model_type": "hunyuan_v1_moe",
        "mode": "convert",
        "model_module": "transformers.models.hunyuan_v1_moe.modeling_hunyuan_v1_moe",
        "model_class": "HunYuanMoEV1ForCausalLM",
        "config_module": "transformers.models.hunyuan_v1_moe.configuration_hunyuan_v1_moe",
        "config_class": "HunYuanMoEV1Config",
        "target_class_paths": ("transformers.models.hunyuan_v1_moe.modeling_hunyuan_v1_moe.HunYuanMoEV1Experts",),
        "validator": "experts",
    },
    {
        "model_type": "jamba",
        "mode": "convert",
        "model_module": "transformers.models.jamba.modeling_jamba",
        "model_class": "JambaForCausalLM",
        "config_module": "transformers.models.jamba.configuration_jamba",
        "config_class": "JambaConfig",
        "target_class_paths": ("transformers.models.jamba.modeling_jamba.JambaExperts",),
        "validator": "experts",
    },
    {
        "model_type": "jetmoe",
        "mode": "convert",
        "model_module": "transformers.models.jetmoe.modeling_jetmoe",
        "model_class": "JetMoeForCausalLM",
        "config_module": "transformers.models.jetmoe.configuration_jetmoe",
        "config_class": "JetMoeConfig",
        "target_class_paths": ("transformers.models.jetmoe.modeling_jetmoe.JetMoeParallelExperts",),
        "validator": "parallel",
        "min_targets": 4,
    },
    {
        "model_type": "laguna",
        "mode": "convert",
        "model_module": "transformers.models.laguna.modeling_laguna",
        "model_class": "LagunaForCausalLM",
        "config_module": "transformers.models.laguna.configuration_laguna",
        "config_class": "LagunaConfig",
        "target_class_paths": ("transformers.models.laguna.modeling_laguna.LagunaExperts",),
        "validator": "experts",
    },
    {
        "model_type": "lfm2_moe",
        "mode": "convert",
        "model_module": "transformers.models.lfm2_moe.modeling_lfm2_moe",
        "model_class": "Lfm2MoeForCausalLM",
        "config_module": "transformers.models.lfm2_moe.configuration_lfm2_moe",
        "config_class": "Lfm2MoeConfig",
        "target_class_paths": ("transformers.models.lfm2_moe.modeling_lfm2_moe.Lfm2MoeExperts",),
        "validator": "experts",
        "min_targets": 2,
    },
    {
        "model_type": "llama4",
        "mode": "convert",
        "model_module": "transformers.models.llama4.modeling_llama4",
        "model_class": "Llama4ForConditionalGeneration",
        "config_module": "transformers.models.llama4.configuration_llama4",
        "config_class": "Llama4Config",
        "target_class_paths": ("transformers.models.llama4.modeling_llama4.Llama4TextExperts",),
        "validator": "experts",
        "min_targets": 2,
    },
    {
        "model_type": "longcat_flash",
        "mode": "convert",
        "model_module": "transformers.models.longcat_flash.modeling_longcat_flash",
        "model_class": "LongcatFlashForCausalLM",
        "config_module": "transformers.models.longcat_flash.configuration_longcat_flash",
        "config_class": "LongcatFlashConfig",
        "target_class_paths": ("transformers.models.longcat_flash.modeling_longcat_flash.LongcatFlashExperts",),
        "validator": "longcat",
        "min_targets": 2,
    },
    {
        "model_type": "minimax",
        "mode": "convert",
        "model_module": "transformers.models.minimax.modeling_minimax",
        "model_class": "MiniMaxForCausalLM",
        "config_module": "transformers.models.minimax.configuration_minimax",
        "config_class": "MiniMaxConfig",
        "target_class_paths": ("transformers.models.minimax.modeling_minimax.MiniMaxExperts",),
        "validator": "experts",
    },
    {
        "model_type": "minimax_m2",
        "mode": "convert",
        "model_module": "transformers.models.minimax_m2.modeling_minimax_m2",
        "model_class": "MiniMaxM2ForCausalLM",
        "config_module": "transformers.models.minimax_m2.configuration_minimax_m2",
        "config_class": "MiniMaxM2Config",
        "target_class_paths": ("transformers.models.minimax_m2.modeling_minimax_m2.MiniMaxM2Experts",),
        "validator": "experts",
    },
    {
        "model_type": "mixtral",
        "mode": "replace",
        "model_module": "transformers.models.mixtral.modeling_mixtral",
        "model_class": "MixtralForCausalLM",
        "config_module": "transformers.models.mixtral.configuration_mixtral",
        "config_class": "MixtralConfig",
        "target_class_paths": ("defuser.modeling.unfused_moe.mixtral.LinearMixtralSparseMoeBlock",),
        "validator": "sparse_block",
    },
    {
        "model_type": "nemotron_h",
        "mode": "convert",
        "model_module": "transformers.models.nemotron_h.modeling_nemotron_h",
        "model_class": "NemotronHForCausalLM",
        "config_module": "transformers.models.nemotron_h.configuration_nemotron_h",
        "config_class": "NemotronHConfig",
        "target_class_paths": ("transformers.models.nemotron_h.modeling_nemotron_h.NemotronHExperts",),
        "validator": "nongated_experts",
        "min_targets": 2,
    },
    {
        "model_type": "olmoe",
        "mode": "convert",
        "model_module": "transformers.models.olmoe.modeling_olmoe",
        "model_class": "OlmoeForCausalLM",
        "config_module": "transformers.models.olmoe.configuration_olmoe",
        "config_class": "OlmoeConfig",
        "target_class_paths": ("transformers.models.olmoe.modeling_olmoe.OlmoeExperts",),
        "validator": "experts",
    },
    {
        "model_type": "phi3",
        "mode": "convert",
        "model_module": "transformers.models.phi3.modeling_phi3",
        "model_class": "Phi3ForCausalLM",
        "config_module": "transformers.models.phi3.configuration_phi3",
        "config_class": "Phi3Config",
        "target_class_paths": ("transformers.models.phi3.modeling_phi3.Phi3MLP",),
        "validator": "dense_split",
    },
    {
        "model_type": "phi4_multimodal",
        "mode": "convert",
        "model_module": "transformers.models.phi4_multimodal.modeling_phi4_multimodal",
        "model_class": "Phi4MultimodalForCausalLM",
        "config_module": "transformers.models.phi4_multimodal.configuration_phi4_multimodal",
        "config_class": "Phi4MultimodalConfig",
        "target_class_paths": (
            "transformers.models.phi4_multimodal.modeling_phi4_multimodal.Phi4MultimodalMLP",
            "transformers.models.phi4_multimodal.modeling_phi4_multimodal.Phi4MultimodalAudioMLP",
        ),
        "validator": "dense_split",
        "min_targets": 4,
    },
    {
        "model_type": "phimoe",
        "mode": "convert",
        "model_module": "transformers.models.phimoe.modeling_phimoe",
        "model_class": "PhimoeForCausalLM",
        "config_module": "transformers.models.phimoe.configuration_phimoe",
        "config_class": "PhimoeConfig",
        "target_class_paths": ("transformers.models.phimoe.modeling_phimoe.PhimoeExperts",),
        "validator": "experts",
        "min_targets": 2,
    },
    {
        "model_type": "qwen2_moe",
        "mode": "replace",
        "model_module": "transformers.models.qwen2_moe.modeling_qwen2_moe",
        "model_class": "Qwen2MoeForCausalLM",
        "config_module": "transformers.models.qwen2_moe.configuration_qwen2_moe",
        "config_class": "Qwen2MoeConfig",
        "target_class_paths": ("defuser.modeling.unfused_moe.qwen2_moe.LinearQwen2MoeSparseMoeBlock",),
        "validator": "sparse_block",
    },
    {
        "model_type": "qwen3_5_moe",
        "mode": "convert",
        "model_module": "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
        "model_class": "Qwen3_5MoeForConditionalGeneration",
        "config_module": "transformers.models.qwen3_5_moe.configuration_qwen3_5_moe",
        "config_class": "Qwen3_5MoeConfig",
        "target_class_paths": ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeExperts",),
        "validator": "experts",
        "min_targets": 2,
    },
    {
        "model_type": "qwen3_moe",
        "mode": "replace",
        "model_module": "transformers.models.qwen3_moe.modeling_qwen3_moe",
        "model_class": "Qwen3MoeForCausalLM",
        "config_module": "transformers.models.qwen3_moe.configuration_qwen3_moe",
        "config_class": "Qwen3MoeConfig",
        "target_class_paths": ("defuser.modeling.unfused_moe.qwen3_moe.LinearQwen3MoeSparseMoeBlock",),
        "validator": "sparse_block",
    },
    {
        "model_type": "qwen3_next",
        "mode": "replace",
        "model_module": "transformers.models.qwen3_next.modeling_qwen3_next",
        "model_class": "Qwen3NextForCausalLM",
        "config_module": "transformers.models.qwen3_next.configuration_qwen3_next",
        "config_class": "Qwen3NextConfig",
        "target_class_paths": ("defuser.modeling.unfused_moe.qwen3_next.LinearQwen3NextSparseMoeBlock",),
        "validator": "sparse_block",
    },
    {
        "model_type": "qwen3_omni_moe",
        "mode": "replace",
        "model_module": "transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe",
        "model_class": "Qwen3OmniMoeForConditionalGeneration",
        "config_module": "transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe",
        "config_class": "Qwen3OmniMoeConfig",
        "target_class_paths": (
            "defuser.modeling.unfused_moe.qwen3_omni_moe.LinearQwen3OmniMoeThinkerTextSparseMoeBlock",
            "defuser.modeling.unfused_moe.qwen3_omni_moe.LinearQwen3OmniMoeTalkerTextSparseMoeBlock",
        ),
        "validator": "sparse_block",
        "min_targets": 2,
    },
    {
        "model_type": "qwen3_vl_moe",
        "mode": "convert",
        "model_module": "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe",
        "model_class": "Qwen3VLMoeForConditionalGeneration",
        "config_module": "transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe",
        "config_class": "Qwen3VLMoeConfig",
        "target_class_paths": ("transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextExperts",),
        "validator": "experts",
        "min_targets": 2,
    },
    {
        "model_type": "solar_open",
        "mode": "convert",
        "model_module": "transformers.models.solar_open.modeling_solar_open",
        "model_class": "SolarOpenForCausalLM",
        "config_module": "transformers.models.solar_open.configuration_solar_open",
        "config_class": "SolarOpenConfig",
        "target_class_paths": ("transformers.models.solar_open.modeling_solar_open.SolarOpenNaiveMoe",),
        "validator": "experts",
    },
    {
        "model_type": "zamba2",
        "mode": "convert",
        "model_module": "transformers.models.zamba2.modeling_zamba2",
        "model_class": "Zamba2ForCausalLM",
        "config_module": "transformers.models.zamba2.configuration_zamba2",
        "config_class": "Zamba2Config",
        "target_class_paths": ("transformers.models.zamba2.modeling_zamba2.Zamba2MLP",),
        "validator": "dense_split",
    },
]


def test_meta_model_cases_cover_registered_public_models():
    """Meta-model coverage should stay aligned with the public registry."""
    assert {case["model_type"] for case in META_MODEL_CASES} == set(MODEL_CONFIG) - {"qwen3_5_moe_text"}


@pytest.mark.parametrize("case", META_MODEL_CASES, ids=[case["model_type"] for case in META_MODEL_CASES])
def test_each_model_defuses_direct_meta_model(case):
    """Each registered public model should expose the expected defused modules on meta."""
    if case["mode"] == "replace":
        replace_fused_blocks(case["model_type"])

    config = _build_model_config(case)
    model_cls = _load(case["model_module"], case["model_class"])
    with torch.device("meta"):
        model = model_cls(config)

    _assert_all_model_parameters_meta(model)

    hits = _find_module_hits(model, case["target_class_paths"])
    assert hits
    assert set(case["target_class_paths"]).issubset({class_path for _, class_path in hits})
    assert len(hits) >= case.get("min_targets", 1)

    if case["mode"] == "replace":
        for path, _ in hits:
            _validate_defused_module(case, model.get_submodule(path))
        assert convert_model(model) is False
        _assert_all_model_parameters_meta(model)
        return

    target_paths = [path for path, _ in hits]
    assert convert_model(model) is True
    _assert_all_model_parameters_meta(model)
    for path in target_paths:
        _validate_defused_module(case, model.get_submodule(path))
