# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from enum import Enum

from transformers.core_model_loading import WeightConverter, WeightRenaming

from defuser.checkpoint_ops import OwnedChunk, SplitFusedExpertDownProj, SplitFusedExpertGateUpProj
from defuser.utils.common import MIN_SUPPORTED_TRANSFORMERS_VERSION


class PATCH(str, Enum):
    REPLACE_MODULE = "replace_module"
    EXPERTS_DEFUSE = "experts_defuse"


MODEL_CONFIG = {
    "dbrx": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "deepseek_v2": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "deepseek_v3": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "dia": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "dots1": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "ernie4_5_moe": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "ernie4_5_vl_moe": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "exaone_moe": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "flex_olmo": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "glm": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "glm4": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "mixtral": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
        PATCH.REPLACE_MODULE: [
            (
                "transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock",
                "defuser.modeling.unfused_moe.mixtral.LinearMixtralSparseMoeBlock",
            )
        ],
        "checkpoint_mapping": [
            WeightRenaming(".block_sparse_moe.", ".mlp."),
            WeightRenaming(r".experts.(\d+).w1.weight", r".experts.\1.gate_proj.weight"),
            WeightRenaming(r".experts.(\d+).w2.weight", r".experts.\1.down_proj.weight"),
            WeightRenaming(r".experts.(\d+).w3.weight", r".experts.\1.up_proj.weight"),
            WeightConverter(
                source_patterns=".experts.gate_up_proj",
                target_patterns=[
                    ".experts.0.gate_proj.weight",
                    ".experts.0.up_proj.weight",
                ],
                operations=[SplitFusedExpertGateUpProj()],
            ),
            WeightConverter(
                source_patterns=".experts.down_proj",
                target_patterns=".experts.0.down_proj.weight",
                operations=[SplitFusedExpertDownProj()],
            ),
        ],
    },
    "qwen2_moe": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
        PATCH.REPLACE_MODULE: [
            (
                "transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeSparseMoeBlock",
                "defuser.modeling.unfused_moe.qwen2_moe.LinearQwen2MoeSparseMoeBlock",
            )
        ],
    },
    "qwen3_moe": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
        # structure path only replaces modeling structure
        PATCH.REPLACE_MODULE: [
            (
                "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock",
                "defuser.modeling.unfused_moe.qwen3_moe.LinearQwen3MoeSparseMoeBlock",
            )
        ],
    },
    "qwen3_5_moe": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "qwen3_5_moe_text": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "qwen3_next": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
        PATCH.REPLACE_MODULE: [
            (
                "transformers.models.qwen3_next.modeling_qwen3_next.Qwen3NextSparseMoeBlock",
                "defuser.modeling.unfused_moe.qwen3_next.LinearQwen3NextSparseMoeBlock",
            )
        ],
    },
    "qwen3_omni_moe": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
        PATCH.REPLACE_MODULE: [
            (
                "transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe.Qwen3OmniMoeThinkerTextSparseMoeBlock",
                "defuser.modeling.unfused_moe.qwen3_omni_moe.LinearQwen3OmniMoeThinkerTextSparseMoeBlock",
            ),
            (
                "transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe.Qwen3OmniMoeTalkerTextSparseMoeBlock",
                "defuser.modeling.unfused_moe.qwen3_omni_moe.LinearQwen3OmniMoeTalkerTextSparseMoeBlock",
            )
        ],
    },
    "glm4_moe": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
        PATCH.REPLACE_MODULE: [
            (
                "transformers.models.glm4_moe.modeling_glm4_moe.Glm4MoeMoE",
                "defuser.modeling.unfused_moe.glm4_moe.LinearGlm4MoeMoE",
            )
        ],
    },
    "glm4_moe_lite": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
        PATCH.REPLACE_MODULE: [
            (
                "transformers.models.glm4_moe_lite.modeling_glm4_moe_lite.Glm4MoeLiteMoE",
                "defuser.modeling.unfused_moe.glm4_moe_lite.LinearGlm4MoeLiteMoE",
            )
        ],
    },
    "glm4v": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
        PATCH.REPLACE_MODULE: [
            (
                "transformers.models.glm4v.modeling_glm4v.Glm4vTextMLP",
                "defuser.modeling.glm4v.LinearGlm4vTextMLP",
            )
        ],
        # Split HF checkpoints that still store `gate_up_proj` as one fused tensor.
        "checkpoint_mapping": [
            WeightConverter(
                source_patterns="mlp.gate_up_proj.weight",
                target_patterns=[
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                ],
                operations=[OwnedChunk(dim=0)],
            ),
        ],
    },
    "glm4v_moe": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "glm_image": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "glm_moe_dsa": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "glm_ocr": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "gpt_oss": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "granitemoe": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "granitemoehybrid": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "granitemoeshared": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "hunyuan_v1_moe": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "jamba": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "jetmoe": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "laguna": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "llama4": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
        PATCH.EXPERTS_DEFUSE: [
            {
                "module_class": "transformers.models.llama4.modeling_llama4.Llama4TextExperts",
                "forward_impl": "batched_input",
            }
        ],
    },
    "lfm2_moe": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "longcat_flash": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "minimax": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "minimax_m2": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "nemotron_h": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "olmoe": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "phi3": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "phi4_multimodal": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "phimoe": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "qwen3_vl_moe": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "solar_open": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
    "zamba2": {
        "min_transformers_version": MIN_SUPPORTED_TRANSFORMERS_VERSION,
    },
}
