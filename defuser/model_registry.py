# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from enum import Enum


class PATCH(str, Enum):
    REPLACE_MODULE = "replace_module"


MODEL_CONFIG = {
    "mixtral": {
        "min_transformers_version": "5.0.0",
        PATCH.REPLACE_MODULE: [
            (
                "transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock",
                "defuser.modeling.unfused_moe.mixtral.LinearMixtralSparseMoeBlock",
            )
        ],
    },
    "qwen2_moe": {
        "min_transformers_version": "5.0.0",
        PATCH.REPLACE_MODULE: [
            (
                "transformers.models.qwen2_moe.modeling_qwen2_moe.Qwen2MoeSparseMoeBlock",
                "defuser.modeling.unfused_moe.qwen2_moe.LinearQwen2MoeSparseMoeBlock",
            )
        ],
    },
    "qwen3_moe": {
        "min_transformers_version": "5.0.0",
        # structure path only replaces modeling structure
        PATCH.REPLACE_MODULE: [
            (
                "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock",
                "defuser.modeling.unfused_moe.qwen3_moe.LinearQwen3MoeSparseMoeBlock",
            )
        ],
    },
    "qwen3_5_moe": {
        "min_transformers_version": "5.2.0",
    },
    "qwen3_5_moe_text": {
        "min_transformers_version": "5.2.0",
    },
    "qwen3_next": {
        "min_transformers_version": "5.0.0",
        PATCH.REPLACE_MODULE: [
            (
                "transformers.models.qwen3_next.modeling_qwen3_next.Qwen3NextSparseMoeBlock",
                "defuser.modeling.unfused_moe.qwen3_next.LinearQwen3NextSparseMoeBlock",
            )
        ],
    },
    "qwen3_omni_moe": {
        "min_transformers_version": "5.0.0",
        PATCH.REPLACE_MODULE: [
            (
                "transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe.Qwen3OmniMoeThinkerTextSparseMoeBlock",
                "defuser.modeling.unfused_moe.qwen3_omni_moe.LinearQwen3OmniMoeThinkerTextSparseMoeBlock",
            )
        ],
    },
    "glm4_moe": {
        "min_transformers_version": "5.0.0",
        PATCH.REPLACE_MODULE: [
            (
                "transformers.models.glm4_moe.modeling_glm4_moe.Glm4MoeMoE",
                "defuser.modeling.unfused_moe.glm4_moe.LinearGlm4MoeMoE",
            )
        ],
    },
}
