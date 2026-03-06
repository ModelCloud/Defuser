# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from enum import Enum


class PATCH(str, Enum):
    DEFUSE = "defuse"
    REPLACE_MODULE = "replace_module"


MODEL_CONFIG = {
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
        # Replacement module path imported only when the defuse workflow runs
        PATCH.DEFUSE: "defuser.modeling.fused_moe.qwen3_5_moe",
    },
    "qwen3_5_moe_text": {
        "min_transformers_version": "5.2.0",
        # Replacement module path imported only when the defuse workflow runs
        PATCH.DEFUSE: "defuser.modeling.fused_moe.qwen3_5_moe",
    },
}
