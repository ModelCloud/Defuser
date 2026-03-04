# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from transformers import AutoModelForCausalLM

from defuser import convert_hf_model


def test_qwen3_moe():
    from defuser.modeling.unfused_moe.qwen3_moe import LinearQwen3MoeSparseMoeBlock

    model = AutoModelForCausalLM.from_pretrained("/monster/data/model/Qwen3-30B-A3B")

    assert model.config.model_type == "qwen3_moe"

    converted = convert_hf_model(model)
    assert converted
    assert isinstance(model.model.layers[0].mlp, LinearQwen3MoeSparseMoeBlock)
