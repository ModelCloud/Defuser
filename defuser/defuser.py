# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from torch import nn

from defuser.utils.hf import apply_modeling_patch
from defuser.modeling.fused_moe.update_module import update_module

def convert_hf_model(model: nn.Module):
    # Patch modeling structure for legacy Qwen3 MoE
    is_applied = apply_modeling_patch(model)
    if not is_applied:
        # For Qwen3.5 MoE: defuse fused modules and tensors
        model = update_module(model, cleanup_original=True)

__all__ = ["convert_hf_model"]
