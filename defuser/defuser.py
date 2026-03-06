# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from torch import nn

from defuser.utils.hf import apply_modeling_patch
from defuser.modeling.fused_moe.update_module import update_module


def convert_hf_model(model: nn.Module, cleanup_original: bool):
    # Patch modeling structure for legacy Qwen3 MoE
    #
    # There are two slightly different checkpoint formats we need to support:
    #   1) Qwen3 MoE
    #   2) Qwen3.5 MoE
    #
    # The key difference is how the expert MLP weights are stored in the original
    # checkpoint (fused vs. defused). Because of that, the amount of work needed
    # after replacing the modeling structure is different.
    #
    # ---------------------------------------------------------------------------
    # Step 1: Try applying a lightweight modeling patch
    # ---------------------------------------------------------------------------
    # `apply_modeling_patch(model)` only replaces the *modeling structure*
    # (module definitions / forward logic) to match our runtime implementation.
    #
    # For **Qwen3 MoE**, this is sufficient because:
    #   - The original checkpoint already stores `mlp.experts` weights in a
    #     **defused format**.
    #   - In other words, the tensors are already separated as:
    #
    #       gate_proj
    #       up_proj
    #       down_proj
    #
    #   - Therefore we only need to swap the modeling implementation so that the
    #     module structure matches the expected layout, without touching the
    #     underlying tensors.
    #
    # If this patch succeeds, it means the model is in the Qwen3 MoE format and
    # no further tensor transformation is required.
    is_applied = apply_modeling_patch(model)
    if not is_applied:
        # -----------------------------------------------------------------------
        # Step 2: Handle Qwen3.5 MoE checkpoints
        # -----------------------------------------------------------------------
        #
        # If `apply_modeling_patch` fails, we assume the checkpoint corresponds to
        # **Qwen3.5 MoE**.
        #
        # In Qwen3.5 MoE, the expert MLP weights are stored in a **fused format**.
        # Specifically, the checkpoint keeps tensors such as:
        #
        #     gate_up_proj
        #     down_proj
        #
        # where `gate_proj` and `up_proj` are fused together.
        #
        # Because our runtime modeling expects **defused tensors**, simply replacing
        # the module structure is not enough. We must also convert the stored
        # parameters.
        #
        # `update_module()` performs two tasks:
        #
        #   1) Replace the modeling structure so that it matches the expected
        #      defused MoE implementation.
        #
        #   2) Prepare the module for **tensor defusion** of the expert weights.
        #
        # After the structure update, `materialize_model_()` will be invoked to
        # actually split the fused tensors:
        #
        #     gate_up_proj  -->  gate_proj + up_proj
        #
        # and ensure the module finally contains the expected parameters:
        #
        #     gate_proj
        #     up_proj
        #     down_proj
        #
        # This ensures compatibility between the Qwen3.5 fused checkpoint format
        # and the runtime model implementation that operates on defused weights.
        model = update_module(model, cleanup_original=cleanup_original)

__all__ = ["convert_hf_model"]
