# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from torch import nn

from defuser.model_registry import MODEL_CONFIG
from defuser.modeling.model_patches import apply_model_patches
from defuser.modeling.update_module import update_module
from packaging import version
import transformers
from logbar import LogBar

logger = LogBar(__name__)

def check_model_compatibility(model: nn.Module) -> bool:
    """Validate model type and transformers version compatibility."""
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    if model_type not in MODEL_CONFIG:
        return False

    min_ver = MODEL_CONFIG[model_type].get("min_transformers_version")
    current_ver = version.parse(transformers.__version__)
    if min_ver and current_ver < version.parse(min_ver):
        logger.warn(
            f"Skip conversion for model_type={model_type}: "
            f"requires transformers>={min_ver}, current version is {transformers.__version__}."
        )
        return False

    return True


def convert_model(
        model: nn.Module,
        cleanup_original: bool = False,
        max_layers: int | None = None,
) -> nn.Module:
    if max_layers is not None and max_layers < 1:
        raise ValueError("max_layers must be >= 1 when provided")

    # Patch modeling structure for legacy Qwen3 MoE
    #
    # There are two slightlyfis_within_max_layers different checkpoint formats we need to support:
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

    if not check_model_compatibility(model):
        return model

    apply_model_patches(model)

    return update_module(
        model,
        cleanup_original=cleanup_original,
        max_layers=max_layers,
    )

__all__ = ["convert_model"]
