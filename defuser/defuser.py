# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
import importlib
from copy import deepcopy

from torch import nn

from defuser.model_registry import MODEL_CONFIG, PATCH
from defuser.modeling.model_patches import apply_model_class_patches, apply_model_patches
from defuser.modeling.update_module import update_module
from defuser.utils.common import (
    MIN_SUPPORTED_TRANSFORMERS_VERSION,
    is_supported_transformers_version,
    warn_if_public_api_transformers_unsupported,
)
from packaging import version
import transformers
from logbar import LogBar

logger = LogBar(__name__)

def get_checkpoint_conversion_mapping(model_type):
    """Return Defuser's checkpoint remapping rules for one registered model type."""
    from transformers import conversion_mapping

    if not hasattr(conversion_mapping, "orig_get_checkpoint_conversion_mapping"):
        conversion_mapping.orig_get_checkpoint_conversion_mapping = conversion_mapping.get_checkpoint_conversion_mapping

    cfg = MODEL_CONFIG.get(model_type)
    if cfg and "checkpoint_mapping" in cfg:
        return deepcopy(cfg["checkpoint_mapping"])

    return conversion_mapping.orig_get_checkpoint_conversion_mapping(model_type)


class PatchError(Exception):
    pass


def replace_fused_blocks(model_type: str) -> bool:
    """Patch supported HF model classes so future loads instantiate defused blocks."""
    if warn_if_public_api_transformers_unsupported("replace_fused_blocks()", logger):
        return False

    apply_model_class_patches(model_type)

    cfg = MODEL_CONFIG.get(model_type)
    if cfg is None:
        return False

    patched_any = False
    for orig_path, custom_path in cfg.get(PATCH.REPLACE_MODULE, []):
        orig_module_path, orig_class_name = orig_path.rsplit(".", 1)
        custom_module_path, custom_class_name = custom_path.rsplit(".", 1)

        try:
            orig_module = importlib.import_module(orig_module_path)
            custom_module = importlib.import_module(custom_module_path)
            print("orig_module", orig_module, orig_class_name)
            # Validate class existence before patching
            if not hasattr(orig_module, orig_class_name):
                raise PatchError(f"Original class[{orig_class_name}] not found: {orig_module}")

            if not hasattr(custom_module, custom_class_name):
                raise PatchError(f"Custom class[{custom_class_name}] not found: {custom_module}")

            custom_class = getattr(custom_module, custom_class_name)
            setattr(orig_module, orig_class_name, custom_class)

            if version.parse(transformers.__version__) >= version.parse(MIN_SUPPORTED_TRANSFORMERS_VERSION):
                from transformers import conversion_mapping

                if not hasattr(conversion_mapping, "orig_get_checkpoint_conversion_mapping"):
                    conversion_mapping.orig_get_checkpoint_conversion_mapping = (
                        conversion_mapping.get_checkpoint_conversion_mapping
                    )

                conversion_mapping.get_checkpoint_conversion_mapping = get_checkpoint_conversion_mapping
                transformers.modeling_utils.get_checkpoint_conversion_mapping = get_checkpoint_conversion_mapping
            logger.info(f"Patched {orig_path} -> {custom_path}")
            patched_any = True

        except Exception as e:
            if isinstance(e, PatchError):
                raise e

            logger.warning(f"Failed to patch {orig_path}: {e}")
            return False
    return patched_any


def check_model_compatibility(model: nn.Module) -> bool:
    """Validate model type and transformers version compatibility."""
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    if model_type not in MODEL_CONFIG:
        return False

    if not is_supported_transformers_version():
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
) -> bool:
    """Convert one loaded model in place from fused experts to defused modules."""
    if warn_if_public_api_transformers_unsupported("convert_model()", logger):
        return False

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
        return False

    apply_model_patches(model, max_layers=max_layers)

    # If fused blocks have already been structurally replaced at load model before,
    # there is no need to perform runtime defusing again
    if MODEL_CONFIG[model.config.model_type].get(PATCH.REPLACE_MODULE):
        return False

    # Perform runtime defusing of fused projections
    # Split already-loaded fused modules (e.g., gate_up_proj/down_proj) into
    # independent expert layers: gate_proj / up_proj / down_proj
    update_module(
        model,
        cleanup_original=cleanup_original,
        max_layers=max_layers,
    )

    return True

__all__ = ["convert_model", "replace_fused_blocks"]
