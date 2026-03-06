# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# Adapted from intel/auto-round
# at https://github.com/intel/auto-round/blob/main/auto_round/modeling/unfused_moe/__init__.py

import torch

from packaging import version

from transformers import AutoConfig
import transformers
import os
import importlib

from typing import Final

from defuser.logger import logger
from defuser.utils.common import (
    LazyImport,
)

MODEL_CONFIG = {
    "qwen3_moe": {
        "min_transformers_version": "5.0.0",
        # structure_patch only replaces modeling structure
        "structure_patch": [
            (
                "transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock",
                "defuser.modeling.unfused_moe.qwen3_moe.LinearQwen3MoeSparseMoeBlock",
            )
        ],
    },
    "qwen3_5_moe": {
        "min_transformers_version": "5.2.0",
        # defuse_patch include tensor defusing/materialization workflow
        "defuse_patch": LazyImport("defuser.modeling.fused_moe.qwen3_5_moe"),
    },
    "qwen3_5_moe_text": {
        "min_transformers_version": "5.2.0",
        # defuse_patch include tensor defusing/materialization workflow
        "defuse_patch": LazyImport("defuser.modeling.fused_moe.qwen3_5_moe"),
    },

}

_ENV_VAR: Final[str] = "GPTQMODEL_USE_MODELSCOPE"


TRUTHFUL = {"1", "true", "yes", "on", "y"}


def env_flag(name: str, default: str | bool | None = "0") -> bool:
    """Return ``True`` when an env var is set to a truthy value."""

    value = os.getenv(name)
    if value is None:
        if default is None:
            return False
        if isinstance(default, bool):
            return default
        value = default
    return str(value).strip().lower() in TRUTHFUL

def modelscope_requested() -> bool:
    """
    Return ``True`` when the user explicitly enabled ModelScope integration
    via the GPTQMODEL_USE_MODELSCOPE environment variable.
    """
    return env_flag(_ENV_VAR, default="0")


def get_file_path_via_model_name(model_or_path: str, file_name):
    from huggingface_hub import hf_hub_download

    # 1) local folder
    if os.path.isdir(model_or_path):
        index_path = os.path.join(model_or_path, file_name)

    # 2) HF model name
    elif not modelscope_requested():
        index_path = hf_hub_download(
            repo_id=model_or_path,
            filename=file_name,
            repo_type="model",
        )
    elif modelscope_requested():
        from modelscope import snapshot_download  # pylint: disable=E0401

        # ModelSCOPE is different, it returns the folder path
        folder = snapshot_download(model_or_path, allow_patterns=[file_name])
        index_path = os.path.join(folder, file_name)
    else:
        index_path = None

    return index_path


def pre_check_config(model_name: str | torch.nn.Module):
    if isinstance(model_name, str):
        config = AutoConfig.from_pretrained(model_name)
    elif isinstance(model_name, torch.nn.Module):
        config = getattr(model_name, "config", None)
        if config is None:
            return False

    model_type = getattr(config, "model_type", None)
    if model_type is None or model_type not in MODEL_CONFIG:
        return False

    cfg = MODEL_CONFIG[model_type]

    min_ver = cfg.get("min_transformers_version")
    tf_ver = version.parse(transformers.__version__)
    if min_ver and tf_ver < version.parse(min_ver):
        return False
    try:
        file_path = get_file_path_via_model_name(model_name, "model.safetensors.index.json")
        if os.path.exists(file_path):
            import json

            with open(file_path, "r") as f:
                index_data = json.load(f)
            model_keys = list(index_data.get("weight_map", {}).keys())
            for key in model_keys:
                if "gate_up_proj" in key:
                    return False
    except:
        return True
    return True


def patch(model: torch.nn.Module) -> bool:
    res = pre_check_config(model)
    if not res:
        return False
    model_type = getattr(model.config, "model_type")
    cfg = MODEL_CONFIG[model_type]
    # patch blocks
    for orig_path, custom_path in cfg.get("structure_patch", []):
        orig_module_path, orig_class_name = orig_path.rsplit(".", 1)
        custom_module_path, custom_class_name = custom_path.rsplit(".", 1)
        try:
            orig_module = importlib.import_module(orig_module_path)
            custom_module = importlib.import_module(custom_module_path)
            custom_class = getattr(custom_module, custom_class_name)
            orig_class = getattr(orig_module, orig_class_name)
            names = []
            for n, m in model.named_modules():
                if isinstance(m, orig_class):
                    names.append((n, next(m.parameters()).dtype))
            for (n, orig_dtype) in names:
                model.set_submodule(n, custom_class(model.config).to(orig_dtype), True)
            logger.info(f"Patched module: {orig_path} -> {custom_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to patch {orig_path}: {e}")
            return False
    return False



