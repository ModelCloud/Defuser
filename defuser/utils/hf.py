# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# Adapted from intel/auto-round
# at https://github.com/intel/auto-round/blob/main/auto_round/modeling/unfused_moe/__init__.py

import importlib
import os
from typing import Final

import torch
import transformers
from logbar import LogBar
from packaging import version
from transformers import AutoConfig

from defuser.model_registry import MODEL_CONFIG

logger = LogBar(__name__)

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
