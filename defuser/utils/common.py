# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
import os
# Adapted from intel/auto-round
# at https://github.com/intel/auto-round/blob/main/auto_round/utils/common.py

from functools import lru_cache
import re


# Match module paths like "...layers.0..." and capture the numeric layer index.
_LAYER_NAME_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")

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


@lru_cache(None)
def is_transformers_version_greater_or_equal_5():
    import transformers
    from packaging import version

    return version.parse(transformers.__version__) >= version.parse("5.0.0")


def is_within_max_layers(module_name: str, max_layers: int | None) -> bool:
    """Return True when module path is within requested layer limit."""
    if max_layers is None:
        return True
    if max_layers < 1:
        return False

    match = _LAYER_NAME_RE.search(module_name)
    if match is None:
        return True
    return int(match.group(1)) < max_layers
