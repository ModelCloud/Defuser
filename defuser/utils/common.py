# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
import os
# Adapted from intel/auto-round
# at https://github.com/intel/auto-round/blob/main/auto_round/utils/common.py

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from packaging import version
import pcre


# Match module paths like "...layers.0..." and capture the numeric layer index.
_LAYER_NAME_RE = pcre.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")

TRUTHFUL = {"1", "true", "yes", "on", "y"}
MIN_SUPPORTED_TRANSFORMERS_VERSION = "5.3.0"


@dataclass(frozen=True)
class ModuleNameFilter:
    """Compiled include or exclude rules for module path matching."""

    positive: tuple[pcre.Pattern, ...]
    negative: tuple[pcre.Pattern, ...]


def _parse_version(value: str | version.Version) -> version.Version:
    """Return a normalized packaging version object."""
    if isinstance(value, version.Version):
        return value
    return version.parse(value)


def is_version_at_least(
    installed_version: str | version.Version,
    minimum_version: str | version.Version,
) -> bool:
    """Return whether a version meets a minimum, allowing same-release dev snapshots.

    Hugging Face main-branch builds report versions like ``5.3.0-dev`` which
    packaging normalizes to ``5.3.0.dev0`` and orders before ``5.3.0``. Defuser
    treats those dev snapshots as satisfying the corresponding stable floor.
    """
    installed = _parse_version(installed_version)
    minimum = _parse_version(minimum_version)

    if installed >= minimum:
        return True

    if installed.is_devrelease:
        return version.parse(installed.base_version) >= minimum

    return False


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
    """Cache the coarse ``transformers>=5`` capability check used by fast paths."""
    import transformers

    return is_version_at_least(transformers.__version__, "5.0.0")


def is_supported_transformers_version() -> bool:
    """Return whether the installed transformers version is supported by Defuser's public API."""
    import transformers

    return is_version_at_least(transformers.__version__, MIN_SUPPORTED_TRANSFORMERS_VERSION)


def warn_if_public_api_transformers_unsupported(api_name: str, logger) -> bool:
    """Emit a single consistent warning when the runtime transformers version is too old."""
    import transformers

    if is_supported_transformers_version():
        return False

    logger.warning(
        f"Defuser public API `{api_name}` requires transformers>={MIN_SUPPORTED_TRANSFORMERS_VERSION}. "
        f"Current version is {transformers.__version__}. This call is unsupported and will be skipped."
    )
    return True


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


def compile_module_name_filter(
    filter_rules: Sequence[str] | ModuleNameFilter | None,
) -> ModuleNameFilter | None:
    """Compile user-facing module filter rules once for repeated matching.

    Rules support three forms:
    - ``+:regex`` explicit positive match
    - ``-:regex`` explicit negative match
    - ``regex`` implicit positive match

    Negative rules take priority over positive rules during matching.
    """
    if filter_rules is None:
        return None

    if isinstance(filter_rules, ModuleNameFilter):
        return filter_rules

    if isinstance(filter_rules, (str, bytes)) or not isinstance(filter_rules, Sequence):
        raise TypeError("filter must be a sequence of regex strings")

    positive: list[pcre.Pattern] = []
    negative: list[pcre.Pattern] = []
    for raw_rule in filter_rules:
        if not isinstance(raw_rule, str):
            raise TypeError("filter rules must be strings")

        if raw_rule.startswith("-:"):
            bucket = negative
            pattern = raw_rule[2:]
        elif raw_rule.startswith("+:"):
            bucket = positive
            pattern = raw_rule[2:]
        else:
            bucket = positive
            pattern = raw_rule

        bucket.append(pcre.compile(pattern))

    return ModuleNameFilter(
        positive=tuple(positive),
        negative=tuple(negative),
    )


def matches_module_name_filter(
    module_name: str,
    filter_rules: Sequence[str] | ModuleNameFilter | None,
) -> bool:
    """Return whether ``module_name`` is allowed by the configured filter rules."""
    compiled = compile_module_name_filter(filter_rules)
    if compiled is None:
        return True

    for pattern in compiled.negative:
        if pattern.search(module_name):
            return False

    for pattern in compiled.positive:
        if pattern.search(module_name):
            return True

    return False
