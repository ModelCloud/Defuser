# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from defuser.utils.common import env_flag, warn_if_public_api_transformers_unsupported
from logbar import LogBar

DEBUG_ON = env_flag("DEBUG")

logger = LogBar(__name__)


def convert_model(*args, **kwargs) -> bool:
    """Lazily import conversion entrypoint to avoid import-time cycles."""
    if warn_if_public_api_transformers_unsupported("convert_model()", logger):
        return False

    from .defuser import convert_model as _convert_model

    return _convert_model(*args, **kwargs)


def replace_fused_blocks(*args, **kwargs) -> bool:
    """Lazily import conversion entrypoint to avoid import-time cycles."""
    if warn_if_public_api_transformers_unsupported("replace_fused_blocks()", logger):
        return False

    from .defuser import replace_fused_blocks as _replace_fused_blocks

    return _replace_fused_blocks(*args, **kwargs)



__all__ = ["convert_model", "replace_fused_blocks"]
