# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

def convert_hf_model(*args, **kwargs):
    """Lazily import conversion entrypoint to avoid import-time cycles."""
    from .defuser import convert_hf_model as _convert_hf_model

    return _convert_hf_model(*args, **kwargs)


__all__ = ["convert_hf_model"]
