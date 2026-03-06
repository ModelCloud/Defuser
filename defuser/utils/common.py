# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# Adapted from intel/auto-round
# at https://github.com/intel/auto-round/blob/main/auto_round/utils/common.py

from functools import lru_cache


@lru_cache(None)
def is_transformers_version_greater_or_equal_5():
    import transformers
    from packaging import version

    return version.parse(transformers.__version__) >= version.parse("5.0.0")
