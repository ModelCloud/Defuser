# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from defuser.modeling.fused_moe.replace_modules import apply_replacements, release_original_module_


def update_module(
    model, cleanup_original: bool = True
):
    model = apply_replacements(model)

    if cleanup_original:
        release_original_module_(model)

    return model
