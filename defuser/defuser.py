# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from torch import nn

from defuser.utils.hf import apply_modeling_patch


def convert_hf_model(model: nn.Module) -> bool:
    return apply_modeling_patch(model)

__all__ = ["convert_hf_model"]
