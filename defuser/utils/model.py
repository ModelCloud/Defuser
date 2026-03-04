# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch


def _update_parameter(
    module: torch.nn.Module,
    name: str,
    data: torch.Tensor,
) -> None:
    old_param = getattr(module, name)
    new_param = torch.nn.Parameter(data, requires_grad=old_param.requires_grad)
    setattr(module, name, new_param)
