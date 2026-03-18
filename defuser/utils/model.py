# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# Adapted from intel/auto-round
# at https://github.com/intel/auto-round/blob/main/auto_round/utils/model.py

import torch


def _update_parameter(
        module: torch.nn.Module,
        name: str,
        data: torch.Tensor,
) -> None:
    """Replace one module parameter while preserving its ``requires_grad`` flag."""
    old_param = getattr(module, name)
    new_param = torch.nn.Parameter(data, requires_grad=old_param.requires_grad)
    setattr(module, name, new_param)


def unsupported_meta_device(model):
    """Return ``True`` when mixed real/meta parameters make lazy materialization unsafe."""
    target_device = None
    for param in model.parameters():
        if target_device is None:
            target_device = param.device
        if param.device != target_device:
            if param.device.type == "meta" or target_device.type == "meta":
                return True
    if target_device.type == "meta":
        if hasattr(model, "path"):
            return False
        else:
            return True
    return False
