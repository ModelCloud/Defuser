# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

def clear_memory(
    tensor: torch.Tensor | list[torch.Tensor] | None = None,
    device_list: tuple | list | str | torch.device | None = None,
):
    # ------------------------
    # Clear CPU-side references
    # ------------------------
    if isinstance(tensor, list):
        for i in range(len(tensor)):
            tensor[i] = None
    tensor = None
    gc.collect()

    # ------------------------
    # Normalize device_list
    # ------------------------
    if isinstance(device_list, (str, torch.device)):
        device_list = [device_list]

    # -----------------------------------
    # CUDA-specific clearing
    # -----------------------------------
    if torch.cuda.is_available():
        # No device_list → clear all GPUs
        if not device_list:
            # Fix https://github.com/intel/auto-round/issues/1004
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        else:
            # Parse valid CUDA device IDs
            devices = []
            for dev in device_list:
                dev = str(dev)
                if not dev.startswith("cuda"):
                    continue
                # cuda / cuda:0 / cuda:1
                if ":" in dev:
                    devid = int(dev.split(":")[-1])
                else:
                    devid = 0
                devices.append(devid)

            for d in devices:
                torch.cuda.synchronize(d)

            torch.cuda.empty_cache()

    # -----------------------------------
    # XPU-specific clearing
    # -----------------------------------
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()
        torch.xpu.empty_cache()

def unsupported_meta_device(model):
    """Checks if the model is a valid model.

    Args:
    model: The model to be checked.

    Returns:
    bool: True if the model is valid, False otherwise.
    """
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
