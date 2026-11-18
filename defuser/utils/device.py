# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# Adapted from intel/auto-round
# at https://github.com/intel/auto-round/blob/main/auto_round/utils/device.py

import gc

import torch


def to_meta(
        obj: torch.nn.Module | torch.nn.Parameter | torch.Tensor,
) -> torch.nn.Module | torch.nn.Parameter | torch.Tensor:
    """Move module/parameter/tensor storage to meta without copying values."""
    if isinstance(obj, torch.nn.Module):
        obj.to_empty(device="meta")
        return obj

    if isinstance(obj, torch.nn.Parameter):
        return torch.nn.Parameter(
            torch.empty_like(obj.data, device="meta"),
            requires_grad=obj.requires_grad,
        )

    if isinstance(obj, torch.Tensor):
        return torch.empty_like(obj, device="meta")

    raise TypeError(f"Unsupported type for to_meta: {type(obj)!r}")


def model_accelerator_devices(module: torch.nn.Module) -> list[torch.device]:
    """Return the distinct non-CPU, non-meta devices holding ``module``'s tensors."""
    devices: list[torch.device] = []
    for tensor in (*module.parameters(), *module.buffers()):
        device = tensor.device
        if device.type in ("cpu", "meta") or device in devices:
            continue
        devices.append(device)
    return devices


def _accelerator_ordinals(device_list, device_type: str, current_device) -> list[int]:
    ordinals: list[int] = []
    for dev in device_list:
        dev = torch.device(dev)
        if dev.type != device_type:
            continue
        index = dev.index if dev.index is not None else current_device()
        if index not in ordinals:
            ordinals.append(index)
    return ordinals


def clear_memory(
        tensor: torch.Tensor | list[torch.Tensor] | None = None,
        device_list: tuple | list | str | torch.device | None = None,
):
    """Release Python references and flush accelerator caches after tensor surgery.

    ``device_list=None`` synchronizes every accelerator this process has already
    initialized. An explicit ``device_list`` synchronizes only the listed CUDA/XPU
    devices; a list with no accelerator entries (for example ``[]`` or ``["cpu"]``)
    synchronizes nothing. An accelerator runtime that was never initialized is
    never touched, so cleaning up after CPU or meta work cannot create a device
    context (ModelCloud/GPTQModel#5105).
    """
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
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        if device_list is None:
            # Fix https://github.com/intel/auto-round/issues/1004
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        else:
            ordinals = _accelerator_ordinals(device_list, "cuda", torch.cuda.current_device)
            for ordinal in ordinals:
                torch.cuda.synchronize(ordinal)
            if ordinals:
                torch.cuda.empty_cache()

    # -----------------------------------
    # XPU-specific clearing
    # -----------------------------------
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available() and getattr(xpu, "is_initialized", lambda: True)():
        if device_list is None:
            xpu.synchronize()
            xpu.empty_cache()
        else:
            ordinals = _accelerator_ordinals(device_list, "xpu", xpu.current_device)
            for ordinal in ordinals:
                xpu.synchronize(ordinal)
            if ordinals:
                xpu.empty_cache()


def unsupported_meta_device(model):
    """Return ``True`` when a model mixes meta and real devices in unsupported ways."""
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
