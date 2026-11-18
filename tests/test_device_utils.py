# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import pytest
import torch

from defuser.utils.device import to_meta


def test_to_meta_on_module_moves_parameters_to_meta():
    module = torch.nn.Linear(8, 4)

    to_meta(module)

    assert next(module.parameters()).is_meta


def test_to_meta_on_parameter_releases_parameter_storage():
    parameter = torch.nn.Parameter(torch.randn(3, 5))

    meta_parameter = to_meta(parameter)

    assert not parameter.is_meta
    assert meta_parameter.is_meta
    assert meta_parameter.requires_grad == parameter.requires_grad
    assert tuple(meta_parameter.shape) == (3, 5)


def test_to_meta_on_tensor_returns_meta_tensor():
    tensor = torch.randn(2, 6)

    meta_tensor = to_meta(tensor)

    assert not tensor.is_meta
    assert meta_tensor.is_meta
    assert tuple(meta_tensor.shape) == (2, 6)


def test_to_meta_rejects_unsupported_type():
    with pytest.raises(TypeError, match="Unsupported type for to_meta"):
        to_meta("not-a-tensor")


class _FakeAccelerator:
    """Records synchronize/empty_cache calls instead of touching real hardware."""

    def __init__(self, *, initialized, current=0):
        self.initialized = initialized
        self.current = current
        self.synced = []
        self.emptied = 0

    def is_available(self):
        return True

    def is_initialized(self):
        return self.initialized

    def current_device(self):
        return self.current

    def synchronize(self, device=None):
        self.synced.append(self.current if device is None else device)

    def empty_cache(self):
        self.emptied += 1


@pytest.fixture
def fake_cuda(monkeypatch):
    def install(**kwargs):
        fake = _FakeAccelerator(**kwargs)
        for name in ("is_available", "is_initialized", "current_device", "synchronize", "empty_cache"):
            monkeypatch.setattr(torch.cuda, name, getattr(fake, name))
        if hasattr(torch, "xpu"):
            monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
        return fake

    return install


def test_clear_memory_never_initializes_an_unused_accelerator(fake_cuda):
    from defuser.utils.device import clear_memory

    fake = fake_cuda(initialized=False)
    clear_memory()
    clear_memory(device_list=["cuda:1"])
    assert fake.synced == [] and fake.emptied == 0


@pytest.mark.parametrize("device_list", [[], ["cpu"], "cpu", ["meta"]])
def test_clear_memory_with_no_accelerator_devices_syncs_nothing(fake_cuda, device_list):
    from defuser.utils.device import clear_memory

    fake = fake_cuda(initialized=True)
    clear_memory(device_list=device_list)
    assert fake.synced == [] and fake.emptied == 0


def test_clear_memory_syncs_only_listed_cuda_ordinals(fake_cuda):
    from defuser.utils.device import clear_memory

    fake = fake_cuda(initialized=True, current=0)
    clear_memory(device_list=["cuda:1", torch.device("cuda", 2), "cuda:1", "cpu"])
    assert fake.synced == [1, 2]
    assert fake.emptied == 1


def test_clear_memory_bare_cuda_means_the_current_device(fake_cuda):
    from defuser.utils.device import clear_memory

    fake = fake_cuda(initialized=True, current=3)
    clear_memory(device_list="cuda")
    assert fake.synced == [3]


def test_clear_memory_default_still_flushes_an_initialized_runtime(fake_cuda):
    from defuser.utils.device import clear_memory

    fake = fake_cuda(initialized=True, current=0)
    clear_memory()
    assert fake.synced == [0] and fake.emptied == 1


def test_model_accelerator_devices_ignores_cpu_and_meta():
    from defuser.utils.device import model_accelerator_devices

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2, device="meta"))
    model.register_buffer("scale", torch.ones(1))
    assert model_accelerator_devices(model) == []


def test_moe_preparation_on_a_cpu_model_syncs_no_accelerator(fake_cuda, monkeypatch):
    """A CPU/meta conversion must not create or sync any CUDA context (GPTQModel#5105)."""
    from defuser.modeling import moe_experts_interface

    calls = []
    real_clear_memory = moe_experts_interface.clear_memory

    def recording_clear_memory(*args, **kwargs):
        calls.append(kwargs.get("device_list", args[1] if len(args) > 1 else None))
        return real_clear_memory(*args, **kwargs)

    monkeypatch.setattr(moe_experts_interface, "clear_memory", recording_clear_memory)
    monkeypatch.setattr(moe_experts_interface, "_unfuse_experts_weights_inplace", lambda *a, **k: True)
    monkeypatch.setattr(moe_experts_interface, "_model_experts_defuse_specs", lambda model: [])
    fake = fake_cuda(initialized=True)

    model = torch.nn.Module()
    model.experts = torch.nn.Linear(2, 2)
    unfused = moe_experts_interface.prepare_model_for_moe_quantization(model)

    assert unfused
    assert calls == [[]]
    assert fake.synced == [] and fake.emptied == 0
