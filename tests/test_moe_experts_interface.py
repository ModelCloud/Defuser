# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import torch
import torch.nn as nn

from defuser.modeling.moe_experts_interface import _detect_expert_projections, _unfuse_experts_weights_inplace


class _UnknownExpertsWithLossProperty(nn.Module):
    """Regression fixture for expert detection on modules with unrelated properties."""

    def __init__(self) -> None:
        super().__init__()
        self.expert_weight = nn.Parameter(torch.randn(4, 8, 16))
        self.loss_property_accesses = 0

    @property
    def loss_function(self):
        self.loss_property_accesses += 1
        raise AssertionError("expert detection should not touch unrelated properties")


def test_detect_expert_projections_ignores_unrelated_properties() -> None:
    """Projection detection should ignore unrelated properties on expert fixtures."""
    module = _UnknownExpertsWithLossProperty()

    detected = _detect_expert_projections(module)

    assert detected == {
        "expert_weight": {"is_input_proj": True, "output_multiplier": 1},
    }
    assert module.loss_property_accesses == 0


class _BufferBackedExperts(nn.Module):
    """Exercise the buffer-backed fused expert path end to end."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("gate_up_proj", torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4))
        self.register_buffer("down_proj", torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3))


def test_unfuse_experts_supports_registered_buffers() -> None:
    """Buffer-backed fused experts should unfuse into per-expert Linear layers."""
    module = _BufferBackedExperts()
    expected_gate_proj = module.gate_up_proj[0, :3].clone()
    expected_up_proj = module.gate_up_proj[0, 3:].clone()
    expected_down_proj = module.down_proj[0].clone()

    changed = _unfuse_experts_weights_inplace(module, check_decorator=False)

    assert changed is True
    expert0 = getattr(module, "0")
    torch.testing.assert_close(expert0.gate_proj.weight, expected_gate_proj)
    torch.testing.assert_close(expert0.up_proj.weight, expected_up_proj)
    torch.testing.assert_close(expert0.down_proj.weight, expected_down_proj)
