# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import torch
import torch.nn as nn

from defuser.modeling.moe_experts_interface import _detect_expert_projections


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
    module = _UnknownExpertsWithLossProperty()

    detected = _detect_expert_projections(module)

    assert detected == {
        "expert_weight": {"is_input_proj": True, "output_multiplier": 1},
    }
    assert module.loss_property_accesses == 0
