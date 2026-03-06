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
