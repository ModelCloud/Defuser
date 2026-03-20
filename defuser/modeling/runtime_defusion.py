# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from types import MethodType

import torch
from torch import nn

from defuser.modeling.moe_experts_interface import _ExpertContainer, _install_compact_expert_repr


def _activation_fn(module: nn.Module):
    act_fn = getattr(module, "activation_fn", None)
    if act_fn is None:
        act_fn = getattr(module, "act_fn", None)
    if act_fn is None:
        raise AttributeError(f"{module.__class__.__name__} is missing `activation_fn`/`act_fn`.")
    return act_fn


def _make_linear(
    *,
    in_features: int,
    out_features: int,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> nn.Linear:
    linear = nn.Linear(
        in_features,
        out_features,
        bias=bias is not None,
        device=weight.device,
        dtype=weight.dtype,
    )
    with torch.no_grad():
        linear.weight.copy_(weight)
        if bias is not None:
            linear.bias.copy_(bias)
    return linear


def _split_gate_up_linear(gate_up_proj: nn.Linear) -> tuple[nn.Linear, nn.Linear]:
    split_size = gate_up_proj.out_features // 2
    bias = gate_up_proj.bias
    gate_bias = bias[:split_size].contiguous() if bias is not None else None
    up_bias = bias[split_size:].contiguous() if bias is not None else None
    gate_proj = _make_linear(
        in_features=gate_up_proj.in_features,
        out_features=split_size,
        weight=gate_up_proj.weight[:split_size].contiguous(),
        bias=gate_bias,
    )
    up_proj = _make_linear(
        in_features=gate_up_proj.in_features,
        out_features=split_size,
        weight=gate_up_proj.weight[split_size:].contiguous(),
        bias=up_bias,
    )
    return gate_proj, up_proj


def _standard_split_gate_up_forward(self: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    gate = self.gate_proj(hidden_states)
    up = self.up_proj(hidden_states)
    return self.down_proj(up * _activation_fn(self)(gate))


def _phi4_audio_split_gate_up_forward(self: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = self.layer_norm(hidden_states)
    gate = self.gate_proj(hidden_states)
    up = self.up_proj(hidden_states)
    up = up * self.act_fn(gate)
    up = self.dropout(up)
    hidden_states = self.down_proj(up)
    return self.dropout(hidden_states)


def _zamba2_split_gate_up_forward(
    self: nn.Module,
    hidden_state: torch.Tensor,
    layer_idx: int | None = None,
) -> torch.Tensor:
    layer_idx = self.layer_dic[layer_idx]
    adapter_out = self.gate_up_proj_adapter_list[layer_idx](hidden_state)
    gate_adapter, up_adapter = torch.chunk(adapter_out, 2, dim=-1)
    gate_state = self.gate_proj(hidden_state) + gate_adapter
    up_state = self.up_proj(hidden_state) + up_adapter
    return self.down_proj(self.act_fn(gate_state) * up_state)


def patch_split_gate_up_mlp(module: nn.Module, variant: str = "standard") -> bool:
    if getattr(module, "_defuser_split_gate_up_runtime", False):
        return False

    gate_up_proj = getattr(module, "gate_up_proj", None)
    if not isinstance(gate_up_proj, nn.Linear):
        return False

    gate_proj, up_proj = _split_gate_up_linear(gate_up_proj)
    if variant == "phi4_audio":
        gate_proj, up_proj = up_proj, gate_proj
    module.add_module("gate_proj", gate_proj)
    module.add_module("up_proj", up_proj)
    delattr(module, "gate_up_proj")

    if variant == "standard":
        module.forward = MethodType(_standard_split_gate_up_forward, module)
    elif variant == "phi4_audio":
        module.forward = MethodType(_phi4_audio_split_gate_up_forward, module)
    elif variant == "zamba2":
        module.forward = MethodType(_zamba2_split_gate_up_forward, module)
    else:
        raise ValueError(f"Unsupported split gate_up MLP variant: {variant}")

    module._defuser_split_gate_up_runtime = True
    return True


def _parallel_experts_forward(self: nn.Module, inputs: torch.Tensor, expert_size) -> torch.Tensor:
    input_list = inputs.split(expert_size, dim=0)
    output_list = []
    for expert_idx in range(self.num_experts):
        output_list.append(getattr(self, str(expert_idx)).linear(input_list[expert_idx]))
    return torch.cat(output_list, dim=0)


def patch_parallel_experts(module: nn.Module) -> bool:
    if getattr(module, "_defuser_parallel_experts_runtime", False):
        return False

    weight = getattr(module, "weight", None)
    if not isinstance(weight, nn.Parameter) or weight.dim() != 3:
        return False

    for expert_idx in range(module.num_experts):
        container = _ExpertContainer()
        linear = _make_linear(
            in_features=module.input_size,
            out_features=module.output_size,
            weight=weight[expert_idx].contiguous(),
        )
        container.add_module("linear", linear)
        module.add_module(str(expert_idx), container)

    delattr(module, "weight")
    module.forward = MethodType(_parallel_experts_forward, module)
    module._unfused_experts = True
    module._defuser_parallel_experts_runtime = True
    _install_compact_expert_repr(module)
    return True


def _longcat_flash_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    final_hidden_states = torch.zeros_like(hidden_states)
    if top_k_index.numel() == 0:
        return final_hidden_states

    expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.total_experts).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False)
    for expert_idx_tensor in expert_hit:
        expert_idx = int(expert_idx_tensor.item())
        selection_idx, token_idx = torch.where(expert_mask[expert_idx].squeeze(0))
        if token_idx.numel() == 0:
            continue

        expert = getattr(self, str(expert_idx))
        current_state = hidden_states[token_idx]
        if hasattr(expert, "identity"):
            current_hidden_states = expert.identity(current_state)
        else:
            gate = expert.gate_proj(current_state)
            up = expert.up_proj(current_state)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = expert.down_proj(current_hidden_states)

        current_hidden_states = current_hidden_states * top_k_weights[token_idx, selection_idx, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(hidden_states.dtype))

    return final_hidden_states


def patch_longcat_flash_experts(module: nn.Module) -> bool:
    if getattr(module, "_defuser_longcat_runtime", False):
        return False

    gate_up_proj = getattr(module, "gate_up_proj", None)
    down_proj = getattr(module, "down_proj", None)
    if gate_up_proj is None and down_proj is None and hasattr(module, "0"):
        return False

    for expert_idx in range(module.total_experts):
        container = _ExpertContainer()
        if expert_idx < module.num_routed_experts and gate_up_proj is not None and down_proj is not None:
            fused_gate_up = gate_up_proj[expert_idx]
            split_size = fused_gate_up.shape[0] // 2
            gate_proj = _make_linear(
                in_features=module.hidden_size,
                out_features=split_size,
                weight=fused_gate_up[:split_size].contiguous(),
            )
            up_proj = _make_linear(
                in_features=module.hidden_size,
                out_features=split_size,
                weight=fused_gate_up[split_size:].contiguous(),
            )
            down_linear = _make_linear(
                in_features=module.intermediate_size,
                out_features=module.hidden_size,
                weight=down_proj[expert_idx].contiguous(),
            )
            container.add_module("gate_proj", gate_proj)
            container.add_module("up_proj", up_proj)
            container.add_module("down_proj", down_linear)
        else:
            container.add_module("identity", nn.Identity())
        module.add_module(str(expert_idx), container)

    if hasattr(module, "gate_up_proj"):
        delattr(module, "gate_up_proj")
    if hasattr(module, "down_proj"):
        delattr(module, "down_proj")

    module.num_experts = module.total_experts
    module.forward = MethodType(_longcat_flash_forward, module)
    module._unfused_experts = True
    module._defuser_longcat_runtime = True
    _install_compact_expert_repr(module)
    return True


def _dbrx_experts_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    batch_size = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(-1, self.ffn_hidden_size)
    next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)

    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().flatten()

    for expert_idx in expert_hit.tolist():
        idx, token_idx = torch.where(expert_mask[expert_idx])
        expert = getattr(self, str(expert_idx))
        current_state = hidden_states[token_idx]
        current_hidden_states = self.act_fn(expert.gate_proj(current_state)) * expert.up_proj(current_state)
        current_hidden_states = expert.down_proj(current_hidden_states)
        current_hidden_states = current_hidden_states.view(-1, self.ffn_hidden_size) * top_k_weights[token_idx, idx, None]
        next_states.index_add_(0, token_idx, current_hidden_states)

    return next_states.view(batch_size, -1, self.ffn_hidden_size)


def patch_dbrx_experts(module: nn.Module) -> bool:
    if getattr(module, "_defuser_dbrx_runtime", False):
        return False

    mlp = getattr(module, "mlp", None)
    if mlp is None or not hasattr(mlp, "w1") or not hasattr(mlp, "v1") or not hasattr(mlp, "w2"):
        return False

    split_shape = (module.num_experts, module.ffn_hidden_size, module.hidden_size)
    w1 = mlp.w1.view(split_shape)
    v1 = mlp.v1.view(split_shape)
    w2 = mlp.w2.view(split_shape)

    for expert_idx in range(module.num_experts):
        container = _ExpertContainer()
        container.add_module(
            "gate_proj",
            _make_linear(
                in_features=module.ffn_hidden_size,
                out_features=module.hidden_size,
                weight=w1[expert_idx].t().contiguous(),
            ),
        )
        container.add_module(
            "up_proj",
            _make_linear(
                in_features=module.ffn_hidden_size,
                out_features=module.hidden_size,
                weight=v1[expert_idx].t().contiguous(),
            ),
        )
        container.add_module(
            "down_proj",
            _make_linear(
                in_features=module.hidden_size,
                out_features=module.ffn_hidden_size,
                weight=w2[expert_idx].contiguous(),
            ),
        )
        module.add_module(str(expert_idx), container)

    module.act_fn = mlp.activation_fn
    delattr(module, "mlp")
    module.forward = MethodType(_dbrx_experts_forward, module)
    module._unfused_experts = True
    module._defuser_dbrx_runtime = True
    _install_compact_expert_repr(module)
    return True
