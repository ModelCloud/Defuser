# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch
from torch import nn


def router_logits_from_output(router_output):
    """Return raw router logits from a router tensor or the Transformers 5.22 tuple.

    Transformers 5.22 routers return ``(router_logits, topk_weights,
    topk_indices)``. Defuser keeps its own grouped top-k selection, so it only
    needs the raw logits; the router's precomputed top-k is intentionally
    discarded so the routing math stays byte-identical across Transformers
    versions.
    """
    if isinstance(router_output, tuple):
        return router_output[0]
    return router_output


def run_routed_experts(
    experts: nn.ModuleList,
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Run a standard top-k routed MoE expert loop over explicit expert modules."""
    hidden_dim = hidden_states.shape[-1]
    final_hidden_states = torch.zeros(
        (hidden_states.shape[0], hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    expert_hit = torch.nonzero(expert_mask.sum(dim=(-1, -2)), as_tuple=False).flatten()

    for expert_idx in expert_hit.tolist():
        expert_layer = experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    return final_hidden_states
