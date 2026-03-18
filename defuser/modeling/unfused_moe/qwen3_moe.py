# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# Adapted from intel/auto-round
# at https://github.com/intel/auto-round/blob/main/auto_round/modeling/unfused_moe/qwen3_moe.py

import torch
import torch.nn as nn

from defuser.modeling.unfused_moe.common import run_routed_experts


class LinearQwen3MoeSparseMoeBlock(nn.Module):
    """Qwen3 MoE block rewritten to expose one ``nn.Module`` per expert."""

    def __init__(self, config):
        super().__init__()
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP, Qwen3MoeTopKRouter

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # Reuse the upstream router module so `output_router_logits` hooks still attach.
        self.gate = Qwen3MoeTopKRouter(config)
        self.experts = nn.ModuleList(
            [Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Route tokens exactly like HF Qwen3 MoE, then run explicit expert modules."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(hidden_states)
        routing_weights = routing_weights.to(hidden_states.dtype)
        final_hidden_states = run_routed_experts(
            self.experts,
            hidden_states,
            routing_weights,
            selected_experts,
            self.num_experts,
        )
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states
