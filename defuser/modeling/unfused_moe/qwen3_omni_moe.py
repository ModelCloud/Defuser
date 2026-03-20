# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch
import torch.nn as nn

from defuser.modeling.unfused_moe.common import run_routed_experts

class LinearQwen3OmniMoeThinkerTextSparseMoeBlock(nn.Module):
    """Text thinker MoE block for qwen3-omni with explicit per-expert modules."""

    def __init__(self, config):
        super().__init__()
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeThinkerTextMLP,
            Qwen3OmniMoeThinkerTextTopKRouter,
        )

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # Reuse the upstream router module so `output_router_logits` hooks still attach.
        self.gate = Qwen3OmniMoeThinkerTextTopKRouter(config)
        self.experts = nn.ModuleList(
            [
                Qwen3OmniMoeThinkerTextMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Route tokens exactly like HF qwen3-omni text MoE, then run explicit experts."""
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


class LinearQwen3OmniMoeTalkerTextSparseMoeBlock(nn.Module):
    """Text talker MoE block for qwen3-omni with explicit per-expert modules."""

    def __init__(self, config):
        super().__init__()
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeTalkerTextMLP,
            Qwen3OmniMoeTalkerTextTopKRouter,
        )

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = Qwen3OmniMoeTalkerTextTopKRouter(config)
        self.experts = nn.ModuleList(
            [
                Qwen3OmniMoeTalkerTextMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(self.num_experts)
            ]
        )
        self.shared_expert = Qwen3OmniMoeTalkerTextMLP(
            config,
            intermediate_size=config.shared_expert_intermediate_size,
        )
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states)
        _, routing_weights, selected_experts = self.gate(hidden_states)
        final_hidden_states = run_routed_experts(
            self.experts,
            hidden_states,
            routing_weights.to(hidden_states.dtype),
            selected_experts,
            self.num_experts,
        )
        shared_expert_output = torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        final_hidden_states = final_hidden_states + shared_expert_output
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
