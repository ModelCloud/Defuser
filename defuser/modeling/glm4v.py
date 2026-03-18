from torch import nn
from transformers.activations import ACT2FN


class LinearGlm4vTextMLP(nn.Module):
    """GLM4V text MLP with the fused gate/up projection split into two linears."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        """Reproduce the original fused GLM4V text MLP using split linear layers."""
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        # Match the original fused `gate_up_proj.chunk(2, dim=-1)` activation path.
        return self.down_proj(up * self.activation_fn(gate))
