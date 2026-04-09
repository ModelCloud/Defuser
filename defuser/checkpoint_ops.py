# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
import torch
from transformers.core_model_loading import Chunk, Concatenate, ConversionOps, MergeModulelist


def _owned_contiguous_clone(tensor: torch.Tensor) -> torch.Tensor:
    """Return a contiguous tensor with its own storage using a single clone."""
    return tensor.clone(memory_format=torch.contiguous_format)


class OwnedChunk(Chunk):
    """Split fused tensors into independent chunks so save/load keeps both weights."""

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        split = super().convert(input_dict, source_patterns, target_patterns, **kwargs)
        # `torch.chunk()` returns views into shared storage, which can make safetensors
        # drop one side of the split tensor during save. Clone each chunk to own storage.
        return {name: _owned_contiguous_clone(tensor) for name, tensor in split.items()}


class SplitFusedExpertGateUpProj(ConversionOps):
    """Split Mixtral-style fused expert weights into per-expert gate/up projections."""

    def __init__(self, expert_dim: int = 0, proj_dim: int = 0):
        self.expert_dim = expert_dim
        self.proj_dim = proj_dim

    @staticmethod
    def _expert_target(pattern: str, expert_idx: int) -> str:
        """Expand one target pattern into the per-expert key for ``expert_idx``."""
        if "*" in pattern:
            return pattern.replace("*", str(expert_idx))
        return pattern.replace(".0.", f".{expert_idx}.", 1)

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        """Split one fused gate/up tensor into cloned per-expert gate and up tensors."""
        if len(target_patterns) != 2:
            raise ValueError("SplitFusedExpertGateUpProj expects exactly two target patterns.")

        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        num_experts = tensor.size(self.expert_dim)

        split_tensors: dict[str, torch.Tensor] = {}
        for expert_idx in range(num_experts):
            expert_tensor = tensor.select(self.expert_dim, expert_idx)
            gate_proj, up_proj = torch.chunk(expert_tensor, 2, dim=self.proj_dim)
            split_tensors[self._expert_target(target_patterns[0], expert_idx)] = _owned_contiguous_clone(gate_proj)
            split_tensors[self._expert_target(target_patterns[1], expert_idx)] = _owned_contiguous_clone(up_proj)

        return split_tensors

    @property
    def reverse_op(self) -> ConversionOps:
        """Return the inverse merge op used when writing fused checkpoints."""
        return MergeSplitExpertGateUpProj()


class MergeSplitExpertGateUpProj(ConversionOps):
    """Merge per-expert gate/up projections back into a fused Mixtral tensor."""

    def __init__(self, expert_dim: int = 0, proj_dim: int = 0):
        self.expert_dim = expert_dim
        self.proj_dim = proj_dim
        self._concat = Concatenate(dim=self.proj_dim)
        self._stack = MergeModulelist(dim=self.expert_dim)

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, list[torch.Tensor]], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        """Merge per-expert gate/up tensors back into one fused expert tensor."""
        if len(source_patterns) != 2:
            raise ValueError("MergeSplitExpertGateUpProj expects exactly two source patterns.")
        if len(target_patterns) != 1:
            raise ValueError("MergeSplitExpertGateUpProj expects a single target pattern.")

        gate_weights = input_dict[source_patterns[0]]
        up_weights = input_dict[source_patterns[1]]
        if len(gate_weights) != len(up_weights):
            raise ValueError(
                "Mismatched per-expert gate/up weights while merging Mixtral gate_up_proj: "
                f"{len(gate_weights)} gate vs {len(up_weights)} up."
            )

        fused_per_expert = []
        for gate_proj, up_proj in zip(gate_weights, up_weights):
            fused = self._concat.convert(
                {
                    source_patterns[0]: gate_proj,
                    source_patterns[1]: up_proj,
                },
                source_patterns,
                target_patterns,
            )[target_patterns[0]]
            fused_per_expert.append(fused.contiguous())

        return self._stack.convert({target_patterns[0]: fused_per_expert}, [target_patterns[0]], target_patterns)


class SplitFusedExpertDownProj(ConversionOps):
    """Split Mixtral-style fused expert down projections into per-expert linears."""

    def __init__(self, expert_dim: int = 0):
        self.expert_dim = expert_dim

    @staticmethod
    def _expert_target(pattern: str, expert_idx: int) -> str:
        """Expand one target pattern into the per-expert key for ``expert_idx``."""
        if "*" in pattern:
            return pattern.replace("*", str(expert_idx))
        return pattern.replace(".0.", f".{expert_idx}.", 1)

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        """Split one fused expert down projection into cloned per-expert tensors."""
        if len(target_patterns) != 1:
            raise ValueError("SplitFusedExpertDownProj expects a single target pattern.")

        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        num_experts = tensor.size(self.expert_dim)

        split_tensors: dict[str, torch.Tensor] = {}
        for expert_idx in range(num_experts):
            expert_tensor = tensor.select(self.expert_dim, expert_idx)
            split_tensors[self._expert_target(target_patterns[0], expert_idx)] = _owned_contiguous_clone(expert_tensor)

        return split_tensors

    @property
    def reverse_op(self) -> ConversionOps:
        """Return the inverse merge op used when writing fused checkpoints."""
        return MergeModulelist(dim=self.expert_dim)
