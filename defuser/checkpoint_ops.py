import torch
from transformers.core_model_loading import Chunk, Concatenate, ConversionOps, MergeModulelist


class OwnedChunk(Chunk):
    """Split fused tensors into independent chunks so save/load keeps both weights."""

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        split = super().convert(input_dict, source_patterns, target_patterns, **kwargs)
        # `torch.chunk()` returns views into shared storage, which can make safetensors
        # drop one side of the split tensor during save. Clone each chunk to own storage.
        return {name: tensor.contiguous().clone() for name, tensor in split.items()}


class SplitFusedExpertGateUpProj(ConversionOps):
    """Split Mixtral-style fused expert weights into per-expert gate/up projections."""

    def __init__(self, expert_dim: int = 0, proj_dim: int = 0):
        self.expert_dim = expert_dim
        self.proj_dim = proj_dim

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        if len(target_patterns) != 2:
            raise ValueError("SplitFusedExpertGateUpProj expects exactly two target patterns.")

        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        num_experts = tensor.size(self.expert_dim)

        split_tensors: dict[str, torch.Tensor] = {}
        for expert_idx in range(num_experts):
            expert_tensor = tensor.select(self.expert_dim, expert_idx)
            gate_proj, up_proj = torch.chunk(expert_tensor, 2, dim=self.proj_dim)
            split_tensors[target_patterns[0].replace("*", str(expert_idx))] = gate_proj.contiguous().clone()
            split_tensors[target_patterns[1].replace("*", str(expert_idx))] = up_proj.contiguous().clone()

        return split_tensors

    @property
    def reverse_op(self) -> ConversionOps:
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
