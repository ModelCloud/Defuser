import torch
from transformers.core_model_loading import Chunk


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
