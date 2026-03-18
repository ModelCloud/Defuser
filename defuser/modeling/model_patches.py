# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from collections.abc import Callable

from logbar import LogBar

from defuser import DEBUG_ON
import torch

logger = LogBar(__name__)


_MODEL_CLASS_PATCH_REGISTRY: dict[str, Callable] = {}
_MODEL_PATCH_REGISTRY: dict[str, Callable] = {}


def register_model_class_patch(model_type: str):
    """Register a one-time class patch that runs before model construction."""
    def decorator(func: Callable):
        _MODEL_CLASS_PATCH_REGISTRY[model_type] = func
        return func

    return decorator


def register_model_patch(model_type: str):
    """Register a runtime patch that runs on an instantiated model object."""
    def decorator(func: Callable):
        _MODEL_PATCH_REGISTRY[model_type] = func
        return func

    return decorator

@register_model_class_patch("qwen3_omni_moe")
def patch_qwen3_omni_text_class() -> list[str]:
    """Teach HF init code how to initialize unfused qwen3-omni thinker experts."""
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoePreTrainedModel
    from defuser.modeling.unfused_moe.qwen3_omni_moe import LinearQwen3OmniMoeThinkerTextSparseMoeBlock
    orig_init_weights = Qwen3OmniMoePreTrainedModel._init_weights

    def patched_init_weights(self, module):
        try:
            orig_init_weights(self, module)
        except AttributeError as e:
            # fallback for unfused experts
            if isinstance(module, LinearQwen3OmniMoeThinkerTextSparseMoeBlock):
                std = self.config.initializer_range
                experts = module.experts

                if hasattr(experts, "gate_proj"):
                    torch.nn.init.normal_(experts.gate_proj.weight, 0.0, std)
                if hasattr(experts, "up_proj"):
                    torch.nn.init.normal_(experts.up_proj.weight, 0.0, std)
                if hasattr(experts, "down_proj"):
                    torch.nn.init.normal_(experts.down_proj.weight, 0.0, std)

                if hasattr(module, "gate"):
                    torch.nn.init.normal_(module.gate.weight, 0.0, std)
            else:
                raise e

    Qwen3OmniMoePreTrainedModel._init_weights = patched_init_weights

    return []


@register_model_patch("qwen3_omni_moe")
def patch_qwen3_omni_text_runtime(model) -> list[str]:
    """Restore text-only ``forward`` and ``generate`` behavior after class swapping."""
    model_cls = type(model)
    if not getattr(model_cls, "__module__", "").startswith("transformers.models.qwen3_omni_moe."):
        return []

    applied = []
    original_generate = getattr(model_cls, "generate", None)
    if original_generate is not None and not getattr(original_generate, "_defuser_qwen3_omni_text_runtime", False):

        def generate(self, *args, return_audio=None, **kwargs):
            if return_audio is None:
                return_audio = False
            return original_generate(self, *args, return_audio=return_audio, **kwargs)

        generate._defuser_qwen3_omni_text_runtime = True
        model_cls.generate = generate
        applied.append("generate")

    if "forward" not in model_cls.__dict__:
        def forward(self, *args, **kwargs):
            return self.thinker(*args, **kwargs)

        forward._defuser_qwen3_omni_text_runtime = True
        model_cls.forward = forward
        applied.append("forward")

    return applied


def apply_model_class_patches(model_type) -> list[str]:
    """Run any registered pre-construction patch for ``model_type``."""
    patch_model_class = _MODEL_CLASS_PATCH_REGISTRY.get(model_type)
    if patch_model_class is None:
        return []

    applied = patch_model_class()
    if applied and DEBUG_ON:
        logger.debug(f"Applied model class patches for model_type={model_type}: {', '.join(applied)}")
    return applied


def apply_model_patches(model) -> list[str]:
    """Run any registered runtime patch for the instantiated ``model``."""
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    patch = _MODEL_PATCH_REGISTRY.get(model_type)
    if patch is None:
        return []

    applied = patch(model)
    if applied and DEBUG_ON:
        logger.debug(f"Applied model patches for model_type={model_type}: {', '.join(applied)}")
    return applied
