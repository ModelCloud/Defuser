# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from collections.abc import Callable

from logbar import LogBar

from defuser import DEBUG_ON

logger = LogBar(__name__)


_MODEL_PATCH_REGISTRY: dict[str, Callable] = {}


def register_model_patch(model_type: str):
    def decorator(func: Callable):
        _MODEL_PATCH_REGISTRY[model_type] = func
        return func

    return decorator


@register_model_patch("qwen3_omni_moe")
def patch_qwen3_omni_text_runtime(model) -> list[str]:
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


def apply_model_patches(model) -> list[str]:
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    patch = _MODEL_PATCH_REGISTRY.get(model_type)
    if patch is None:
        return []

    applied = patch(model)
    if applied and DEBUG_ON:
        logger.debug(f"Applied model patches for model_type={model_type}: {', '.join(applied)}")
    return applied

