# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from collections.abc import Callable

from logbar import LogBar

from defuser import DEBUG_ON
from defuser.modeling.runtime_defusion import (
    patch_dbrx_experts,
    patch_longcat_flash_experts,
    patch_parallel_experts,
    patch_split_gate_up_mlp,
)
from defuser.utils.common import compile_module_name_filter, is_within_max_layers, matches_module_name_filter
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
    from defuser.modeling.unfused_moe.qwen3_omni_moe import (
        LinearQwen3OmniMoeTalkerTextSparseMoeBlock,
        LinearQwen3OmniMoeThinkerTextSparseMoeBlock,
    )
    orig_init_weights = Qwen3OmniMoePreTrainedModel._init_weights

    def patched_init_weights(self, module):
        try:
            orig_init_weights(self, module)
        except AttributeError as e:
            # fallback for unfused experts
            if isinstance(module, (LinearQwen3OmniMoeThinkerTextSparseMoeBlock, LinearQwen3OmniMoeTalkerTextSparseMoeBlock)):
                std = self.config.initializer_range
                experts = module.experts

                if hasattr(experts, "gate_proj"):
                    torch.nn.init.normal_(experts.gate_proj.weight, 0.0, std)
                if hasattr(experts, "up_proj"):
                    torch.nn.init.normal_(experts.up_proj.weight, 0.0, std)
                if hasattr(experts, "down_proj"):
                    torch.nn.init.normal_(experts.down_proj.weight, 0.0, std)
                if isinstance(experts, torch.nn.ModuleList):
                    for expert in experts:
                        torch.nn.init.normal_(expert.gate_proj.weight, 0.0, std)
                        torch.nn.init.normal_(expert.up_proj.weight, 0.0, std)
                        torch.nn.init.normal_(expert.down_proj.weight, 0.0, std)

                if hasattr(module, "gate"):
                    torch.nn.init.normal_(module.gate.weight, 0.0, std)
                if hasattr(module, "shared_expert"):
                    module.shared_expert._is_hf_initialized = True
                if hasattr(module, "shared_expert_gate"):
                    torch.nn.init.normal_(module.shared_expert_gate.weight, 0.0, std)
            else:
                raise e

    Qwen3OmniMoePreTrainedModel._init_weights = patched_init_weights

    return []


@register_model_patch("qwen3_omni_moe")
def patch_qwen3_omni_text_runtime(model, max_layers: int | None = None, filter_rules=None) -> list[str]:
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


def _patch_modules_by_class(
    model,
    patchers: dict[str, Callable],
    *,
    max_layers: int | None = None,
    filter_rules=None,
) -> list[str]:
    module_name_filter = compile_module_name_filter(filter_rules)
    applied = []
    for name, module in list(model.named_modules()):
        if not is_within_max_layers(name, max_layers):
            continue
        if not matches_module_name_filter(name, module_name_filter):
            continue
        class_path = f"{module.__class__.__module__}.{module.__class__.__name__}"
        patcher = patchers.get(class_path)
        if patcher is None:
            continue
        if patcher(module):
            applied.append(name)
    return applied


def _patch_split_gate_up_mlps(
    model,
    patchers: dict[str, str],
    *,
    max_layers: int | None = None,
    filter_rules=None,
) -> list[str]:
    return _patch_modules_by_class(
        model,
        {
            class_path: (lambda module, variant=variant: patch_split_gate_up_mlp(module, variant=variant))
            for class_path, variant in patchers.items()
        },
        max_layers=max_layers,
        filter_rules=filter_rules,
    )


_STANDARD_SPLIT_GATE_UP_CLASSES = {
    "transformers.models.dia.modeling_dia.DiaMLP": "standard",
    "transformers.models.glm.modeling_glm.GlmMLP": "standard",
    "transformers.models.glm4.modeling_glm4.Glm4MLP": "standard",
    "transformers.models.glm_image.modeling_glm_image.GlmImageTextMLP": "standard",
    "transformers.models.glm_ocr.modeling_glm_ocr.GlmOcrTextMLP": "standard",
    "transformers.models.phi3.modeling_phi3.Phi3MLP": "standard",
    "transformers.models.phi4_multimodal.modeling_phi4_multimodal.Phi4MultimodalMLP": "standard",
    "transformers.models.phi4_multimodal.modeling_phi4_multimodal.Phi4MultimodalAudioMLP": "phi4_audio",
    "transformers.models.zamba2.modeling_zamba2.Zamba2MLP": "zamba2",
}


@register_model_patch("dia")
def patch_dia_runtime(model, max_layers: int | None = None, filter_rules=None) -> list[str]:
    return _patch_split_gate_up_mlps(
        model,
        {"transformers.models.dia.modeling_dia.DiaMLP": _STANDARD_SPLIT_GATE_UP_CLASSES["transformers.models.dia.modeling_dia.DiaMLP"]},
        max_layers=max_layers,
        filter_rules=filter_rules,
    )


@register_model_patch("glm")
def patch_glm_runtime(model, max_layers: int | None = None, filter_rules=None) -> list[str]:
    return _patch_split_gate_up_mlps(
        model,
        {"transformers.models.glm.modeling_glm.GlmMLP": _STANDARD_SPLIT_GATE_UP_CLASSES["transformers.models.glm.modeling_glm.GlmMLP"]},
        max_layers=max_layers,
        filter_rules=filter_rules,
    )


@register_model_patch("glm4")
def patch_glm4_runtime(model, max_layers: int | None = None, filter_rules=None) -> list[str]:
    return _patch_split_gate_up_mlps(
        model,
        {"transformers.models.glm4.modeling_glm4.Glm4MLP": _STANDARD_SPLIT_GATE_UP_CLASSES["transformers.models.glm4.modeling_glm4.Glm4MLP"]},
        max_layers=max_layers,
        filter_rules=filter_rules,
    )


@register_model_patch("glm_image")
def patch_glm_image_runtime(model, max_layers: int | None = None, filter_rules=None) -> list[str]:
    return _patch_split_gate_up_mlps(
        model,
        {"transformers.models.glm_image.modeling_glm_image.GlmImageTextMLP": _STANDARD_SPLIT_GATE_UP_CLASSES["transformers.models.glm_image.modeling_glm_image.GlmImageTextMLP"]},
        max_layers=max_layers,
        filter_rules=filter_rules,
    )


@register_model_patch("glm_ocr")
def patch_glm_ocr_runtime(model, max_layers: int | None = None, filter_rules=None) -> list[str]:
    return _patch_split_gate_up_mlps(
        model,
        {"transformers.models.glm_ocr.modeling_glm_ocr.GlmOcrTextMLP": _STANDARD_SPLIT_GATE_UP_CLASSES["transformers.models.glm_ocr.modeling_glm_ocr.GlmOcrTextMLP"]},
        max_layers=max_layers,
        filter_rules=filter_rules,
    )


@register_model_patch("phi3")
def patch_phi3_runtime(model, max_layers: int | None = None, filter_rules=None) -> list[str]:
    return _patch_split_gate_up_mlps(
        model,
        {"transformers.models.phi3.modeling_phi3.Phi3MLP": _STANDARD_SPLIT_GATE_UP_CLASSES["transformers.models.phi3.modeling_phi3.Phi3MLP"]},
        max_layers=max_layers,
        filter_rules=filter_rules,
    )


@register_model_patch("phi4_multimodal")
def patch_phi4_multimodal_runtime(model, max_layers: int | None = None, filter_rules=None) -> list[str]:
    return _patch_split_gate_up_mlps(
        model,
        {
            "transformers.models.phi4_multimodal.modeling_phi4_multimodal.Phi4MultimodalMLP":
                _STANDARD_SPLIT_GATE_UP_CLASSES[
                    "transformers.models.phi4_multimodal.modeling_phi4_multimodal.Phi4MultimodalMLP"
                ],
            "transformers.models.phi4_multimodal.modeling_phi4_multimodal.Phi4MultimodalAudioMLP":
                _STANDARD_SPLIT_GATE_UP_CLASSES[
                    "transformers.models.phi4_multimodal.modeling_phi4_multimodal.Phi4MultimodalAudioMLP"
                ],
        },
        max_layers=max_layers,
        filter_rules=filter_rules,
    )


@register_model_patch("zamba2")
def patch_zamba2_runtime(model, max_layers: int | None = None, filter_rules=None) -> list[str]:
    return _patch_split_gate_up_mlps(
        model,
        {"transformers.models.zamba2.modeling_zamba2.Zamba2MLP": _STANDARD_SPLIT_GATE_UP_CLASSES["transformers.models.zamba2.modeling_zamba2.Zamba2MLP"]},
        max_layers=max_layers,
        filter_rules=filter_rules,
    )


@register_model_patch("dbrx")
def patch_dbrx_runtime(model, max_layers: int | None = None, filter_rules=None) -> list[str]:
    return _patch_modules_by_class(
        model,
        {"transformers.models.dbrx.modeling_dbrx.DbrxExperts": patch_dbrx_experts},
        max_layers=max_layers,
        filter_rules=filter_rules,
    )


def _patch_parallel_runtime(model, class_path: str, *, max_layers: int | None = None, filter_rules=None) -> list[str]:
    return _patch_modules_by_class(
        model,
        {class_path: patch_parallel_experts},
        max_layers=max_layers,
        filter_rules=filter_rules,
    )


@register_model_patch("granitemoe")
def patch_granitemoe_runtime(model, max_layers: int | None = None, filter_rules=None) -> list[str]:
    return _patch_parallel_runtime(
        model,
        "transformers.models.granitemoe.modeling_granitemoe.GraniteMoeParallelExperts",
        max_layers=max_layers,
        filter_rules=filter_rules,
    )


@register_model_patch("granitemoehybrid")
def patch_granitemoehybrid_runtime(model, max_layers: int | None = None, filter_rules=None) -> list[str]:
    return _patch_parallel_runtime(
        model,
        "transformers.models.granitemoehybrid.modeling_granitemoehybrid.GraniteMoeHybridParallelExperts",
        max_layers=max_layers,
        filter_rules=filter_rules,
    )


@register_model_patch("granitemoeshared")
def patch_granitemoeshared_runtime(model, max_layers: int | None = None, filter_rules=None) -> list[str]:
    return _patch_parallel_runtime(
        model,
        "transformers.models.granitemoeshared.modeling_granitemoeshared.GraniteMoeSharedParallelExperts",
        max_layers=max_layers,
        filter_rules=filter_rules,
    )


@register_model_patch("jetmoe")
def patch_jetmoe_runtime(model, max_layers: int | None = None, filter_rules=None) -> list[str]:
    return _patch_parallel_runtime(
        model,
        "transformers.models.jetmoe.modeling_jetmoe.JetMoeParallelExperts",
        max_layers=max_layers,
        filter_rules=filter_rules,
    )


@register_model_patch("longcat_flash")
def patch_longcat_flash_runtime(model, max_layers: int | None = None, filter_rules=None) -> list[str]:
    return _patch_modules_by_class(
        model,
        {"transformers.models.longcat_flash.modeling_longcat_flash.LongcatFlashExperts": patch_longcat_flash_experts},
        max_layers=max_layers,
        filter_rules=filter_rules,
    )


def apply_model_class_patches(model_type) -> list[str]:
    """Run any registered pre-construction patch for ``model_type``."""
    patch_model_class = _MODEL_CLASS_PATCH_REGISTRY.get(model_type)
    if patch_model_class is None:
        return []

    applied = patch_model_class()
    if applied and DEBUG_ON:
        logger.debug(f"Applied model class patches for model_type={model_type}: {', '.join(applied)}")
    return applied


def apply_model_patches(model, max_layers: int | None = None, filter_rules=None) -> list[str]:
    """Run any registered runtime patch for the instantiated ``model``."""
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    patch = _MODEL_PATCH_REGISTRY.get(model_type)
    if patch is None:
        return []

    applied = patch(model, max_layers=max_layers, filter_rules=filter_rules)
    if applied and DEBUG_ON:
        logger.debug(f"Applied model patches for model_type={model_type}: {', '.join(applied)}")
    return applied
