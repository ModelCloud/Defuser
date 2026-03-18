# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# Adapted from intel/auto-round
# at https://github.com/intel/auto-round/blob/main/auto_round/modeling/fused_moe/replace_modules.py

import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Type

import torch
from logbar import LogBar
from tqdm import tqdm

from defuser.utils.common import is_within_max_layers, is_transformers_version_greater_or_equal_5

from defuser import DEBUG_ON

logger = LogBar(__name__)


def materialize_model(model: torch.nn.Module) -> None:
    """Materialize any deferred replacement weights attached to ``model``."""

    def _materialize_module(module: torch.nn.Module) -> None:
        if isinstance(module, ReplacementModuleBase):
            module.materialize_weights()

    # materialize all .children() and self
    model.apply(_materialize_module)

    # check if any module on meta device remains
    found_meta = False
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            logger.warn(f"Parameter {name} is still on meta device after materialization.")
            found_meta = True

    for name, buffer in model.named_buffers():
        if buffer.device.type == "meta":
            logger.warn(f"Buffer {name} is still on meta device after materialization.")
            found_meta = True

    if not found_meta:
        if DEBUG_ON: logger.debug("All parameters and buffers have been materialized from meta device.")
    release_original_module_(model)


def release_original_module_(model: torch.nn.Module) -> None:
    """Drop references to original fused modules after replacement is complete."""

    def _clear_source_module(module: torch.nn.Module) -> None:
        if isinstance(module, ReplacementModuleBase):
            module.release_original_module()

    model.apply(_clear_source_module)


class ReplacementModuleBase(ABC, torch.nn.Module):
    """
    Abstract base class for module replacement during calibration phase.

    Replacement modules replace original modules to ensure all components
    receive data for proper quantization statistics.

    Subclasses must:
    1. Implement `original_module_class()` to return the target module class name
    2. Implement `__init__()` with signature:
       (self, original, config)
    """

    # Registry: module class name -> replacement module class
    _replacement_registry: Dict[str, Type["ReplacementModuleBase"]] = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses in the replacement registry."""
        super().__init_subclass__(**kwargs)

        # Only register if it's a concrete implementation (not ABC)
        if not getattr(cls, "__abstractmethods__", None):
            if cls.original_module_class() is None:
                raise TypeError(
                    f"{cls.__name__} must implement 'original_module_class()' class method "
                    "to return the name of the module class it replaces"
                )

            if cls.original_module_class() in cls._replacement_registry:
                existing = cls._replacement_registry[cls.original_module_class()]
                raise ValueError(
                    f"Module '{cls.original_module_class()}' already registered to "
                    f"{existing.__name__}. Cannot register {cls.__name__}."
                )

            cls._replacement_registry[cls.original_module_class()] = cls
            if DEBUG_ON: logger.debug(f"Registered {cls.__name__} for replacing {cls.original_module_class()}")

    def __init__(self, original: torch.nn.Module):
        super().__init__()
        self._original_module = original
        self._tracker_name = str(id(self))
        _global_tracker.register_replacement(
            name=self._tracker_name,
            original_class=original.__class__.__name__,
            replacement=self,
        )
        self._materialized = False

    @classmethod
    def get_replacement_class(cls, module_class_name: str) -> Type["ReplacementModuleBase"]:
        """Get replacement class for a given module class name."""
        return cls._replacement_registry.get(module_class_name)

    @classmethod
    def is_registered(cls, module_class_name: str) -> bool:
        """Check if a module class has a replacement implementation."""
        return module_class_name in cls._replacement_registry

    @classmethod
    def is_to_be_replaced(
            cls,
            original: torch.nn.Module,
    ) -> bool:
        """Determine if the given module should be replaced.

        Users can extend this method to add custom logic for replacement.
        """
        return cls.is_registered(original.__class__.__name__)

    @classmethod
    def get_registered_modules(cls) -> list:
        """Get list of all registered module class names."""
        return list(cls._replacement_registry.keys())

    @classmethod
    @abstractmethod
    def original_module_class(cls) -> str:
        """Return the class name of the module this replaces."""
        pass

    @classmethod
    @abstractmethod
    def from_original(
            cls,
            original: torch.nn.Module,
            config,
    ) -> "ReplacementModuleBase":
        """Create replacement module from original module."""
        pass

    def materialize_weights(self):
        """Materialize weights if needed."""
        if not self._materialized:
            self._materialize_weights()
            self.post_process_materialization()

    def _materialize_weights(self) -> None:
        """Materialize weights from the original module.

        Subclasses should override this method to implement
        weight materialization logic.
        """
        pass

    def release_original_module(self) -> None:
        """Release reference to the original module to free memory."""
        self._original_module = None
        _global_tracker.release_original(self)

    def _get_original_module(self) -> torch.nn.Module:
        """Get the original module associated with this replacement."""
        if self._original_module is None:
            raise RuntimeError("Original module has already been released.")
        return self._original_module

    def post_process_materialization(self) -> None:
        """Mark the replacement module as materialized."""
        self._materialized = True
        self.release_original_module()


def _log_first_moe_block(model: torch.nn.Module, label: str) -> None:
    """Log the first experts module found in the model for debugging."""
    for name, module in model.named_modules():
        if (name.endswith(".experts")
                and "_original_module" not in name): # FIXME `_original_module` will be moved to a different location in the future; a temporary check has been added.
            logger.info(f"Experts ({label}) [{name}] ({module.__class__.__name__}):\n{module}")
            return


def _handle_moe_modules(model: torch.nn.Module) -> list[str]:
    """Handle fused MOE modules using transformers' linear_loop backend.

    Args:
        model: The model to process

    Returns:
        List of module names that were processed
    """
    from defuser.modeling.moe_experts_interface import (
        is_linear_loop_available,
        prepare_model_for_moe_quantization,
    )

    if not is_linear_loop_available():
        logger.warn(
            "transformers' linear_loop experts interface not available (requires transformers 5.0+). "
            "MOE modules with @use_experts_implementation decorator will fall back to custom replacements "
            "if registered."
        )
        return []

    # Use transformers' experts interface
    unfused = prepare_model_for_moe_quantization(model)
    if unfused:
        logger.info(f"Prepared {len(unfused)} MOE modules for quantization")
    return unfused


def apply_replacements(
        model: torch.nn.Module,
        auto_detect_moe: bool = True,
        max_layers: int | None = None,
) -> torch.nn.Module:
    """
    Function to apply module replacements to a model.

    This scans all modules in the model and replaces any registered modules with their
    replacement equivalents. Non-permanent modules are tracked for later restoration.

    The model is modified in-place, so the same model object should be used.

    Args:
        model: The model to apply module replacement to (modified in-place).
        auto_detect_moe: If True, automatically detect and handle fused MOE modules
            (transformers 5.0+ pattern). Default is True.
        max_layers: If provided, only replace modules under ``layers.<idx>``
            where ``idx < max_layers``.

    Returns:
        The model with modules replaced.
    """
    _log_first_moe_block(model, "before replacement")

    # Run the generic MoE tensor defusion pass first so models with supported
    # fused experts can stay on their upstream module structure.
    if auto_detect_moe and is_transformers_version_greater_or_equal_5():
        _handle_moe_modules(model)

    # Fall back to replacement modules for any models that still need a custom
    # structural wrapper after the generic experts pass.
    _apply_custom_replacements(model, max_layers=max_layers)

    _log_first_moe_block(model, "after replacement")

    return model


def _apply_custom_replacements(
        model: torch.nn.Module,
        max_layers: int | None = None,
) -> list:
    """Scan model and replace registered modules with custom implementations.

    Args:
        model: The model to scan and apply replacements to (modified in-place).

    Returns:
        List of (name, replacement_class) tuples for replaced modules.
    """
    replaced = []

    # Step 1: Collect all modules that need replacement
    if DEBUG_ON: logger.debug("Scanning for modules to replace")
    modules_to_replace = []
    for name, module in model.named_modules():
        # skip replaced modules
        if isinstance(module, ReplacementModuleBase):
            continue
        if not is_within_max_layers(name, max_layers):
            continue
        class_name = module.__class__.__name__
        if ReplacementModuleBase.is_registered(class_name) and ReplacementModuleBase.get_replacement_class(
                class_name
        ).is_to_be_replaced(module):
            modules_to_replace.append((name, module, class_name))

    # Step 2: Replace modules
    if modules_to_replace:
        logger.info(f"Found {len(modules_to_replace)} modules to replace")
        for name, module, class_name in tqdm(modules_to_replace, desc="Replacing modules"):
            module = model.get_submodule(name)
            # The module might have been replaced earlier in the loop (parent-first replacement).
            # Skip if the class has changed or it no longer matches replacement criteria.
            if module.__class__.__name__ != class_name:
                if DEBUG_ON:
                    logger.debug(
                        f"Skipping replacement for {name}: class changed from {class_name} to {module.__class__.__name__}"
                    )
                continue
            replacement_cls = ReplacementModuleBase.get_replacement_class(class_name)
            if not replacement_cls.is_to_be_replaced(module):
                if DEBUG_ON: logger.debug(f"Skipping replacement for {name}: no longer matches replacement criteria")
                continue
            orig_dtype = next(module.parameters()).dtype
            replacement = replacement_cls.from_original(
                module,
                model.config,
            ).to(orig_dtype)
            print("replacement", replacement)
            model.set_submodule(name, replacement)
            replaced.append((name, replacement_cls))
    else:
        if DEBUG_ON: logger.debug("No modules found for replacement")

    # Log what was replaced
    if replaced:
        logger.info(f"Replaced {len(replaced)} modules")

    return replaced


@dataclass
class ReplacedModuleInfo:
    original_module_class: str
    replacement_module_class: str
    replacement_module_ref: "weakref.ReferenceType[ReplacementModuleBase]"


class ModuleReplacementTracker:
    """Tracker to maintain metadata for replacement modules.

    This is a singleton class - only one instance can exist.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModuleReplacementTracker, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if ModuleReplacementTracker._initialized:
            return

        # Map from replacement module id to tracker name
        self._replacement_to_name: Dict[int, str] = {}
        # Map from module name to ReplacedModuleInfo
        self._name_to_info: Dict[str, ReplacedModuleInfo] = {}

        ModuleReplacementTracker._initialized = True

    @classmethod
    def get_instance(cls) -> "ModuleReplacementTracker":
        """Get the singleton instance of the tracker."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_replacement(
            self,
            name: str,
            original_class: str,
            replacement: ReplacementModuleBase,
    ) -> None:
        """Register a module replacement."""
        replacement_id = id(replacement)
        self._replacement_to_name[replacement_id] = name
        self._name_to_info[name] = ReplacedModuleInfo(
            original_module_class=original_class,
            replacement_module_class=replacement.__class__.__name__,
            replacement_module_ref=weakref.ref(replacement),
        )
        if DEBUG_ON: logger.debug(f"Registered replacement for module: {name}")

    def get_original(self, replacement: ReplacementModuleBase) -> torch.nn.Module | None:
        """Get the original module for a given replacement module."""
        return replacement._original_module

    def get_info_by_name(self, name: str) -> ReplacedModuleInfo | None:
        """Get replacement info by module name."""
        info = self._name_to_info.get(name)
        if info is None:
            return None
        if info.replacement_module_ref() is None:
            del self._name_to_info[name]
            return None
        return info

    def release_original(self, replacement: ReplacementModuleBase) -> None:
        """Release the original module associated with a replacement module."""
        replacement_id = id(replacement)
        name = self._replacement_to_name.pop(replacement_id, None)
        if name is not None:
            info = self._name_to_info.get(name)
            if info is not None:
                replacement_ref = info.replacement_module_ref()
                if replacement_ref is None or replacement_ref is replacement:
                    del self._name_to_info[name]
            if DEBUG_ON: logger.debug(f"Released original module for replacement {replacement_id}")

    def release_all_originals(self) -> None:
        """Release all tracked original modules."""
        count = 0
        for info in self._name_to_info.values():
            replacement = info.replacement_module_ref()
            if replacement is not None and replacement._original_module is not None:
                replacement._original_module = None
                count += 1

        self._replacement_to_name.clear()
        self._name_to_info.clear()
        if count > 0:
            if DEBUG_ON: logger.debug(f"Released {count} original modules from tracker")

    def clear(self) -> None:
        """Clear all tracked information."""
        self._replacement_to_name.clear()
        self._name_to_info.clear()
        if DEBUG_ON: logger.debug("Cleared module replacement tracker")


_global_tracker = ModuleReplacementTracker()
