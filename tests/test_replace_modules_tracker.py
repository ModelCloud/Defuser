# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import gc
import weakref

import pytest
import torch

from defuser.modeling.fused_moe.replace_modules import (
    ModuleReplacementTracker,
    ReplacementModuleBase,
    release_original_module_,
)


class DummyOriginalForTrackerTests(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)


class DummyReplacementForTrackerTests(ReplacementModuleBase):
    @classmethod
    def original_module_class(cls) -> str:
        return "DummyOriginalForTrackerTests"

    @classmethod
    def from_original(cls, original: torch.nn.Module, config) -> "DummyReplacementForTrackerTests":
        return cls(original)

    def _materialize_weights(self) -> None:
        pass


class DummyContainerForTrackerTests(torch.nn.Module):
    def __init__(self, replacement: ReplacementModuleBase):
        super().__init__()
        self.replacement = replacement


@pytest.fixture(autouse=True)
def _cleanup_tracker():
    tracker = ModuleReplacementTracker.get_instance()
    tracker.clear()
    yield
    tracker.clear()


def test_replacement_owns_original_reference_until_release():
    original = DummyOriginalForTrackerTests()
    replacement = DummyReplacementForTrackerTests(original)

    assert replacement._get_original_module() is original

    replacement.release_original_module()

    with pytest.raises(RuntimeError, match="already been released"):
        replacement._get_original_module()


def test_tracker_metadata_does_not_keep_original_alive_after_release():
    tracker = ModuleReplacementTracker.get_instance()
    original = DummyOriginalForTrackerTests()
    original_ref = weakref.ref(original)
    replacement = DummyReplacementForTrackerTests(original)
    tracker_name = str(id(replacement))

    info = tracker.get_info_by_name(tracker_name)
    assert info is not None
    assert info.original_module_class == DummyOriginalForTrackerTests.__name__
    assert info.replacement_module_class == DummyReplacementForTrackerTests.__name__
    assert info.replacement_module_ref() is replacement

    replacement.release_original_module()
    del original
    gc.collect()

    assert original_ref() is None
    assert tracker.get_info_by_name(tracker_name) is None


def test_release_all_originals_releases_replacement_owned_originals():
    tracker = ModuleReplacementTracker.get_instance()
    replacement = DummyReplacementForTrackerTests(DummyOriginalForTrackerTests())
    tracker_name = str(id(replacement))

    tracker.release_all_originals()

    with pytest.raises(RuntimeError, match="already been released"):
        replacement._get_original_module()
    assert tracker.get_info_by_name(tracker_name) is None


def test_release_original_module_helper_releases_all_replacements_in_model():
    tracker = ModuleReplacementTracker.get_instance()
    replacement = DummyReplacementForTrackerTests(DummyOriginalForTrackerTests())
    tracker_name = str(id(replacement))
    model = DummyContainerForTrackerTests(replacement)

    release_original_module_(model)

    with pytest.raises(RuntimeError, match="already been released"):
        replacement._get_original_module()
    assert tracker.get_info_by_name(tracker_name) is None
