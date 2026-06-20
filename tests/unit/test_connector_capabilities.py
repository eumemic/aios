"""Unit tests for the typed connector-capability descriptor.

``ConnectorCapabilities`` is a *tools_schema sibling* on the connector-type
catalog row: a typed richness descriptor whose every field is a present/absent
typed sub-object (a declared KIND), never a boolean flag.  These tests pin the
"variation is a KIND, not a flag" invariant — the bool-bag is unrepresentable,
illegal-combo (capability on but parameters absent) is unrepresentable, and an
empty descriptor is the conservative/plain-text rendering floor.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aios.models.connectors import (
    ConnectorCapabilities,
    DraftStreaming,
    NativeButtons,
)


def test_empty_descriptor_is_the_floor() -> None:
    """An empty (or row-default ``{}``) descriptor reads as "no declared
    capabilities" — every sub-descriptor absent, the conservative floor."""
    caps = ConnectorCapabilities.model_validate({})
    assert caps.draft_streaming is None
    assert caps.native_buttons is None


def test_draft_streaming_present_carries_its_parameters() -> None:
    caps = ConnectorCapabilities.model_validate({"draft_streaming": {"overflow_limit": 4000}})
    assert caps.draft_streaming is not None
    assert caps.draft_streaming.overflow_limit == 4000
    # The other KIND stays absent — independent present/absent objects.
    assert caps.native_buttons is None


def test_draft_streaming_overflow_limit_defaults_to_unbounded() -> None:
    caps = ConnectorCapabilities.model_validate({"draft_streaming": {}})
    assert caps.draft_streaming is not None
    assert caps.draft_streaming.overflow_limit is None  # unbounded


def test_native_buttons_present_carries_its_parameters() -> None:
    caps = ConnectorCapabilities.model_validate({"native_buttons": {"max_buttons": 5}})
    assert caps.native_buttons is not None
    assert caps.native_buttons.max_buttons == 5
    assert caps.draft_streaming is None


def test_native_buttons_rejects_extra_keys() -> None:
    """``extra="forbid"`` on the sub-descriptor rejects drifted parameters."""
    with pytest.raises(ValidationError):
        ConnectorCapabilities.model_validate({"native_buttons": {"max_buttons": 5, "bogus": 1}})


def test_capabilities_rejects_extra_top_level_keys() -> None:
    with pytest.raises(ValidationError):
        ConnectorCapabilities.model_validate({"supports_draft_streaming": True})


def test_bool_where_object_expected_is_unrepresentable() -> None:
    """A bool where a KIND sub-object is expected raises — proving the
    boolean-bag (``supports_draft_streaming: bool``) is unrepresentable."""
    with pytest.raises(ValidationError):
        ConnectorCapabilities.model_validate({"draft_streaming": True})


def test_native_buttons_requires_max_buttons() -> None:
    """``max_buttons`` is a required KIND parameter — capability-on but
    parameters-missing is unrepresentable rather than runtime-guarded."""
    with pytest.raises(ValidationError):
        ConnectorCapabilities.model_validate({"native_buttons": {}})


def test_both_kinds_present() -> None:
    caps = ConnectorCapabilities.model_validate(
        {
            "draft_streaming": {"overflow_limit": 2000},
            "native_buttons": {"max_buttons": 3},
        }
    )
    assert caps.draft_streaming == DraftStreaming(overflow_limit=2000)
    assert caps.native_buttons == NativeButtons(max_buttons=3)


def test_model_dump_round_trips() -> None:
    caps = ConnectorCapabilities(
        draft_streaming=DraftStreaming(overflow_limit=4000),
        native_buttons=NativeButtons(max_buttons=5),
    )
    dumped = caps.model_dump()
    assert ConnectorCapabilities.model_validate(dumped) == caps
