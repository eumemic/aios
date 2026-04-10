"""Tests for prefixed ULID id generation."""

from __future__ import annotations

import pytest

from aios.ids import AGENT, CREDENTIAL, ENVIRONMENT, EVENT, SESSION, make_id, split_id


class TestMakeId:
    def test_returns_prefixed_ulid(self) -> None:
        new_id = make_id(AGENT)
        prefix, ulid_part = new_id.split("_", 1)
        assert prefix == "agent"
        assert len(ulid_part) == 26

    def test_each_call_is_unique(self) -> None:
        ids = {make_id(SESSION) for _ in range(100)}
        assert len(ids) == 100

    @pytest.mark.parametrize(
        "prefix",
        [AGENT, ENVIRONMENT, SESSION, EVENT, CREDENTIAL],
    )
    def test_all_canonical_prefixes_accepted(self, prefix: str) -> None:
        result = make_id(prefix)
        assert result.startswith(f"{prefix}_")

    def test_unknown_prefix_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown id prefix"):
            make_id("widget")

    def test_typoed_prefix_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown id prefix"):
            make_id("agnet")


class TestSplitId:
    def test_splits_well_formed_id(self) -> None:
        new_id = make_id(EVENT)
        prefix, ulid_part = split_id(new_id)
        assert prefix == "evt"
        assert len(ulid_part) == 26

    def test_rejects_missing_underscore(self) -> None:
        with pytest.raises(ValueError, match="malformed prefixed id"):
            split_id("agent01HQR2K7VXBZ9MNPL3WYCT8F")

    def test_rejects_unknown_prefix(self) -> None:
        with pytest.raises(ValueError, match="malformed prefixed id"):
            split_id("widget_01HQR2K7VXBZ9MNPL3WYCT8F")

    def test_rejects_short_ulid(self) -> None:
        with pytest.raises(ValueError, match="malformed prefixed id"):
            split_id("agent_TOOSHORT")
