"""Tests for prefixed ULID id generation."""

from __future__ import annotations

import pytest

from aios.ids import (
    AGENT,
    CREDENTIAL,
    ENVIRONMENT,
    EVENT,
    SESSION,
    WORKFLOW_RUN,
    is_run_owner_id,
    make_id,
    split_id,
)


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


class TestMakeIdDeterministic:
    """The additive ``body=`` path for content-addressed (e.g. workflow child) ids."""

    def test_same_body_yields_same_id(self) -> None:
        body = bytes(range(16))
        assert make_id(SESSION, body=body) == make_id(SESSION, body=body)

    def test_deterministic_id_is_split_id_parseable(self) -> None:
        prefix, ulid_part = split_id(make_id(SESSION, body=bytes(range(16))))
        assert prefix == "sess"
        assert len(ulid_part) == 26

    def test_different_bodies_yield_different_ids(self) -> None:
        assert make_id(SESSION, body=bytes(range(16))) != make_id(SESSION, body=bytes(range(1, 17)))

    def test_body_must_be_16_bytes(self) -> None:
        with pytest.raises(ValueError, match="must be exactly 16 bytes"):
            make_id(SESSION, body=b"too short")


class TestIsRunOwnerId:
    """The intrinsic owner-kind discriminator the sandbox registry routes teardown
    on (issue #995): a single source of truth for "is this owner a workflow run?"
    so a future owner prefix can't silently fall through the run-vs-session fork."""

    def test_true_for_a_workflow_run_id(self) -> None:
        assert is_run_owner_id(make_id(WORKFLOW_RUN)) is True

    def test_false_for_a_session_id(self) -> None:
        assert is_run_owner_id(make_id(SESSION)) is False

    def test_false_for_other_prefixes(self) -> None:
        # An unrelated owner-shaped id must NOT be misclassified as a run.
        for prefix in (AGENT, ENVIRONMENT, EVENT):
            assert is_run_owner_id(make_id(prefix)) is False

    def test_requires_the_separator_not_just_the_prefix_chars(self) -> None:
        # A literal ``wfr`` substring without the ``wfr_`` boundary is not a run id.
        assert is_run_owner_id("wfrog_01HQR2K7VXBZ9MNPL3WYCT8F") is False


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
