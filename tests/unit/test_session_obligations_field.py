"""unify-obligations #8 (#1526): the Session HTTP view exposes the owed set as
``Session.obligations: list[Obligation]`` — the off-terminology ``owed_requests``
field and the duplicate ``OwedRequest`` model are gone (clean break, no alias).
"""

from __future__ import annotations

from datetime import UTC, datetime

import aios.models.sessions as sessions_models
import aios.services.sessions as sessions_service
from aios.models.sessions import Obligation, Session


def test_owed_request_model_is_deleted() -> None:
    """``OwedRequest`` collapsed into ``Obligation`` — the duplicate name is gone."""
    assert not hasattr(sessions_models, "OwedRequest")


def test_compute_owed_requests_renamed_to_compute_obligations() -> None:
    assert hasattr(sessions_service, "compute_obligations")
    assert not hasattr(sessions_service, "compute_owed_requests")


def test_session_exposes_obligations_field_typed_as_obligation() -> None:
    """The owed set is ``Session.obligations: list[Obligation]``; no ``owed_requests``."""
    assert "obligations" in Session.model_fields
    assert "owed_requests" not in Session.model_fields

    ob = Obligation(
        request_id="req-1",
        caller_kind="run",
        caller_id="run-1",
        opened_at=datetime(2025, 1, 1, tzinfo=UTC),
        summary="do the thing",
    )
    session = Session(
        id="s1",
        agent_id="a1",
        environment_id="e1",
        agent_version=1,
        title=None,
        metadata={},
        status="idle",
        stop_reason=None,
        obligations=[ob],
        last_event_seq=0,
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
        updated_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert session.obligations == [ob]
    # The serialized HTTP view carries the on-terminology key.
    dumped = session.model_dump()
    assert "obligations" in dumped
    assert "owed_requests" not in dumped


class TestComputeObligationsBatched:
    """#1561: ``compute_obligations`` must issue ONE DB round-trip for a batch
    of N sessions — the batched dual of ``compute_awaiting`` — not one query
    per session (the old N+1 loop)."""

    async def test_one_round_trip_for_batch(self) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        sessions = [
            Session(
                id=f"s{i}",
                agent_id="a1",
                environment_id="e1",
                agent_version=1,
                title=None,
                metadata={},
                status="idle",
                stop_reason=None,
                last_event_seq=0,
                created_at=datetime(2025, 1, 1, tzinfo=UTC),
                updated_at=datetime(2025, 1, 1, tzinfo=UTC),
            )
            for i in range(5)
        ]

        conn = MagicMock()

        class _AcquireCtx:
            async def __aenter__(self) -> MagicMock:
                return conn

            async def __aexit__(self, *_: object) -> bool:
                return False

        pool = MagicMock()
        pool.acquire = MagicMock(return_value=_AcquireCtx())

        batch = AsyncMock(return_value={"s0": [], "s2": []})
        with patch("aios.services.sessions.queries.get_open_obligations_batch", batch):
            out = await sessions_service.compute_obligations(pool, sessions, account_id="acct-1")

        # Exactly one round-trip: the batch query is called once with every id.
        assert batch.await_count == 1
        _, kwargs = batch.call_args
        passed_ids = batch.call_args.args[1]
        assert sorted(passed_ids) == [f"s{i}" for i in range(5)]
        assert kwargs["account_id"] == "acct-1"
        assert out == {"s0": [], "s2": []}

    async def test_empty_sessions_no_query(self) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        pool = MagicMock()
        batch = AsyncMock(return_value={})
        with patch("aios.services.sessions.queries.get_open_obligations_batch", batch):
            out = await sessions_service.compute_obligations(pool, [], account_id="acct-1")
        assert out == {}
        assert batch.await_count == 0
