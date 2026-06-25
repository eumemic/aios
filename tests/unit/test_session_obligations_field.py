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
