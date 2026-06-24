"""Unit tests for the inbound-admission gate (#1500 — the spine).

The connector inbound path was historically fail-open: every message a
connector daemon posted for a bound connection was appended to a session and
woke the model, with no predicate asking "is this sender allowed to talk to
this agent." This suite pins the gate inserted into ``handle_inbound``:

* the pure ``_admits`` predicate over the discriminated policy union;
* the resolve query's NULL→server-default ``DenyAll`` (fail-closed);
* the gate dropping an unadmitted sender BEFORE any side effect (no
  ``resolve_target_session``, no attachment staging, no ``append_event``, no
  ``defer_wake``) — the *fresh-connection* and *single_session* acceptance
  cases; and
* the router mapping ``denied_by_policy`` → 422 (``ValidationError``), never
  403 / 5xx.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError as PydanticValidationError

from aios.api.routers.connectors import _inbound_drop_error
from aios.db.queries.inbound_policy import resolve_effective_inbound_policy
from aios.errors import ValidationError
from aios.models.connections import Connection
from aios.models.inbound_policy import AllowAll, AllowList, DenyAll
from aios.services.inbound import (
    InboundDrop,
    InboundResult,
    _admits,
    handle_inbound,
)

# ─── _admits: the pure predicate ─────────────────────────────────────────────


def test_admits_allow_all() -> None:
    assert _admits(AllowAll(), "anyone") is True


def test_admits_allow_list_member() -> None:
    assert _admits(AllowList(chat_ids=["a", "b"]), "b") is True


def test_admits_allow_list_non_member() -> None:
    assert _admits(AllowList(chat_ids=["a", "b"]), "stranger") is False


def test_admits_allow_list_slash_bearing_chat_id() -> None:
    # A Signal group id contains '/'. It must be admitted intact — the backfill
    # keeps the whole remainder rather than truncating at the third '/' segment.
    group_id = "group/abc123=="
    assert _admits(AllowList(chat_ids=[group_id]), group_id) is True


def test_admits_deny_all() -> None:
    assert _admits(DenyAll(), "anyone") is False


# ─── AllowList validation ────────────────────────────────────────────────────


def test_allow_list_empty_is_rejected() -> None:
    # An empty list must 422 at validation time — never a silent deny-all.
    with pytest.raises(PydanticValidationError):
        AllowList(chat_ids=[])


# ─── resolve_effective_inbound_policy: NULL → server default DenyAll ──────────


def _connection(*, inbound_policy: Any = None, archived_at: Any = None) -> Connection:
    now = datetime.now(UTC)
    return Connection(
        id="conn_1",
        connector="signal",
        external_account_id="+15551234567",
        session_id="ses_bound",
        session_template_id=None,
        metadata={},
        secrets_set=False,
        created_at=now,
        attached_at=now,
        updated_at=now,
        archived_at=archived_at,
        inbound_policy=inbound_policy,
    )


async def test_resolve_null_policy_is_deny_all() -> None:
    conn = _connection(inbound_policy=None)
    policy = await resolve_effective_inbound_policy(
        MagicMock(), connection=conn, account_id="acc_1"
    )
    assert isinstance(policy, DenyAll)


async def test_resolve_passes_through_stored_policy() -> None:
    stored = AllowList(chat_ids=["op_chat"])
    conn = _connection(inbound_policy=stored)
    policy = await resolve_effective_inbound_policy(
        MagicMock(), connection=conn, account_id="acc_1"
    )
    assert policy is stored


# ─── router status mapping: denied_by_policy → 422 ───────────────────────────


def test_denied_by_policy_maps_to_validation_error_422() -> None:
    err = _inbound_drop_error("denied_by_policy")
    assert isinstance(err, ValidationError)
    assert err.status_code == 422
    assert err.detail == {"drop_reason": "denied_by_policy"}


# ─── handle_inbound gate: fail-closed before any side effect ─────────────────


@pytest.fixture
def side_effect_spies() -> dict[str, MagicMock]:
    """Patch every post-gate side effect so a denial can be proven inert."""
    return {}


def _patch_gate_side_effects(stored_policy: Any) -> Any:
    """Context patching get_connection + every downstream side effect.

    Returns a tuple of (context-manager, spies dict). The spies let the test
    assert that NOTHING after the gate ran when the sender is denied.
    """
    conn = _connection(inbound_policy=stored_policy)

    get_connection = AsyncMock(return_value=conn)
    resolve_target_session = AsyncMock()
    stage = AsyncMock()
    append_event = AsyncMock()
    defer_wake = AsyncMock()

    spies = {
        "resolve_target_session": resolve_target_session,
        "stage_inbound_attachments": stage,
        "append_event": append_event,
        "defer_wake": defer_wake,
    }

    patches = [
        patch("aios.services.inbound.queries.get_connection", get_connection),
        patch(
            "aios_connectors.resolver.resolve_target_session",
            resolve_target_session,
        ),
        patch(
            "aios.services.inbound.stage_inbound_attachments",
            stage,
        ),
        patch("aios.services.inbound.queries.append_event", append_event),
        patch("aios.services.inbound.defer_wake", defer_wake),
    ]
    return patches, spies


def _fake_pool() -> MagicMock:
    pool = MagicMock()

    class _AsyncCm:
        def __init__(self, value: Any = None) -> None:
            self._value = value

        async def __aenter__(self) -> Any:
            return self._value

        async def __aexit__(self, *_args: Any) -> None:
            return None

    conn = MagicMock()
    conn.transaction.return_value = _AsyncCm()
    pool.acquire.return_value = _AsyncCm(conn)
    return pool


async def test_fresh_connection_null_policy_denies_unknown_sender() -> None:
    """A NULL-policy (fresh) connection denies an unknown chat_id with
    DENIED_BY_POLICY, and runs zero downstream side effects."""
    patches, spies = _patch_gate_side_effects(stored_policy=None)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = await handle_inbound(
            _fake_pool(),
            account_id="acc_1",
            connection_id="conn_1",
            event_id="evt_1",
            chat_id="stranger",
            sender={"id": "stranger"},
            content="hello",
        )

    assert result == InboundResult(None, None, InboundDrop.DENIED_BY_POLICY, False)
    # No session resolution, no staging, no append, no wake.
    spies["resolve_target_session"].assert_not_called()
    spies["stage_inbound_attachments"].assert_not_called()
    spies["append_event"].assert_not_called()
    spies["defer_wake"].assert_not_called()


async def test_single_session_drops_stranger_before_shared_append() -> None:
    """A single_session connection whose policy is AllowList([operator]) drops a
    stranger BEFORE any append to the shared bound session."""
    policy = AllowList(chat_ids=["operator_chat"])
    patches, spies = _patch_gate_side_effects(stored_policy=policy)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = await handle_inbound(
            _fake_pool(),
            account_id="acc_1",
            connection_id="conn_1",
            event_id="evt_2",
            chat_id="stranger",
            sender={"id": "stranger"},
            content="injection attempt",
        )

    assert result.drop_reason is InboundDrop.DENIED_BY_POLICY
    spies["resolve_target_session"].assert_not_called()
    spies["append_event"].assert_not_called()
    spies["defer_wake"].assert_not_called()


async def test_allowed_sender_passes_gate_into_resolution() -> None:
    """An admitted sender flows past the gate into resolve_target_session
    (proving the gate is not a blanket deny)."""
    policy = AllowList(chat_ids=["operator_chat", "group/abc=="])
    patches, spies = _patch_gate_side_effects(stored_policy=policy)
    # Make resolution drop DETACHED so we stop right after the gate without
    # needing the full append machinery — we only assert the gate let us
    # through to resolution.
    from aios_connectors.resolver import ResolveDrop, ResolveResult

    spies["resolve_target_session"].return_value = ResolveResult(
        session_id=None, drop=ResolveDrop.DETACHED
    )

    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = await handle_inbound(
            _fake_pool(),
            account_id="acc_1",
            connection_id="conn_1",
            event_id="evt_3",
            chat_id="group/abc==",
            sender={"id": "grp"},
            content="hi from the allowed group",
        )

    # The slash-bearing admitted chat_id reached resolution.
    spies["resolve_target_session"].assert_awaited_once()
    assert result.drop_reason is InboundDrop.DETACHED
