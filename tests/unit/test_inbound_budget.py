"""Unit tests for the per-counterparty inbound rate/cost budget (#1504).

The inbound-admission gate (#1500) decides *whether* a counterparty may talk;
this budget bounds *how much*. Once admitted, every inbound costs one model
inference, so an admitted-but-abusive counterparty can drive unbounded spend.
This suite pins:

* the budget helper's disabled-by-default short-circuit (threshold 0 ⇒ admit
  with NO query — byte-identical to pre-feature behavior);
* the rolling-window admit/throttle decision (under threshold ⇒ admit, at/over
  ⇒ throttle), for both the ``orig_channel`` (chat) and ``session_id`` keys;
* ``handle_inbound`` dropping an over-budget counterparty with
  ``RATE_LIMITED`` BEFORE any side effect (no ``resolve_target_session``, no
  staging, no append, no ``defer_wake``);
* the router mapping ``rate_limited`` → 429, and the regression pin that
  ``_is_fatal_inbound_status(429) is False`` (a throttle is a routine drop,
  never a crash-restart of the connector container).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.api.routers.connectors import _inbound_drop_error
from aios.config import Settings, get_settings
from aios.errors import RateLimitedError
from aios.models.connections import Connection
from aios.models.inbound_policy import AllowAll
from aios.services import inbound_budget
from aios.services.inbound import (
    InboundDrop,
    InboundResult,
    handle_inbound,
)
from aios.services.inbound_budget import (
    check_inbound_budget,
    check_inbound_budget_session,
    inbound_orig_channel,
)
from aios_connector_http.runner import _is_fatal_inbound_status

# ─── orig_channel key matches the stamped event key ──────────────────────────


def test_orig_channel_matches_append_with_dedup_shape() -> None:
    # Mirrors ``_append_with_dedup``'s
    # ``f"{connector}/{external_account_id}/{chat_id}"`` so the window counts
    # exactly the rows the admitted-inbound append writes.
    assert (
        inbound_orig_channel("signal", "+15551234567", "group/abc==")
        == "signal/+15551234567/group/abc=="
    )


# ─── disabled-by-default: short-circuit admit with NO query ──────────────────


def _settings(*, window: int, threshold: int) -> Settings:
    return get_settings().model_copy(
        update={
            "inbound_rate_window_seconds": window,
            "inbound_rate_max_per_window": threshold,
        }
    )


async def test_disabled_by_default_admits_without_query() -> None:
    """Default (threshold 0) ⇒ admit with no DB read — byte-identical path."""
    pool = MagicMock()
    pool.acquire = MagicMock(side_effect=AssertionError("must not query when disabled"))
    with patch.object(
        inbound_budget, "get_settings", return_value=_settings(window=0, threshold=0)
    ):
        assert (
            await check_inbound_budget(
                pool,
                account_id="acc_1",
                connector="signal",
                external_account_id="+1",
                chat_id="peer",
            )
            is True
        )
    pool.acquire.assert_not_called()


async def test_threshold_zero_disables_even_with_window_set() -> None:
    pool = MagicMock()
    pool.acquire = MagicMock(side_effect=AssertionError("must not query"))
    with patch.object(
        inbound_budget, "get_settings", return_value=_settings(window=60, threshold=0)
    ):
        assert (
            await check_inbound_budget_session(pool, account_id="acc_1", session_id="ses_1") is True
        )


# ─── rolling-window decision ─────────────────────────────────────────────────


def _pool_returning_count(count: int) -> MagicMock:
    pool = MagicMock()

    class _Cm:
        async def __aenter__(self) -> Any:
            conn = MagicMock()
            conn.fetchval = AsyncMock(return_value=count)
            return conn

        async def __aexit__(self, *_a: Any) -> None:
            return None

    pool.acquire.return_value = _Cm()
    return pool


@pytest.mark.parametrize(
    ("recent", "threshold", "expected"),
    [
        (0, 3, True),  # empty window
        (2, 3, True),  # under threshold
        (3, 3, False),  # at threshold ⇒ this one tips over
        (9, 3, False),  # over threshold
    ],
)
async def test_chat_budget_decision(recent: int, threshold: int, expected: bool) -> None:
    pool = _pool_returning_count(recent)
    with patch.object(
        inbound_budget, "get_settings", return_value=_settings(window=3600, threshold=threshold)
    ):
        got = await check_inbound_budget(
            pool,
            account_id="acc_1",
            connector="signal",
            external_account_id="+1",
            chat_id="peer",
        )
    assert got is expected


@pytest.mark.parametrize(
    ("recent", "threshold", "expected"),
    [(0, 2, True), (1, 2, True), (2, 2, False), (5, 2, False)],
)
async def test_session_budget_decision(recent: int, threshold: int, expected: bool) -> None:
    pool = _pool_returning_count(recent)
    with patch.object(
        inbound_budget, "get_settings", return_value=_settings(window=3600, threshold=threshold)
    ):
        got = await check_inbound_budget_session(pool, account_id="acc_1", session_id="ses_1")
    assert got is expected


# ─── router status mapping + the non-fatal pin ───────────────────────────────


def test_rate_limited_maps_to_429() -> None:
    err = _inbound_drop_error(InboundDrop.RATE_LIMITED)
    assert isinstance(err, RateLimitedError)
    assert err.status_code == 429
    assert err.detail == {"drop_reason": "rate_limited"}


def test_rate_limited_status_is_not_fatal() -> None:
    """Regression pin (mirrors the spine's DENIED_BY_POLICY → 422 pin): the
    chosen throttle status must be non-fatal so a throttle never crash-restarts
    the connector container."""
    status = _inbound_drop_error(InboundDrop.RATE_LIMITED).status_code
    assert status == 429
    assert _is_fatal_inbound_status(status) is False
    # The two non-fatal candidates the issue calls out both hold.
    assert _is_fatal_inbound_status(429) is False
    assert _is_fatal_inbound_status(422) is False


# ─── handle_inbound: over-budget drops before any side effect ────────────────


def _connection() -> Connection:
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
        archived_at=None,
        inbound_policy=AllowAll(),
        inbound_policy_effective=AllowAll(),
    )


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


async def test_over_budget_drops_before_resolution_and_side_effects() -> None:
    """An admitted-but-over-budget counterparty returns RATE_LIMITED with zero
    resolution / staging / append / wake."""
    get_connection = AsyncMock(return_value=_connection())
    resolve_target_session = AsyncMock()
    stage = AsyncMock()
    append_event = AsyncMock()
    defer_wake = AsyncMock()
    # Admitted (AllowAll), but the budget says over.
    check_budget = AsyncMock(return_value=False)

    with (
        patch("aios.services.inbound.queries.get_connection", get_connection),
        patch("aios.services.inbound.check_inbound_budget", check_budget),
        patch("aios_connectors.resolver.resolve_target_session", resolve_target_session),
        patch("aios.services.inbound.stage_inbound_attachments", stage),
        patch("aios.services.inbound.queries.append_event", append_event),
        patch("aios.services.inbound.defer_wake", defer_wake),
    ):
        result = await handle_inbound(
            _fake_pool(),
            account_id="acc_1",
            connection_id="conn_1",
            event_id="evt_1",
            chat_id="peer",
            sender={"id": "peer"},
            content="flood",
        )

    assert result == InboundResult(None, None, InboundDrop.RATE_LIMITED, False)
    check_budget.assert_awaited_once()
    resolve_target_session.assert_not_called()
    stage.assert_not_called()
    append_event.assert_not_called()
    defer_wake.assert_not_called()


async def test_within_budget_passes_into_resolution() -> None:
    """An admitted, within-budget sender flows past the budget into resolution
    (proving the gate is not a blanket throttle)."""
    from aios_connectors.resolver import ResolveDrop, ResolveResult

    get_connection = AsyncMock(return_value=_connection())
    resolve_target_session = AsyncMock(
        return_value=ResolveResult(session_id=None, drop=ResolveDrop.DETACHED)
    )
    check_budget = AsyncMock(return_value=True)

    with (
        patch("aios.services.inbound.queries.get_connection", get_connection),
        patch("aios.services.inbound.check_inbound_budget", check_budget),
        patch("aios_connectors.resolver.resolve_target_session", resolve_target_session),
    ):
        result = await handle_inbound(
            _fake_pool(),
            account_id="acc_1",
            connection_id="conn_1",
            event_id="evt_2",
            chat_id="peer",
            sender={"id": "peer"},
            content="hi",
        )

    check_budget.assert_awaited_once()
    resolve_target_session.assert_awaited_once()
    assert result.drop_reason is InboundDrop.DETACHED


# ─── #1558: the inference-bearing union — wake-bearing lifecycle is counted ───


def _pool_capturing_conn() -> tuple[MagicMock, MagicMock]:
    """A pool whose acquired conn is the SAME ``MagicMock`` every time, so a
    test can assert on the ``fetchval`` call args (the built SQL + params).
    Returns ``(pool, conn)``."""
    conn = MagicMock()
    conn.fetchval = AsyncMock(return_value=0)
    pool = MagicMock()

    class _Cm:
        async def __aenter__(self) -> Any:
            return conn

        async def __aexit__(self, *_a: Any) -> None:
            return None

    pool.acquire.return_value = _Cm()
    return pool, conn


async def test_chat_counter_sql_includes_wake_lifecycle_arm() -> None:
    """Predicate-level (master-failing): the chat-grain counter's SQL must
    count the wake-bearing ``kind='lifecycle'`` arm, not just message/role.

    On master the query has ONLY the ``kind = 'message' AND role='user'`` arm,
    so the ``kind = 'lifecycle'`` ∧ ``wake`` assertion fails."""
    pool, conn = _pool_capturing_conn()
    await inbound_budget._count_recent_inbounds(
        pool, account_id="acc_1", orig_channel="signal/+1/peer", window_seconds=3600
    )
    sql = _await_args(conn.fetchval).args[0]
    # The new inference-bearing union arm.
    assert "kind = 'lifecycle'" in sql
    assert "data->>'wake')::boolean IS TRUE" in sql
    # The original admitted-inbound arm is preserved.
    assert "kind = 'message'" in sql
    assert "data->>'role' = 'user'" in sql


async def test_session_counter_sql_includes_wake_lifecycle_arm() -> None:
    """Predicate-level (master-failing): same for the session-grain counter."""
    pool, conn = _pool_capturing_conn()
    await inbound_budget._count_recent_session_inbounds(
        pool, account_id="acc_1", session_id="ses_1", window_seconds=3600
    )
    sql = _await_args(conn.fetchval).args[0]
    assert "kind = 'lifecycle'" in sql
    assert "data->>'wake')::boolean IS TRUE" in sql
    assert "kind = 'message'" in sql
    assert "data->>'role' = 'user'" in sql


def test_both_counters_share_one_predicate_fragment() -> None:
    """The inference-bearing shape is single-sourced (one shared SQL fragment),
    not duplicated-by-vigilance, so the two counters can never drift apart."""
    frag = inbound_budget._INFERENCE_BEARING_PREDICATE
    assert "kind = 'message'" in frag
    assert "data->>'role' = 'user'" in frag
    assert "kind = 'lifecycle'" in frag
    assert "(data->>'wake')::boolean IS TRUE" in frag


# ─── #1558: route-level — the wake-bearing lifecycle write is stamped ─────────


def _runtime_auth() -> tuple[str, str, str, list[str] | None]:
    # (token, connector, account_id, connection_ids allowlist)
    return ("tok", "signal", "acc_1", None)


def _route_pool() -> MagicMock:
    """A pool whose ``acquire()`` yields a conn usable for the routes' inline
    ``async with pool.acquire() as conn`` lookups (all DB calls are patched at
    the ``queries`` module, so the conn itself is an inert MagicMock)."""
    pool = MagicMock()
    conn = MagicMock()

    class _Cm:
        async def __aenter__(self) -> Any:
            return conn

        async def __aexit__(self, *_a: Any) -> None:
            return None

    pool.acquire.return_value = _Cm()
    return pool


def _fake_event() -> MagicMock:
    ev = MagicMock()
    ev.id = "evt_appended"
    return ev


def _await_args(mock: AsyncMock) -> Any:
    """Return ``mock.await_args`` narrowed to non-``None`` (mypy/asserts)."""
    call = mock.await_args
    assert call is not None
    return call


async def test_session_lifecycle_wake_stamps_wake_marker() -> None:
    """Route-level (master-failing): a ``wake=True`` session-lifecycle stamps
    ``wake: True`` on the appended payload so the session-grain budget can meter
    it. On master the payload has no ``wake`` key ⇒ this fails."""
    from aios.api.routers.connectors import (
        RuntimeSessionLifecycleRequest,
        post_runtime_session_lifecycle,
    )

    body = RuntimeSessionLifecycleRequest(
        connection_id="conn_1",
        session_id="ses_1",
        event="connector_delivery_failed",
        wake=True,
    )
    append_event = AsyncMock(return_value=_fake_event())

    with (
        patch(
            "aios.api.routers.connectors.queries.get_connection",
            AsyncMock(return_value=_connection()),
        ),
        patch("aios.api.routers.connectors._check_runtime_scope", MagicMock()),
        patch("aios.api.routers.connectors._check_runtime_connection_scope", MagicMock()),
        patch(
            "aios.api.routers.connectors.queries.is_session_bound_to_connection",
            AsyncMock(return_value=True),
        ),
        # Budget enabled, and under threshold ⇒ admit (so we reach the append).
        patch(
            "aios.api.routers.connectors.check_inbound_budget_session", AsyncMock(return_value=True)
        ),
        patch("aios.api.routers.connectors.sessions_service.append_event", append_event),
        patch("aios.api.routers.connectors.defer_wake", AsyncMock()),
    ):
        await post_runtime_session_lifecycle(body, _route_pool(), _runtime_auth())

    call = _await_args(append_event)
    payload = call.args[3]
    assert payload["wake"] is True


async def test_session_lifecycle_no_wake_omits_marker() -> None:
    """Regression pin: a ``wake=False`` session-lifecycle writes NO ``wake``
    marker (the free path stays byte-identical / uncounted)."""
    from aios.api.routers.connectors import (
        RuntimeSessionLifecycleRequest,
        post_runtime_session_lifecycle,
    )

    body = RuntimeSessionLifecycleRequest(
        connection_id="conn_1",
        session_id="ses_1",
        event="connector_message_delivered",
        wake=False,
    )
    append_event = AsyncMock(return_value=_fake_event())
    check = AsyncMock(return_value=True)

    with (
        patch(
            "aios.api.routers.connectors.queries.get_connection",
            AsyncMock(return_value=_connection()),
        ),
        patch("aios.api.routers.connectors._check_runtime_scope", MagicMock()),
        patch("aios.api.routers.connectors._check_runtime_connection_scope", MagicMock()),
        patch(
            "aios.api.routers.connectors.queries.is_session_bound_to_connection",
            AsyncMock(return_value=True),
        ),
        patch("aios.api.routers.connectors.check_inbound_budget_session", check),
        patch("aios.api.routers.connectors.sessions_service.append_event", append_event),
        patch("aios.api.routers.connectors.defer_wake", AsyncMock()),
    ):
        await post_runtime_session_lifecycle(body, _route_pool(), _runtime_auth())

    call = _await_args(append_event)
    payload = call.args[3]
    assert "wake" not in payload
    # The no-wake path never even consults the budget.
    check.assert_not_called()


async def test_chat_lifecycle_wake_stamps_wake_and_orig_channel() -> None:
    """Route-level (master-failing): a ``wake=True`` chat-lifecycle stamps both
    ``wake: True`` AND ``orig_channel`` (= ``inbound_orig_channel(...)``) so the
    chat-grain budget counts it on the same key the inbound path uses. On
    master neither is passed ⇒ this fails."""
    from aios.api.routers.connectors import (
        RuntimeChatLifecycleRequest,
        post_runtime_chat_lifecycle,
    )

    connection = _connection()
    body = RuntimeChatLifecycleRequest(
        connection_id="conn_1",
        chat_id="peer",
        event="connector_delivery_failed",
        wake=True,
    )
    append_event = AsyncMock(return_value=_fake_event())

    with (
        patch(
            "aios.api.routers.connectors.queries.get_connection", AsyncMock(return_value=connection)
        ),
        patch("aios.api.routers.connectors._check_runtime_scope", MagicMock()),
        patch("aios.api.routers.connectors._check_runtime_connection_scope", MagicMock()),
        patch(
            "aios.api.routers.connectors.queries.get_chat_session_row",
            AsyncMock(return_value=("peer", "ses_resolved", datetime.now(UTC))),
        ),
        patch("aios.api.routers.connectors.check_inbound_budget", AsyncMock(return_value=True)),
        patch("aios.api.routers.connectors.sessions_service.append_event", append_event),
        patch("aios.api.routers.connectors.defer_wake", AsyncMock()),
    ):
        await post_runtime_chat_lifecycle(body, _route_pool(), _runtime_auth())

    call = _await_args(append_event)
    payload = call.args[3]
    assert payload["wake"] is True
    expected_oc = inbound_orig_channel(connection.connector, connection.external_account_id, "peer")
    assert call.kwargs["orig_channel"] == expected_oc


async def test_chat_lifecycle_no_wake_omits_wake_and_orig_channel() -> None:
    """Regression pin: a ``wake=False`` chat-lifecycle writes no ``wake`` marker
    and passes no ``orig_channel`` (free path stays uncounted)."""
    from aios.api.routers.connectors import (
        RuntimeChatLifecycleRequest,
        post_runtime_chat_lifecycle,
    )

    body = RuntimeChatLifecycleRequest(
        connection_id="conn_1",
        chat_id="peer",
        event="connector_message_delivered",
        wake=False,
    )
    append_event = AsyncMock(return_value=_fake_event())
    check = AsyncMock(return_value=True)

    with (
        patch(
            "aios.api.routers.connectors.queries.get_connection",
            AsyncMock(return_value=_connection()),
        ),
        patch("aios.api.routers.connectors._check_runtime_scope", MagicMock()),
        patch("aios.api.routers.connectors._check_runtime_connection_scope", MagicMock()),
        patch(
            "aios.api.routers.connectors.queries.get_chat_session_row",
            AsyncMock(return_value=("peer", "ses_resolved", datetime.now(UTC))),
        ),
        patch("aios.api.routers.connectors.check_inbound_budget", check),
        patch("aios.api.routers.connectors.sessions_service.append_event", append_event),
        patch("aios.api.routers.connectors.defer_wake", AsyncMock()),
    ):
        await post_runtime_chat_lifecycle(body, _route_pool(), _runtime_auth())

    call = _await_args(append_event)
    payload = call.args[3]
    assert "wake" not in payload
    assert call.kwargs.get("orig_channel") is None
    check.assert_not_called()
