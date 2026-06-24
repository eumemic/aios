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
    err = _inbound_drop_error("rate_limited")
    assert isinstance(err, RateLimitedError)
    assert err.status_code == 429
    assert err.detail == {"drop_reason": "rate_limited"}


def test_rate_limited_status_is_not_fatal() -> None:
    """Regression pin (mirrors the spine's DENIED_BY_POLICY → 422 pin): the
    chosen throttle status must be non-fatal so a throttle never crash-restarts
    the connector container."""
    status = _inbound_drop_error("rate_limited").status_code
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
        patch(
            "aios.services.inbound.queries.resolve_effective_inbound_policy",
            AsyncMock(return_value=AllowAll()),
        ),
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
        patch(
            "aios.services.inbound.queries.resolve_effective_inbound_policy",
            AsyncMock(return_value=AllowAll()),
        ),
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
