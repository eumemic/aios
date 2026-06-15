"""Unit tests for the ``wake_session`` tool handler.

These tests stub the worker-process pool and DB so they don't need a
live Postgres. They cover the argument-validation surface, the
same-account permission check, the wake-depth cap, and the per-pair
rate-limit cap. Full-stack DB behavior is covered separately by the
integration tests.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

from aios.tools.wake_session import (
    WAKE_SESSION_MAX_DEPTH,
    WAKE_SESSION_MAX_PER_HOUR,
    WakeSessionArgumentError,
    WakeSessionDepthExceededError,
    WakeSessionPermissionError,
    WakeSessionRateLimitedError,
    WakeSessionTargetUnavailableError,
    wake_session_handler,
)


def _make_pool(*, target_row: dict[str, Any] | None, depth: int, recent_wakes: int) -> MagicMock:
    """Build a mock pool whose ``pool.acquire()`` yields a single shared
    conn whose ``fetchrow``/``fetchval`` walk through the handler's call
    sequence in order:

      1. fetchrow → target session row (SELECT account_id, status, archived_at)
      2. fetchrow → wake_depth read on source (COALESCE … FROM events ...)
      3. fetchval → recent-wake count (SELECT count(*) FROM events ...)
      4. append_event is patched separately via ``queries.append_event``.

    Each ``pool.acquire()`` invocation re-enters the same conn, so the
    side_effect queue is shared across the three acquire blocks in the
    handler.
    """
    pool = MagicMock()

    conn = MagicMock()
    conn.fetchrow = AsyncMock(side_effect=[target_row, {"depth": depth}])
    conn.fetchval = AsyncMock(side_effect=[recent_wakes])
    conn.execute = AsyncMock()

    class _PoolAcquireCM:
        async def __aenter__(self) -> MagicMock:
            return conn

        async def __aexit__(self, *args: Any) -> None:
            return None

    pool.acquire = lambda: _PoolAcquireCM()
    return pool


@pytest.fixture(autouse=True)
def _mock_runtime_pool(monkeypatch: Any) -> None:
    """Most tests construct their own pool — patch the runtime to use it."""

    def _fake_require_pool() -> MagicMock:
        return cast(MagicMock, _DEFAULT_POOL_HOLDER["pool"])

    monkeypatch.setattr("aios.tools.wake_session.runtime.require_pool", _fake_require_pool)


_DEFAULT_POOL_HOLDER: dict[str, Any] = {"pool": None}


@pytest.fixture
def install_pool() -> Any:
    """Install a pool that ``runtime.require_pool()`` will return."""

    def _install(pool: MagicMock) -> None:
        _DEFAULT_POOL_HOLDER["pool"] = pool

    yield _install
    _DEFAULT_POOL_HOLDER["pool"] = None


class TestWakeSessionHandlerArguments:
    async def test_missing_target_rejects(self) -> None:
        with pytest.raises(WakeSessionArgumentError, match="target_session_id"):
            await wake_session_handler("sess_01SOURCE", {"prompt": "hi"})

    async def test_missing_prompt_rejects(self) -> None:
        with pytest.raises(WakeSessionArgumentError, match="prompt"):
            await wake_session_handler("sess_01SOURCE", {"target_session_id": "sess_01TARGET"})

    async def test_empty_target_rejects(self) -> None:
        with pytest.raises(WakeSessionArgumentError, match="target_session_id"):
            await wake_session_handler("sess_01SOURCE", {"target_session_id": "", "prompt": "hi"})

    async def test_self_wake_rejects(self) -> None:
        # The error must recommend wake_self for immediate self-delivery
        # and mention schedule_wake for the delayed case (issue #703 #704).
        with pytest.raises(WakeSessionArgumentError) as exc_info:
            await wake_session_handler(
                "sess_01SAME", {"target_session_id": "sess_01SAME", "prompt": "hi"}
            )
        message = str(exc_info.value)
        assert "wake_self" in message
        assert "schedule_wake" in message


class TestWakeSessionHandlerPermission:
    async def test_unknown_target_rejects(self, monkeypatch: Any, install_pool: Any) -> None:
        install_pool(_make_pool(target_row=None, depth=0, recent_wakes=0))
        monkeypatch.setattr("aios.db.queries.append_event", AsyncMock())
        monkeypatch.setattr("aios.tools.wake_session.defer_wake", AsyncMock())
        with pytest.raises(WakeSessionArgumentError, match="not found"):
            await wake_session_handler(
                "sess_01SOURCE",
                {"target_session_id": "sess_01MISSING", "prompt": "hi"},
            )

    async def test_cross_account_rejects_without_disclosure(
        self, monkeypatch: Any, install_pool: Any
    ) -> None:
        install_pool(
            _make_pool(
                target_row={
                    "account_id": "acc_other",
                    "status": "running",
                    "archived_at": None,
                },
                depth=0,
                recent_wakes=0,
            )
        )
        monkeypatch.setattr("aios.db.queries.append_event", AsyncMock())
        monkeypatch.setattr("aios.tools.wake_session.defer_wake", AsyncMock())
        with pytest.raises(WakeSessionPermissionError, match="not found") as exc_info:
            await wake_session_handler(
                "sess_01SOURCE",
                {"target_session_id": "sess_01TARGET", "prompt": "hi"},
            )
        # Don't leak account_id-presence: detail mirrors the unknown-target shape.
        assert "account_id" not in (exc_info.value.detail or {})

    async def test_archived_target_rejects(self, monkeypatch: Any, install_pool: Any) -> None:
        install_pool(
            _make_pool(
                target_row={
                    "account_id": "acc_test_stub",
                    "status": "idle",
                    "archived_at": "2026-01-01T00:00:00",
                },
                depth=0,
                recent_wakes=0,
            )
        )
        monkeypatch.setattr("aios.db.queries.append_event", AsyncMock())
        monkeypatch.setattr("aios.tools.wake_session.defer_wake", AsyncMock())
        with pytest.raises(WakeSessionTargetUnavailableError, match="archived"):
            await wake_session_handler(
                "sess_01SOURCE",
                {"target_session_id": "sess_01TARGET", "prompt": "hi"},
            )

    async def test_errored_target_allowed(self, monkeypatch: Any, install_pool: Any) -> None:
        """Errored isn't a terminal state — a wake's user message recovers it."""
        install_pool(
            _make_pool(
                target_row={
                    "account_id": "acc_test_stub",
                    "status": "errored",
                    "archived_at": None,
                },
                depth=0,
                recent_wakes=0,
            )
        )
        mock_append = AsyncMock()
        mock_defer = AsyncMock()
        monkeypatch.setattr("aios.db.queries.append_event", mock_append)
        monkeypatch.setattr("aios.tools.wake_session.defer_wake", mock_defer)

        result = await wake_session_handler(
            "sess_01SOURCE",
            {"target_session_id": "sess_01TARGET", "prompt": "wake up"},
        )
        assert result["woken"] is True
        # Two appends: the user message + the trusted wake_lineage span.
        assert mock_append.await_count == 2
        mock_defer.assert_awaited_once()


class TestWakeSessionHandlerLimits:
    async def test_depth_at_cap_rejects(self, monkeypatch: Any, install_pool: Any) -> None:
        install_pool(
            _make_pool(
                target_row={
                    "account_id": "acc_test_stub",
                    "status": "idle",
                    "archived_at": None,
                },
                depth=WAKE_SESSION_MAX_DEPTH,
                recent_wakes=0,
            )
        )
        monkeypatch.setattr("aios.db.queries.append_event", AsyncMock())
        monkeypatch.setattr("aios.tools.wake_session.defer_wake", AsyncMock())
        with pytest.raises(WakeSessionDepthExceededError, match="wake-depth"):
            await wake_session_handler(
                "sess_01SOURCE",
                {"target_session_id": "sess_01TARGET", "prompt": "hi"},
            )

    async def test_rate_limit_at_cap_rejects(self, monkeypatch: Any, install_pool: Any) -> None:
        install_pool(
            _make_pool(
                target_row={
                    "account_id": "acc_test_stub",
                    "status": "idle",
                    "archived_at": None,
                },
                depth=0,
                recent_wakes=WAKE_SESSION_MAX_PER_HOUR,
            )
        )
        monkeypatch.setattr("aios.db.queries.append_event", AsyncMock())
        monkeypatch.setattr("aios.tools.wake_session.defer_wake", AsyncMock())
        with pytest.raises(WakeSessionRateLimitedError, match="rate limit"):
            await wake_session_handler(
                "sess_01SOURCE",
                {"target_session_id": "sess_01TARGET", "prompt": "hi"},
            )

    async def test_happy_path_appends_and_defers(self, monkeypatch: Any, install_pool: Any) -> None:
        install_pool(
            _make_pool(
                target_row={
                    "account_id": "acc_test_stub",
                    "status": "idle",
                    "archived_at": None,
                },
                depth=2,
                recent_wakes=0,
            )
        )
        mock_append = AsyncMock()
        mock_defer = AsyncMock()
        monkeypatch.setattr("aios.db.queries.append_event", mock_append)
        monkeypatch.setattr("aios.tools.wake_session.defer_wake", mock_defer)

        result = await wake_session_handler(
            "sess_01SOURCE",
            {"target_session_id": "sess_01TARGET", "prompt": "escalation summary"},
        )

        # New depth carries over source +1.
        assert result == {
            "woken": True,
            "target_session_id": "sess_01TARGET",
            "wake_depth": 3,
        }
        # The user-message append landed on the TARGET session under its
        # account_id with wake_source_session_id + wake_depth (display) metadata.
        assert mock_append.await_count == 2
        msg_call = next(c for c in mock_append.call_args_list if c.kwargs["kind"] == "message")
        kwargs = msg_call.kwargs
        assert kwargs["session_id"] == "sess_01TARGET"
        assert kwargs["account_id"] == "acc_test_stub"
        assert kwargs["kind"] == "message"
        assert kwargs["data"]["role"] == "user"
        assert kwargs["data"]["content"] == "escalation summary"
        assert kwargs["data"]["metadata"]["wake_source_session_id"] == "sess_01SOURCE"
        assert kwargs["data"]["metadata"]["wake_depth"] == 3

        mock_defer.assert_awaited_once_with(
            ANY,
            "sess_01TARGET",
            cause="agent_wake",
            account_id="acc_test_stub",
        )


class TestWakeSessionTrustedLineage:
    """The wake-depth/source cap must read provenance from a system-owned
    slot (a ``kind='span'`` lineage event), never from caller-suppliable
    user-message metadata that the operator-POST / connector paths forge.

    See issue #1083.
    """

    async def test_depth_query_filters_to_span_lineage(
        self, monkeypatch: Any, install_pool: Any
    ) -> None:
        """The source-depth read must be scoped to ``kind='span'`` lineage
        events. A user message (forgeable) must NOT be a source of truth."""
        captured: dict[str, str] = {}

        pool = MagicMock()
        conn = MagicMock()

        async def _fetchrow(sql: str, *args: Any) -> Any:
            # First fetchrow = target row; second = source depth read.
            if "account_id" in sql and "archived_at" in sql:
                return {
                    "account_id": "acc_test_stub",
                    "status": "idle",
                    "archived_at": None,
                }
            captured["depth_sql"] = sql
            return {"depth": 4}

        conn.fetchrow = AsyncMock(side_effect=_fetchrow)
        conn.fetchval = AsyncMock(return_value=0)
        conn.execute = AsyncMock()

        class _CM:
            async def __aenter__(self) -> MagicMock:
                return conn

            async def __aexit__(self, *a: Any) -> None:
                return None

        pool.acquire = lambda: _CM()
        install_pool(pool)
        monkeypatch.setattr("aios.db.queries.append_event", AsyncMock())
        monkeypatch.setattr("aios.tools.wake_session.defer_wake", AsyncMock())

        result = await wake_session_handler(
            "sess_01SOURCE",
            {"target_session_id": "sess_01TARGET", "prompt": "hi"},
        )

        # The depth read must be scoped to the trusted span carrier and must
        # NOT trust user-message metadata.
        assert "kind = 'span'" in captured["depth_sql"]
        assert "'role' = 'user'" not in captured["depth_sql"]
        # Trusted depth 4 (from the span) → new depth 5.
        assert result["wake_depth"] == 5

    async def test_rate_limit_query_filters_to_span_lineage(
        self, monkeypatch: Any, install_pool: Any
    ) -> None:
        """The per-pair rate-limit count must be scoped to ``kind='span'``
        lineage events, not forgeable user messages."""
        captured: dict[str, str] = {}

        pool = MagicMock()
        conn = MagicMock()

        conn.fetchrow = AsyncMock(
            side_effect=[
                {"account_id": "acc_test_stub", "status": "idle", "archived_at": None},
                {"depth": 0},
            ]
        )

        async def _fetchval(sql: str, *args: Any) -> int:
            captured["rate_sql"] = sql
            return 0

        conn.fetchval = AsyncMock(side_effect=_fetchval)
        conn.execute = AsyncMock()

        class _CM:
            async def __aenter__(self) -> MagicMock:
                return conn

            async def __aexit__(self, *a: Any) -> None:
                return None

        pool.acquire = lambda: _CM()
        install_pool(pool)
        monkeypatch.setattr("aios.db.queries.append_event", AsyncMock())
        monkeypatch.setattr("aios.tools.wake_session.defer_wake", AsyncMock())

        await wake_session_handler(
            "sess_01SOURCE",
            {"target_session_id": "sess_01TARGET", "prompt": "hi"},
        )

        assert "kind = 'span'" in captured["rate_sql"]
        assert "'role' = 'user'" not in captured["rate_sql"]

    async def test_appends_trusted_span_lineage_event(
        self, monkeypatch: Any, install_pool: Any
    ) -> None:
        """wake_session must append a system-owned ``kind='span'`` lineage
        event carrying the trusted wake_depth / wake_source_session_id, in
        addition to the user message."""
        install_pool(
            _make_pool(
                target_row={
                    "account_id": "acc_test_stub",
                    "status": "idle",
                    "archived_at": None,
                },
                depth=2,
                recent_wakes=0,
            )
        )
        mock_append = AsyncMock()
        monkeypatch.setattr("aios.db.queries.append_event", mock_append)
        monkeypatch.setattr("aios.tools.wake_session.defer_wake", AsyncMock())

        await wake_session_handler(
            "sess_01SOURCE",
            {"target_session_id": "sess_01TARGET", "prompt": "hi"},
        )

        # Two appends: the user message AND the trusted span lineage.
        kinds = [c.kwargs["kind"] for c in mock_append.call_args_list]
        assert "span" in kinds
        span_call = next(c for c in mock_append.call_args_list if c.kwargs["kind"] == "span")
        span_data = span_call.kwargs["data"]
        assert span_call.kwargs["session_id"] == "sess_01TARGET"
        assert span_call.kwargs["account_id"] == "acc_test_stub"
        assert span_data["event"] == "wake_lineage"
        assert span_data["wake_source_session_id"] == "sess_01SOURCE"
        assert span_data["wake_depth"] == 3
