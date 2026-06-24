"""Unit tests for the model-path tool dispatch: the ``validate_arguments`` helper and
the ``_classify_tool_error`` eviction policy.

``validate_arguments`` converts JSON Schema failures into a model-readable error string
that lists every problem at once. ``_classify_tool_error`` decides whether a handler
exception is an expected model-visible refusal (clean ``is_error`` result, no eviction)
or a genuine failure (also evicts the sandbox).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.errors import (
    AiosError,
    ConflictError,
    CryptoDecryptError,
    ForbiddenError,
    NotFoundError,
    RateLimitedError,
    ValidationError,
)
from aios.harness import tool_dispatch
from aios.harness.tool_dispatch import _classify_tool_error, _tool_lifecycle
from aios.services import sessions as sessions_service
from aios.tools.bash import BashArgumentError
from aios.tools.invoke import ToolBail, validate_arguments

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "channel_id": {"type": ["string", "null"]},
    },
    "required": ["channel_id"],
    "additionalProperties": False,
}


class TestValidateArguments:
    def test_valid_arguments_return_none(self) -> None:
        assert validate_arguments({"channel_id": "signal/a/b"}, _SCHEMA) is None
        assert validate_arguments({"channel_id": None}, _SCHEMA) is None

    def test_missing_required_key_is_reported(self) -> None:
        err = validate_arguments({}, _SCHEMA)
        assert err is not None
        assert "'channel_id' is a required property" in err

    def test_unexpected_property_is_reported(self) -> None:
        """The common weak-model failure mode: wrong param name.  The
        extra-property error must surface so the model can correct it.
        """
        err = validate_arguments({"target": "signal/a/b"}, _SCHEMA)
        assert err is not None
        assert "'target' was unexpected" in err

    def test_wrong_type_is_reported(self) -> None:
        err = validate_arguments({"channel_id": 42}, _SCHEMA)
        assert err is not None
        assert "42" in err  # what was sent
        # jsonschema's message varies; just assert we got SOMETHING useful
        assert "channel_id" in err or "type" in err.lower() or "42" in err

    def test_multi_error_accumulation(self) -> None:
        """Missing required + extra property → BOTH surface in a single
        error.  Model sees every issue at once; one retry fixes all.
        """
        err = validate_arguments({"target": "x"}, _SCHEMA)
        assert err is not None
        assert "'channel_id' is a required property" in err
        assert "'target' was unexpected" in err

    def test_passed_arguments_echoed_for_context(self) -> None:
        """The error includes the arguments the model sent so the model
        can see exactly what shape it produced vs. what was expected.
        """
        err = validate_arguments({"target": "oops"}, _SCHEMA)
        assert err is not None
        assert '"target": "oops"' in err

    def test_error_ends_with_hint_to_consult_schema(self) -> None:
        """Closing guidance points the model back at the tool's
        declared ``parameters`` — its authoritative reference."""
        err = validate_arguments({"target": "x"}, _SCHEMA)
        assert err is not None
        assert "parameters" in err.lower()

    def test_permissive_schema_allows_anything(self) -> None:
        """A schema with no constraints (no required, no additionalProps
        false) accepts arbitrary payloads — validation is opt-in per
        tool."""
        loose: dict[str, Any] = {"type": "object"}
        assert validate_arguments({"anything": "goes"}, loose) is None
        assert validate_arguments({}, loose) is None


class _InternalAiosError(AiosError):
    error_type = "internal_thing"
    status_code = 500


class TestClassifyToolError:
    @pytest.mark.parametrize(
        ("err", "evict", "message"),
        [
            # Expected, model-visible refusals — never evict the sandbox. The headline
            # fix: a 400 *ArgumentError (BashArgumentError) is a clean refusal now,
            # where it used to evict the container along with every other AiosError.
            (ToolBail("bad args"), False, "bad args"),
            (BashArgumentError("command must be non-empty"), False, "command must be non-empty"),
            (ForbiddenError("denied", detail={"x": 1}), False, 'denied ({"x": 1})'),
            (NotFoundError("no such workflow"), False, "no such workflow"),
            (ConflictError("stale version"), False, "stale version"),
            (RateLimitedError("slow down"), False, "slow down"),
            (ValidationError("bad field"), False, "bad field"),
            # Genuine failures — also evict the sandbox.
            (_InternalAiosError("internal boom"), True, "internal boom"),
            # A 5xx WITH detail still gets the json-suffixed message (the detail-format
            # branch is independent of the evict decision).
            (_InternalAiosError("boom", detail={"k": 2}), True, 'boom ({"k": 2})'),
            (CryptoDecryptError("decrypt failed"), True, "decrypt failed"),
            (ValueError("oops"), True, "ValueError: oops"),
            (RuntimeError("kaboom"), True, "RuntimeError: kaboom"),
        ],
    )
    def test_classification(self, err: BaseException, evict: bool, message: str) -> None:
        assert _classify_tool_error(err) == (evict, message)


class TestToolLifecycleEviction:
    """End-to-end: the classifier wired into ``_tool_lifecycle`` — a refusal appends a
    clean is_error result without evicting; a genuine failure also evicts."""

    @staticmethod
    async def _drive(monkeypatch: Any, err: BaseException) -> tuple[list[str], Any]:
        """Run a handler that raises ``err`` through the lifecycle; the DB writes are
        stubbed. Returns ``(evict_calls, append_result_mock)``."""
        monkeypatch.setattr(
            sessions_service,
            "append_event",
            AsyncMock(return_value=MagicMock(id="span_1")),
        )
        append_result = AsyncMock()
        monkeypatch.setattr(tool_dispatch, "_append_tool_result", append_result)
        monkeypatch.setattr(tool_dispatch, "_trigger_sweep", AsyncMock())

        evicted: list[str] = []
        call = {"id": "tc_1", "function": {"name": "demo", "arguments": "{}"}}
        async with _tool_lifecycle(
            MagicMock(),
            "ses_1",
            call,
            account_id="acc_1",
            log_prefix="tool",
            on_exception=evicted.append,
        ):
            raise err
        return evicted, append_result  # type: ignore[unreachable]

    async def test_client_error_refusal_does_not_evict(self, monkeypatch: Any) -> None:
        evicted, append_result = await self._drive(
            monkeypatch, ForbiddenError("denied", detail={"x": 1})
        )
        assert evicted == []  # a 4xx refusal must NOT evict the sandbox
        assert append_result.await_count == 1  # but still appends a model-visible result
        assert append_result.await_args.kwargs["error"] == 'denied ({"x": 1})'

    async def test_toolbail_does_not_evict(self, monkeypatch: Any) -> None:
        evicted, append_result = await self._drive(monkeypatch, ToolBail("bad json"))
        assert evicted == []
        assert append_result.await_args.kwargs["error"] == "bad json"

    async def test_server_error_evicts(self, monkeypatch: Any) -> None:
        evicted, append_result = await self._drive(monkeypatch, CryptoDecryptError("boom"))
        assert evicted == ["ses_1"]  # a 5xx is a genuine failure → evict
        assert append_result.await_count == 1

    async def test_unexpected_exception_evicts(self, monkeypatch: Any) -> None:
        evicted, append_result = await self._drive(monkeypatch, ValueError("oops"))
        assert evicted == ["ses_1"]
        assert append_result.await_args.kwargs["error"] == "ValueError: oops"


class TestCancelDuringStartSpan:
    """A (single) cancel landing on the ``tool_execute_start`` append — which happens inside
    ``__aenter__``, before the main try/finally (the pg-notify interrupt listener and a
    graceful drain both deliver one) — must still resolve the call in-task:
    a clean ``cancelled`` result + a tail sweep, with the body never running. Without the
    targeted handler the task would escape with no result and no sweep, stranding the call
    until the ≤30s ghost sweep and then misclassifying it as 'may have completed'. (A *second*
    cancel interrupting the cleanup awaits falls back to that ghost sweep — same exposure as
    the in-body cancel handler — so it is the documented backstop, not pinned here.)"""

    async def test_cancel_during_start_span_appends_cancelled_and_sweeps(
        self, monkeypatch: Any
    ) -> None:
        span_committed = asyncio.Event()
        release = asyncio.Event()

        async def fake_append_event(
            _pool: Any, _session_id: str, _kind: str, data: dict[str, Any], **_kw: Any
        ) -> Any:
            # Model the start-span write: signal it committed server-side, then hold the
            # client await open (suspended, cancellable) so the cancel lands right here.
            if data.get("event") == "tool_execute_start":
                span_committed.set()
                await release.wait()
            return MagicMock(id="span_1")

        monkeypatch.setattr(sessions_service, "append_event", fake_append_event)
        append_result: Any = AsyncMock()
        trigger_sweep = AsyncMock()
        monkeypatch.setattr(tool_dispatch, "_append_tool_result", append_result)
        monkeypatch.setattr(tool_dispatch, "_trigger_sweep", trigger_sweep)

        body_ran = asyncio.Event()
        call = {"id": "tc_1", "function": {"name": "demo", "arguments": "{}"}}

        async def driver() -> None:
            async with _tool_lifecycle(
                MagicMock(), "ses_1", call, account_id="acc_1", log_prefix="tool"
            ):
                body_ran.set()  # must NOT happen — the cancel struck during the start span

        task = asyncio.create_task(driver())
        await span_committed.wait()  # span "committed"; the lifecycle is suspended on it
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert not body_ran.is_set()  # the body provably never ran
        assert append_result.await_count == 1  # resolved in-task, not left to the sweep
        assert append_result.await_args.kwargs["error"] == "cancelled"
        assert trigger_sweep.await_count == 1  # tail sweep fired so the step wakes now


class TestParentChannelThreading:
    """The hot builtin dispatch path threads the parent assistant's focal
    stamp (``parent_focal_at_arrival``) into the tool-result append, so
    ``append_event`` never re-derives the channel under the row lock (#862).
    The error/cancel path omits it (default ``...`` sentinel → pre-tx lookup).
    """

    @staticmethod
    def _stub_lifecycle_deps(monkeypatch: Any) -> None:
        monkeypatch.setattr(
            sessions_service,
            "append_event",
            AsyncMock(return_value=MagicMock(id="span_1")),
        )
        monkeypatch.setattr(tool_dispatch, "_trigger_sweep", AsyncMock())

    async def test_execute_tool_async_threads_parent_channel(self, monkeypatch: Any) -> None:
        from aios.tools.registry import ToolResult

        self._stub_lifecycle_deps(monkeypatch)
        monkeypatch.setattr(
            tool_dispatch,
            "invoke_builtin",
            AsyncMock(return_value=ToolResult(content="ok")),
        )
        append_event_ev: Any = AsyncMock()
        monkeypatch.setattr(tool_dispatch, "_append_tool_result_event", append_event_ev)

        call = {"id": "tc_1", "function": {"name": "demo", "arguments": "{}"}}
        await tool_dispatch._execute_tool_async(
            MagicMock(),
            "ses_1",
            call,
            account_id="acc_1",
            parent_focal_at_arrival="tg:42",
        )
        assert append_event_ev.await_count == 1
        assert append_event_ev.await_args.kwargs["tool_parent_channel"] == "tg:42"

    async def test_execute_tool_async_threads_none_stamp(self, monkeypatch: Any) -> None:
        from aios.tools.registry import ToolResult

        self._stub_lifecycle_deps(monkeypatch)
        monkeypatch.setattr(
            tool_dispatch,
            "invoke_builtin",
            AsyncMock(return_value=ToolResult(content="ok")),
        )
        append_event_ev: Any = AsyncMock()
        monkeypatch.setattr(tool_dispatch, "_append_tool_result_event", append_event_ev)

        call = {"id": "tc_1", "function": {"name": "demo", "arguments": "{}"}}
        # A non-connector parent stamps focal None — the hot path must still
        # PASS it (None), not fall through to the lookup default.
        await tool_dispatch._execute_tool_async(
            MagicMock(),
            "ses_1",
            call,
            account_id="acc_1",
            parent_focal_at_arrival=None,
        )
        assert append_event_ev.await_args.kwargs["tool_parent_channel"] is None

    async def test_error_path_omits_parent_channel(self, monkeypatch: Any) -> None:
        """A handler raising routes through ``_append_tool_result`` (the
        error/cancel append), which does NOT pass ``tool_parent_channel`` —
        so ``_append_tool_result_event`` keeps the default ``...`` and
        ``append_event`` falls back to the pre-tx parent lookup."""
        self._stub_lifecycle_deps(monkeypatch)
        monkeypatch.setattr(
            tool_dispatch,
            "invoke_builtin",
            AsyncMock(side_effect=ValueError("boom")),
        )
        append_event_ev: Any = AsyncMock()
        monkeypatch.setattr(tool_dispatch, "_append_tool_result_event", append_event_ev)
        monkeypatch.setattr(tool_dispatch, "_evict_session_container", lambda _s: None)

        call = {"id": "tc_1", "function": {"name": "demo", "arguments": "{}"}}
        await tool_dispatch._execute_tool_async(
            MagicMock(),
            "ses_1",
            call,
            account_id="acc_1",
            parent_focal_at_arrival="tg:42",
        )
        assert append_event_ev.await_count == 1
        assert "tool_parent_channel" not in append_event_ev.await_args.kwargs

    async def test_append_tool_result_event_forwards_precomputed_channel(
        self, monkeypatch: Any
    ) -> None:
        """Hot path (#991): ``_append_tool_result_event`` resolves the precompute
        OUTSIDE the lock and forwards it as ``precomputed=`` to
        ``queries.append_event``.  A supplied ``tool_parent_channel`` flows
        through as ``resolved_tool_channel`` with NO DB lookup."""
        from aios.db import queries

        captured: dict[str, Any] = {}

        async def _fake_append_event(conn: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(queries, "append_event", _fake_append_event)
        monkeypatch.setattr(
            queries,
            "find_tool_result_event",
            AsyncMock(return_value=None),
        )

        conn = MagicMock()
        conn.execute = AsyncMock()
        tx = MagicMock()
        tx.__aenter__ = AsyncMock(return_value=None)
        tx.__aexit__ = AsyncMock(return_value=None)
        conn.transaction = MagicMock(return_value=tx)
        acquire = MagicMock()
        acquire.__aenter__ = AsyncMock(return_value=conn)
        acquire.__aexit__ = AsyncMock(return_value=None)
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=acquire)

        await tool_dispatch._append_tool_result_event(
            pool,
            "ses_1",
            "tc_1",
            {"role": "tool", "tool_call_id": "tc_1", "content": "ok"},
            account_id="acc_1",
            tool_parent_channel="tg:42",
        )
        # The hot path passes ``precomputed=`` (not ``tool_parent_channel=``);
        # its resolved channel echoes the supplied stamp with no DB round-trip.
        assert "tool_parent_channel" not in captured
        assert captured["precomputed"].resolved_tool_channel == "tg:42"


def _mock_pool_conn() -> tuple[Any, Any]:
    """Build a mock asyncpg pool whose ``acquire()`` yields a conn with a
    nesting-safe ``transaction()`` (the inner ``append_event`` SAVEPOINT and
    the outer ``_append_tool_result_event`` transaction both no-op here)."""
    conn = MagicMock()
    conn.execute = AsyncMock()
    tx = MagicMock()
    tx.__aenter__ = AsyncMock(return_value=None)
    tx.__aexit__ = AsyncMock(return_value=None)
    conn.transaction = MagicMock(return_value=tx)
    acquire = MagicMock()
    acquire.__aenter__ = AsyncMock(return_value=conn)
    acquire.__aexit__ = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=acquire)
    return pool, conn


class TestToolResultUniqueFloor:
    """The partial UNIQUE index ``events_tool_result_idx`` (#1082) is the
    structural floor under the manual dedup: a racing committed appender that
    slips past the read-check makes ``append_event`` raise
    ``UniqueViolationError``; ``_append_tool_result_event`` must re-read, treat
    it as a dedup-skip, and apply the ``open_tool_call_count`` compensation —
    NOT propagate the raw DB error."""

    async def test_unique_violation_treated_as_dedup_skip(self, monkeypatch: Any) -> None:
        import asyncpg

        from aios.db import queries

        winner = MagicMock()
        winner.data = {"is_error": False}
        # The read-check sees nothing (the racing row committed AFTER it),
        # then the re-read after the violation finds the winning row.
        find = AsyncMock(side_effect=[None, winner])
        monkeypatch.setattr(queries, "find_tool_result_event", find)
        monkeypatch.setattr(
            queries,
            "append_event",
            AsyncMock(side_effect=asyncpg.UniqueViolationError("dup")),
        )
        # The cold path (default ``...``) precomputes off a brief pool.acquire()
        # BEFORE the lock — stub it so the mock conn needs no real query support.
        monkeypatch.setattr(
            queries,
            "precompute_event_append",
            AsyncMock(return_value=queries._PrecomputedAppend(0, None)),
        )
        decrement = AsyncMock()
        monkeypatch.setattr(queries, "decrement_open_tool_call_count", decrement)

        pool, _conn = _mock_pool_conn()

        # Must NOT raise — the late append no-ops.
        await tool_dispatch._append_tool_result_event(
            pool,
            "ses_1",
            "tc_1",
            {"role": "tool", "tool_call_id": "tc_1", "content": "ok"},
            account_id="acc_1",
        )
        assert find.await_count == 2  # read-check + post-violation re-read
        assert decrement.await_count == 1  # id-blind +1 compensation applied once

    async def test_read_check_hit_skips_append(self, monkeypatch: Any) -> None:
        """Fast path: when the read-check already finds the winning row, the
        append is never attempted and the compensation runs once."""
        from aios.db import queries

        winner = MagicMock()
        winner.data = {"is_error": False}
        monkeypatch.setattr(queries, "find_tool_result_event", AsyncMock(return_value=winner))
        append = AsyncMock()
        monkeypatch.setattr(queries, "append_event", append)
        monkeypatch.setattr(
            queries,
            "precompute_event_append",
            AsyncMock(return_value=queries._PrecomputedAppend(0, None)),
        )
        decrement = AsyncMock()
        monkeypatch.setattr(queries, "decrement_open_tool_call_count", decrement)

        pool, _conn = _mock_pool_conn()
        await tool_dispatch._append_tool_result_event(
            pool,
            "ses_1",
            "tc_1",
            {"role": "tool", "tool_call_id": "tc_1", "content": "ok"},
            account_id="acc_1",
        )
        assert append.await_count == 0
        assert decrement.await_count == 1


class TestAppendToolResultUniqueFloor:
    """``services.sessions.append_tool_result`` (#1082): a ``UniqueViolation``
    from the racing-committed-row case re-reads + re-classifies — collapsing
    into idempotent-retry (return existing) or deny-after-success
    (``ConflictError``) exactly as the read-check path does."""

    @staticmethod
    def _patch_common(monkeypatch: Any) -> None:
        from aios.db import queries

        monkeypatch.setattr(queries, "lock_active_session_for_update", AsyncMock(return_value=None))
        monkeypatch.setattr(
            queries, "lookup_tool_name_by_call_id", AsyncMock(return_value=("demo", None))
        )
        # No spill capping in unit tests: return the content unchanged with
        # no attachment record (the within-cap shape).
        from aios.sandbox.tool_result_spill import CappedToolResult

        monkeypatch.setattr(
            "aios.sandbox.tool_result_spill.cap_tool_result_content",
            AsyncMock(side_effect=lambda *a, **k: CappedToolResult(content=a[2], attachment=None)),
        )

    @staticmethod
    def _mock_conn() -> Any:
        conn = MagicMock()
        tx = MagicMock()
        tx.__aenter__ = AsyncMock(return_value=None)
        tx.__aexit__ = AsyncMock(return_value=None)
        conn.transaction = MagicMock(return_value=tx)
        return conn

    async def test_unique_violation_matching_intent_returns_existing(
        self, monkeypatch: Any
    ) -> None:
        import asyncpg

        from aios.db import queries
        from aios.models.events import Event

        self._patch_common(monkeypatch)
        existing = MagicMock(spec=Event)
        existing.data = {"is_error": False}
        monkeypatch.setattr(
            queries, "find_tool_result_event", AsyncMock(side_effect=[None, existing])
        )
        monkeypatch.setattr(
            queries, "append_event", AsyncMock(side_effect=asyncpg.UniqueViolationError("dup"))
        )
        decrement = AsyncMock()
        monkeypatch.setattr(queries, "decrement_open_tool_call_count", decrement)

        result = await sessions_service.append_tool_result(
            self._mock_conn(),
            account_id="acc_1",
            session_id="ses_1",
            tool_call_id="tc_1",
            content="ok",
            is_error=False,
        )
        assert result is existing
        assert decrement.await_count == 1

    async def test_unique_violation_intent_mismatch_raises_conflict(self, monkeypatch: Any) -> None:
        import asyncpg

        from aios.db import queries
        from aios.models.events import Event

        self._patch_common(monkeypatch)
        existing = MagicMock(spec=Event)
        existing.data = {"is_error": False}  # prior SUCCESS already committed
        monkeypatch.setattr(
            queries, "find_tool_result_event", AsyncMock(side_effect=[None, existing])
        )
        monkeypatch.setattr(
            queries, "append_event", AsyncMock(side_effect=asyncpg.UniqueViolationError("dup"))
        )
        monkeypatch.setattr(queries, "decrement_open_tool_call_count", AsyncMock())

        # Late DENY (is_error=True) against a committed SUCCESS — too-late.
        with pytest.raises(ConflictError):
            await sessions_service.append_tool_result(
                self._mock_conn(),
                account_id="acc_1",
                session_id="ses_1",
                tool_call_id="tc_1",
                content="denied",
                is_error=True,
            )
