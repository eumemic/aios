"""Unit tests for the MCP session pool.

All MCP SDK and httpx interactions are mocked. No network calls.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.harness import runtime
from aios.mcp.pool import McpSessionPool

# ── helpers ────────────────────────────────────────────────────────────────


def _make_mock_session() -> AsyncMock:
    """Build an AsyncMock that looks like a ``ClientSession``."""
    session = AsyncMock()
    session.initialize = AsyncMock(return_value=MagicMock())
    session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
    session.call_tool = AsyncMock(return_value=MagicMock(content=[], isError=False))
    return session


def _transport_mock() -> MagicMock:
    t = MagicMock()
    t.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
    t.__aexit__ = AsyncMock(return_value=False)
    return t


def _session_ctx(session: AsyncMock) -> MagicMock:
    cls = MagicMock()
    cls.return_value.__aenter__ = AsyncMock(return_value=session)
    cls.return_value.__aexit__ = AsyncMock(return_value=False)
    return cls


def _session_ctx_seq(sessions: list[AsyncMock]) -> MagicMock:
    it: Iterator[AsyncMock] = iter(sessions)

    async def _enter(*_args: Any, **_kwargs: Any) -> AsyncMock:
        return next(it)

    cls = MagicMock()
    cls.return_value.__aenter__ = AsyncMock(side_effect=_enter)
    cls.return_value.__aexit__ = AsyncMock(return_value=False)
    return cls


# ── McpSessionPool ─────────────────────────────────────────────────────────


class TestMcpSessionPool:
    async def test_pool_hit_reuses_session(self) -> None:
        """Two calls with same key → only one ``initialize()`` happens."""
        session = _make_mock_session()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            pool = McpSessionPool()
            headers = {"Authorization": "Bearer tok"}
            s1, _ = await pool.get_or_connect("https://m.example/", "v", headers)
            s2, _ = await pool.get_or_connect("https://m.example/", "v", headers)

        assert s1 is s2
        session.initialize.assert_awaited_once()

    async def test_pool_miss_different_vault_ids(self) -> None:
        """Different vault_ids on the same URL → two separate sessions
        (the multi-tenant safety surface: per-account vaults yield
        per-account pool entries even when sharing an MCP URL)."""
        s_a, s_b = _make_mock_session(), _make_mock_session()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx_seq([s_a, s_b])),
        ):
            pool = McpSessionPool()
            r1, _ = await pool.get_or_connect(
                "https://m.example/", "vault_a", {"Authorization": "Bearer t1"}
            )
            r2, _ = await pool.get_or_connect(
                "https://m.example/", "vault_b", {"Authorization": "Bearer t2"}
            )

        assert r1 is not r2
        assert s_a.initialize.await_count == 1
        assert s_b.initialize.await_count == 1

    async def test_evict_causes_reconnect(self) -> None:
        """After eviction, next ``get_or_connect`` opens a new session."""
        s_a, s_b = _make_mock_session(), _make_mock_session()
        headers = {"Authorization": "Bearer tok"}
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx_seq([s_a, s_b])),
        ):
            pool = McpSessionPool()
            r1, _ = await pool.get_or_connect("https://m.example/", "v", headers)
            pool.evict("https://m.example/", "v")
            r2, _ = await pool.get_or_connect("https://m.example/", "v", headers)

        assert r1 is not r2

    async def test_evict_closes_evicted_entry(self) -> None:
        """``evict`` must reclaim the entry, not just drop the reference.

        The idle reaper's docstring spells out the leak shape: dropping a
        reference without going through ``_Entry.close()`` strands the
        owner task, the httpx client, and the SSE stream — exactly what
        the reaper exists to prevent. The reaper iterates ``_entries``,
        so an evicted entry (popped from the map) is invisible to it
        and the leak survives until process exit. Every transient MCP
        failure produces one such evict, so over a long-running worker
        the leak is unbounded.
        """
        url = "https://m.example/"
        headers = {"Authorization": "Bearer tok"}
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(_make_mock_session())),
        ):
            pool = McpSessionPool()
            await pool.get_or_connect(url, "v", headers)
            entry = pool._entries[(url, "v")]

            closed = False
            orig_close = entry.close

            async def _tracked() -> None:
                nonlocal closed
                closed = True
                await orig_close()

            entry.close = _tracked  # type: ignore[method-assign]

            pool.evict(url, "v")
            # Allow any background close task scheduled by evict to run.
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        assert closed, (
            "evict must close the entry through _Entry.close() to release "
            "the owner task + httpx client + SSE stream; a bare pop strands "
            "them for the worker's lifetime"
        )
        assert entry._owner_task.done(), (
            "the owner task must exit after evict; while parked on "
            "shutdown.wait() it holds the httpx client and SSE stream open"
        )

    async def test_thundering_herd_single_initialize(self) -> None:
        """N concurrent ``get_or_connect`` calls → exactly one ``initialize()``."""
        session = _make_mock_session()

        init_count = 0

        async def slow_init() -> MagicMock:
            nonlocal init_count
            init_count += 1
            # Yield to let other concurrent callers pile up on the lock.
            await asyncio.sleep(0)
            return MagicMock()

        session.initialize = AsyncMock(side_effect=slow_init)

        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            pool = McpSessionPool()
            headers = {"Authorization": "Bearer tok"}
            results = await asyncio.gather(
                *[pool.get_or_connect("https://m.example/", "v", headers) for _ in range(8)]
            )

        assert init_count == 1
        first = results[0][0]
        for session_got, _ in results:
            assert session_got is first

    async def test_close_all_closes_entries(self) -> None:
        """``close_all`` tears down every pooled entry and clears the map."""
        session = _make_mock_session()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            pool = McpSessionPool()
            await pool.get_or_connect("https://a.example/", "vault_a", {"Authorization": "A"})
            await pool.get_or_connect("https://b.example/", "vault_b", {"Authorization": "B"})

            close_calls: list[str] = []
            for key, entry in pool._entries.items():
                url = key[0]
                orig = entry.close

                async def _tracked(u: str = url, o: Any = orig) -> None:
                    close_calls.append(u)
                    await o()

                entry.close = _tracked  # type: ignore[method-assign]

            await pool.close_all()

        assert len(close_calls) == 2
        assert len(pool._entries) == 0

    async def test_close_all_empty_pool(self) -> None:
        """``close_all`` on an empty pool is a no-op."""
        pool = McpSessionPool()
        await pool.close_all()
        assert len(pool._entries) == 0

    async def test_idle_reaper_closes_cold_entry(self) -> None:
        """A cold pool entry must be reclaimed via ``_Entry.close()``.

        Post-#459 re-key, the reaper guards the cold-entry vector: a
        ``(url, vault_id)`` whose tenant never returns. A bare ``pop``
        would leave the owner task + httpx + SSE stream dangling —
        exactly the leak this defends against.
        """
        s_cold, s_warm = _make_mock_session(), _make_mock_session()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx_seq([s_cold, s_warm])),
        ):
            pool = McpSessionPool()
            url = "https://m.example/"
            await pool.get_or_connect(url, "vault_cold", {"Authorization": "Bearer A"})
            await pool.get_or_connect(url, "vault_warm", {"Authorization": "Bearer B"})
            assert len(pool._entries) == 2

            # Locate each entry by the session it wraps (keying-agnostic
            # so this test stays a pure reaper-mechanism test).
            key_cold = next(k for k, e in pool._entries.items() if e.session is s_cold)
            key_warm = next(k for k, e in pool._entries.items() if e.session is s_warm)

            closed: list[tuple[str, str | None]] = []
            for key, entry in pool._entries.items():
                orig = entry.close

                async def _tracked(k: tuple[str, str | None] = key, o: Any = orig) -> None:
                    closed.append(k)
                    await o()

                entry.close = _tracked  # type: ignore[method-assign]

            # Cold tenant idle since t=1000; warm tenant freshly used at t=9000.
            pool._entries[key_cold].last_used = 1000.0
            pool._entries[key_warm].last_used = 9000.0

            await pool._reap_idle_once(idle_timeout=300.0, now=9100.0)

        assert closed == [key_cold], (
            f"idle reaper must close exactly the cold entry via the clean "
            f"_Entry.close() path; closed={closed!r}"
        )
        assert key_cold not in pool._entries
        assert key_warm in pool._entries

    async def test_idle_reaper_skips_entry_freshened_during_close_of_another(self) -> None:
        """The reaper must not close an entry whose ``last_used`` was
        freshened by a concurrent ``get_or_connect`` between the scan
        and the actual close.

        Scenario: two pool entries (``vault_a``, ``vault_b``) both land
        in the ``stale`` snapshot. While the reaper is awaiting
        ``entry_a.close()``, a tool task warm-hits
        ``get_or_connect(url, vault_b, ...)``, which bumps
        ``entry_b.last_used`` to ``time.monotonic()``. The reaper must
        NOT proceed to close ``entry_b`` — the entry is no longer idle.

        Pre-fix, the reaper acted on the stale ``stale`` snapshot
        without re-checking ``last_used`` under the per-key lock, so it
        would close an entry actively held by a caller. Same failure
        class as the SandboxRegistry reaper TOCTOU fixed in #654 — the
        warm path (pool.py:193-196) takes no lock.
        """
        park_a = asyncio.Event()
        continue_a = asyncio.Event()
        closed: list[tuple[str, str | None]] = []

        s_a, s_b = _make_mock_session(), _make_mock_session()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx_seq([s_a, s_b])),
        ):
            pool = McpSessionPool()
            url = "https://m.example/"
            await pool.get_or_connect(url, "vault_a", {"Authorization": "Bearer A"})
            await pool.get_or_connect(url, "vault_b", {"Authorization": "Bearer B"})
            assert len(pool._entries) == 2

            key_a = next(k for k, e in pool._entries.items() if e.session is s_a)
            key_b = next(k for k, e in pool._entries.items() if e.session is s_b)
            entry_a = pool._entries[key_a]
            entry_b = pool._entries[key_b]
            orig_close_a = entry_a.close
            orig_close_b = entry_b.close

            async def _close_a() -> None:
                closed.append(key_a)
                park_a.set()
                await continue_a.wait()
                await orig_close_a()

            async def _close_b() -> None:
                closed.append(key_b)
                await orig_close_b()

            entry_a.close = _close_a  # type: ignore[method-assign]
            entry_b.close = _close_b  # type: ignore[method-assign]

            # Both deeply idle so they both land in ``stale`` on the scan.
            entry_a.last_used = 1000.0
            entry_b.last_used = 1000.0

            # Insertion order pins key_a as the first iteration → reaper
            # parks inside _close_a, yielding control to the test.
            reap_task = asyncio.create_task(pool._reap_idle_once(idle_timeout=300.0, now=9100.0))
            try:
                await asyncio.wait_for(park_a.wait(), timeout=1.0)
                # Reaper is parked inside entry_a.close(). Warm-hit
                # vault_b — last_used is bumped to time.monotonic(), so
                # the re-check at entry_b's iteration sees a fresh value.
                returned_session, _ = await pool.get_or_connect(
                    url, "vault_b", {"Authorization": "Bearer B"}
                )
                assert returned_session is s_b
                continue_a.set()
                await asyncio.wait_for(reap_task, timeout=1.0)
            finally:
                if not reap_task.done():
                    reap_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await reap_task

        assert key_b not in closed, (
            f"reaper closed vault_b's session despite a concurrent "
            f"get_or_connect freshening entry_b.last_used. The reaper "
            f"must re-check idleness under the per-key lock before "
            f"calling close(). closed={closed!r}"
        )
        assert key_b in pool._entries, "vault_b's entry must remain in the pool"

    async def test_oauth_refresh_does_not_orphan_entry(self) -> None:
        """A token rotation on the same vault must reuse the pool key.

        Pre-#459 the pool keyed on ``(url, sha256(headers))``, so a
        rotated bearer produced a second key while the predecessor
        ``_Entry`` lingered. Keyed on ``(url, vault_id)``, rotation hits
        the same key and ``len(_entries)`` stays put.
        """
        s_old, s_new = _make_mock_session(), _make_mock_session()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx_seq([s_old, s_new])),
        ):
            pool = McpSessionPool()
            url = "https://m.example/"
            # Same vault (identity), two bearer values — the exact shape
            # an OAuth refresh produces at the inbound boundary.
            await pool.get_or_connect(url, "vault_a", {"Authorization": "Bearer old"})
            await pool.get_or_connect(url, "vault_a", {"Authorization": "Bearer new"})

        assert len(pool._entries) == 1, (
            f"a token refresh of the same vault must NOT orphan an entry; "
            f"got len(_entries)={len(pool._entries)} (pre-fix the rotated "
            f"bearer produced a second key while the predecessor lingered)"
        )

    async def test_cross_tenant_refresh_does_not_disturb_other_tenant(self) -> None:
        """Multi-tenant invariant the #459 re-key is built on.

        ``vault_id`` is account-filtered at the query layer
        (``resolve_session_credential(..., account_id=...)``), so two
        tenants sharing an MCP URL hold independent entries and one's
        refresh reuses its OWN entry without disturbing the other.
        Pre-fix the rotation would open a third entry under a new
        headers_hash (orphaning A's first entry); post-fix it reuses
        A's stable ``(url, vault_id)`` key.
        """
        # Exactly 2 sessions: A's, B's. Pre-fix A's rotation pulls a
        # third → StopIteration; post-fix it reuses A's cached entry.
        s_a, s_b = _make_mock_session(), _make_mock_session()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx_seq([s_a, s_b])),
        ):
            pool = McpSessionPool()
            url = "https://shared.example/"
            sess_a1, _ = await pool.get_or_connect(url, "vault_A", {"Authorization": "Bearer tokA"})
            sess_b1, _ = await pool.get_or_connect(url, "vault_B", {"Authorization": "Bearer tokB"})
            # A rotates: must reuse A's cached entry (no new open).
            sess_a2, _ = await pool.get_or_connect(
                url, "vault_A", {"Authorization": "Bearer tokA_v2"}
            )

        assert len(pool._entries) == 2, (
            f"A's refresh must reuse A's entry, not orphan a third; "
            f"got len(_entries)={len(pool._entries)}"
        )
        assert sess_a2 is sess_a1, "A's session must be reused across rotation"
        assert sess_b1 is not sess_a1, "tenants must hold distinct entries"

    async def test_contexts_exit_in_same_task_they_entered(self) -> None:
        """Cross-task close — regression for #425.

        Pre-fix, the pool held an ``AsyncExitStack`` on whatever task
        first opened the entry. ``close_all`` called ``stack.aclose()``
        from a different task and anyio's ``streamable_http_client``
        raised ``RuntimeError: Attempted to exit cancel scope in a
        different task than it was entered in``. The exception was
        swallowed by ``close_all``'s try/except but logged a warning.

        ``close_all`` still swallows post-fix (defensively, for unrelated
        teardown failures), so this test instead checks the more
        specific invariant: the transport's ``__aenter__`` and
        ``__aexit__`` run in the SAME task. With the owner-task model
        that's guaranteed; with the legacy stack-on-_Entry model it
        wasn't.
        """
        entered_in: dict[str, asyncio.Task[Any] | None] = {"open": None, "close": None}

        def _task_recording_transport() -> MagicMock:
            async def _enter(*_args: Any, **_kwargs: Any) -> Any:
                entered_in["open"] = asyncio.current_task()
                return (MagicMock(), MagicMock(), MagicMock())

            async def _exit(*_args: Any, **_kwargs: Any) -> bool:
                entered_in["close"] = asyncio.current_task()
                return False

            t = MagicMock()
            t.__aenter__ = AsyncMock(side_effect=_enter)
            t.__aexit__ = AsyncMock(side_effect=_exit)
            return t

        session = _make_mock_session()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_task_recording_transport()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            pool = McpSessionPool()

            async def opener() -> None:
                await pool.get_or_connect("https://m.example/", "v", {"Authorization": "tok"})

            await asyncio.create_task(opener())

            async def closer() -> None:
                await pool.close_all()

            await asyncio.create_task(closer())

        assert entered_in["open"] is not None
        assert entered_in["close"] is not None
        # The invariant: contexts exit in the same task that entered them.
        # Pre-fix this would have failed because __aexit__ ran in the
        # closer task while __aenter__ ran in the opener task.
        assert entered_in["open"] is entered_in["close"]


# ── Pool integration via discover_mcp_tools / call_mcp_tool ────────────────


@pytest.fixture
def restore_runtime_pool() -> Iterator[None]:
    """Save + restore ``runtime.mcp_session_pool`` across each test."""
    saved = runtime.mcp_session_pool
    try:
        yield
    finally:
        runtime.mcp_session_pool = saved


class TestDiscoverMcpToolsWithPool:
    async def test_uses_pool_when_set(self, restore_runtime_pool: None) -> None:
        """Pool path: ``initialize`` called once even across two discovers."""
        from aios.mcp.client import discover_mcp_tools

        session = _make_mock_session()
        runtime.mcp_session_pool = McpSessionPool()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            await discover_mcp_tools("https://m.example/", "v", {}, "srv")
            await discover_mcp_tools("https://m.example/", "v", {}, "srv")

        session.initialize.assert_awaited_once()

    async def test_evicts_and_retries_on_list_tools_failure(
        self, restore_runtime_pool: None
    ) -> None:
        """First ``list_tools`` raises → pool evicts, second succeeds."""
        from aios.mcp.client import discover_mcp_tools

        s_a, s_b = _make_mock_session(), _make_mock_session()
        s_a.list_tools = AsyncMock(side_effect=RuntimeError("broken pipe"))
        s_b.list_tools = AsyncMock(return_value=MagicMock(tools=[]))

        runtime.mcp_session_pool = McpSessionPool()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx_seq([s_a, s_b])),
        ):
            tools, _ = await discover_mcp_tools("https://m.example/", "v", {}, "srv")

        assert tools == []
        assert s_a.initialize.await_count == 1
        assert s_b.initialize.await_count == 1

    async def test_falls_back_when_pool_is_none(self, restore_runtime_pool: None) -> None:
        """When the pool is None, fresh-connection path is used."""
        from aios.mcp.client import discover_mcp_tools

        runtime.mcp_session_pool = None
        session = _make_mock_session()
        with (
            patch("aios.mcp.client.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.client.ClientSession", _session_ctx(session)),
        ):
            tools, _ = await discover_mcp_tools("https://m.example/", "v", {}, "srv")

        assert tools == []
        session.initialize.assert_awaited_once()


class TestCallMcpToolWithPool:
    async def test_uses_pool_when_set(self, restore_runtime_pool: None) -> None:
        """Pool path: ``initialize`` called once even across two tool calls."""
        from aios.mcp.client import call_mcp_tool

        session = _make_mock_session()
        session.call_tool = AsyncMock(return_value=MagicMock(content=[], isError=False))

        runtime.mcp_session_pool = McpSessionPool()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            await call_mcp_tool("https://m.example/", "v", {}, "do_thing", {})
            await call_mcp_tool("https://m.example/", "v", {}, "do_thing", {})

        session.initialize.assert_awaited_once()

    async def test_does_not_retry_on_call_tool_failure(self, restore_runtime_pool: None) -> None:
        """A non-timeout transport failure (broken pipe, TCP reset,
        HTTP/2 GOAWAY) may have landed after the server already
        processed the request — retrying would duplicate the side
        effect, just like the timeout case.  The wrapper evicts and
        surfaces the error; the model decides whether to retry."""
        from aios.mcp.client import call_mcp_tool

        s_a = _make_mock_session()
        s_a.call_tool = AsyncMock(side_effect=RuntimeError("broken"))

        runtime.mcp_session_pool = McpSessionPool()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(s_a)),
        ):
            result = await call_mcp_tool("https://m.example/", "v", {}, "do_thing", {})

        assert result.get("code") == "transport_error"
        assert s_a.call_tool.await_count == 1

    async def test_does_not_retry_call_tool_on_timeout(
        self, restore_runtime_pool: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A timeout in ``call_tool`` may mean the request reached the server
        and was processed; retrying would duplicate the side effect. For
        connector-provided tools like ``signal_send``/``telegram_send``
        that's a duplicate user-visible message. The wrapper must NOT
        retry on ``asyncio.TimeoutError`` — evict + surface the error
        so the model decides whether to retry."""
        from aios.mcp import client as mcp_client
        from aios.mcp.client import call_mcp_tool

        # Shrink the timeout so the test runs fast.
        monkeypatch.setattr(mcp_client, "_TOOL_CALL_TIMEOUT_S", 0.05)

        s_a = _make_mock_session()

        async def _hang(*_args: object, **_kwargs: object) -> object:
            await asyncio.sleep(60)
            return None

        s_a.call_tool = AsyncMock(side_effect=_hang)

        runtime.mcp_session_pool = McpSessionPool()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(s_a)),
        ):
            result = await call_mcp_tool("https://m.example/", "v", {}, "do_thing", {})

        # The wrapper returns a transport-error envelope on timeout.
        assert result.get("code") == "transport_error"
        # And crucially, ``call_tool`` was invoked EXACTLY ONCE — no retry.
        assert s_a.call_tool.await_count == 1
