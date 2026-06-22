"""Unit tests for the MCP session pool (per-call checkout model).

All MCP SDK and httpx interactions are mocked. No network calls.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from aios.harness import runtime
from aios.mcp.client import _headers_key
from aios.mcp.pool import McpSessionPool

URL = "https://m.example/"

# Canonical headers-key components for the pool's 3-tuple key. The pool now
# keys on (url, vault_id, headers_key); these are the keys the helper produces
# for the no-headers case and a representative static-headers case.
EMPTY_KEY = _headers_key(None)
K1 = _headers_key({"X-MCP-Toolsets": "issues"})

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


def _live_count(pool: McpSessionPool, key: tuple[str, str | None, str]) -> int:
    return len(pool._idle.get(key, [])) + len(pool._in_use.get(key, set()))


# ── McpSessionPool (checkout model) ─────────────────────────────────────────


class TestMcpSessionPool:
    async def test_warm_reuse_after_release(self) -> None:
        """acquire → release → acquire returns the SAME session (idle reuse);
        ``initialize`` runs only once."""
        session = _make_mock_session()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            pool = McpSessionPool()
            headers = {"Authorization": "Bearer tok"}
            e1 = await pool.acquire(URL, "v", EMPTY_KEY, headers)
            await pool.release(URL, "v", EMPTY_KEY, e1)
            # release moves the entry to idle and cleans up the empty in-use set.
            assert (URL, "v", EMPTY_KEY) not in pool._in_use
            assert pool._idle[(URL, "v", EMPTY_KEY)] == [e1]
            e2 = await pool.acquire(URL, "v", EMPTY_KEY, headers)

        assert e1 is e2
        assert e1.session is e2.session
        session.initialize.assert_awaited_once()

    async def test_distinct_headers_key_distinct_entries(self) -> None:
        """Same (url, vault_id) but different ``headers_key`` → DISTINCT
        entries/sessions. The headers_key is part of the pool key so two
        agents pointing at the same server with different static headers
        (e.g. different ``X-MCP-Toolsets`` selectors) never share a
        session."""
        s_a, s_b = _make_mock_session(), _make_mock_session()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx_seq([s_a, s_b])),
        ):
            pool = McpSessionPool()
            e1 = await pool.acquire(URL, "v", EMPTY_KEY, {})
            # e1 stays in use → the second acquire can't alias it even if the
            # keys collided; distinct headers_key forces a wholly separate key.
            e2 = await pool.acquire(URL, "v", K1, {"X-MCP-Toolsets": "issues"})

        assert e1 is not e2
        assert e1.session is not e2.session
        assert (URL, "v", EMPTY_KEY) in pool._in_use
        assert (URL, "v", K1) in pool._in_use

    async def test_same_no_headers_none_and_empty_reuse_entry(self) -> None:
        """``_headers_key(None)`` and ``_headers_key({})`` canonicalize to the
        same key, so a no-headers acquire reuses an entry released under the
        other — the no-regression guarantee for the today's-default case."""
        session = _make_mock_session()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            pool = McpSessionPool()
            e1 = await pool.acquire(URL, "v", _headers_key(None), {})
            await pool.release(URL, "v", _headers_key(None), e1)
            e2 = await pool.acquire(URL, "v", _headers_key({}), {})

        assert e2 is e1
        assert _live_count(pool, (URL, "v", EMPTY_KEY)) == 1
        session.initialize.assert_awaited_once()

    async def test_different_vault_ids_get_distinct_sessions(self) -> None:
        """Different vault_ids on the same URL → separate sessions (the
        multi-tenant safety surface)."""
        s_a, s_b = _make_mock_session(), _make_mock_session()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx_seq([s_a, s_b])),
        ):
            pool = McpSessionPool()
            e1 = await pool.acquire(URL, "vault_a", EMPTY_KEY, {"Authorization": "Bearer t1"})
            e2 = await pool.acquire(URL, "vault_b", EMPTY_KEY, {"Authorization": "Bearer t2"})

        assert e1.session is not e2.session
        assert s_a.initialize.await_count == 1
        assert s_b.initialize.await_count == 1

    async def test_concurrent_acquires_same_key_get_isolated_entries_and_sinks(self) -> None:
        """Two simultaneous checkouts of the same key get DISTINCT entries and
        DISTINCT HttpErrorSinks, and an HTTP error captured by one entry's
        transport hook never touches the other's sink — the core regression for
        the shared-sink cross-talk bug (#3)."""
        captured_hooks: list[Any] = []

        def fake_client(*_a: Any, **k: Any) -> MagicMock:
            captured_hooks.append(k["event_hooks"]["response"][0])
            m = MagicMock()
            m.__aenter__ = AsyncMock(return_value=m)
            m.__aexit__ = AsyncMock(return_value=False)
            return m

        s1, s2 = _make_mock_session(), _make_mock_session()
        with (
            patch("aios.mcp.pool.httpx.AsyncClient", fake_client),
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx_seq([s1, s2])),
        ):
            pool = McpSessionPool()
            e1 = await pool.acquire(URL, "v", EMPTY_KEY, {})
            e2 = await pool.acquire(URL, "v", EMPTY_KEY, {})  # e1 still in use → distinct entry

            assert e1 is not e2
            assert e1.error_sink is not e2.error_sink

            # Fire a 4xx through entry1's transport response hook only.
            resp = httpx.Response(403, text="forbidden", request=httpx.Request("POST", URL))
            await captured_hooks[0](resp)

        assert e1.error_sink.event.is_set()
        assert e1.error_sink.status == 403
        assert not e2.error_sink.event.is_set(), "entry2's sink must be untouched"

    async def test_discard_causes_reconnect_and_closes_entry(self) -> None:
        """discard drops a broken entry (closing it via _Entry.close so the owner
        task + httpx client + SSE stream are released) and the next acquire opens
        a fresh session."""
        s_a, s_b = _make_mock_session(), _make_mock_session()
        headers = {"Authorization": "Bearer tok"}
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx_seq([s_a, s_b])),
        ):
            pool = McpSessionPool()
            e1 = await pool.acquire(URL, "v", EMPTY_KEY, headers)
            await pool.discard(URL, "v", EMPTY_KEY, e1)
            # Let the fire-and-forget close task run.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert e1._owner_task.done(), "discard must close the entry's owner task"
            # discard removes the entry AND cleans up the now-empty in-use set
            # (no ghost keys accumulating across tenant churn).
            assert (URL, "v", EMPTY_KEY) not in pool._in_use

            e2 = await pool.acquire(URL, "v", EMPTY_KEY, headers)

        assert e1.session is not e2.session

    async def test_http_error_releases_not_discards(self) -> None:
        """Finding #9 at the pool layer: a healthy session returned via release
        is reused (not torn down) — the owner task stays alive."""
        session = _make_mock_session()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            pool = McpSessionPool()
            e1 = await pool.acquire(URL, "v", EMPTY_KEY, {})
            await pool.release(URL, "v", EMPTY_KEY, e1)  # the call_mcp_tool _McpHttpError path
            assert not e1._owner_task.done(), "release must NOT close the session"
            e2 = await pool.acquire(URL, "v", EMPTY_KEY, {})

        assert e2 is e1
        session.initialize.assert_awaited_once()

    async def test_release_rereads_discard_flag_set_while_it_waited_on_lock(self) -> None:
        """evict_by_vault flips ``discard_on_release`` while holding the key
        Condition. If release() reads that flag BEFORE acquiring the same lock,
        a concurrent eviction lands in the gap and the stale-token session is
        returned to idle for reuse — the #1030 rotated-credential gap reopened
        under a race. release() must re-read the flag UNDER the lock.
        """
        session = _make_mock_session()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            pool = McpSessionPool()
            key = (URL, "vault_a", EMPTY_KEY)
            e1 = await pool.acquire(URL, "vault_a", EMPTY_KEY, {"Authorization": "Bearer old"})

            # Hold the key Condition so release() blocks at `async with cond`,
            # then flag the entry (as evict_by_vault does under this same lock)
            # before letting release() proceed — the exact TOCTOU window.
            cond = pool._condition_for(key)
            await cond.acquire()
            rel = asyncio.create_task(pool.release(URL, "vault_a", EMPTY_KEY, e1))
            await asyncio.sleep(0)  # release() runs up to `async with cond` and parks
            e1.discard_on_release = True
            cond.release()
            await rel
            # Let the fire-and-forget close task run.
            await asyncio.sleep(0)
            await asyncio.sleep(0)

            assert e1._owner_task.done(), (
                "a flag set while release() waited on the lock must still discard the entry"
            )
            assert key not in pool._idle, (
                "the stale-token session must not be returned to idle for reuse"
            )

    async def test_concurrent_acquires_open_distinct_sessions_up_to_cap(self) -> None:
        """N concurrent checkouts of a cold key open N distinct sessions (no
        shared state) — replaces the old shared-session 'single initialize'
        guarantee, which was the bug."""
        sessions = [_make_mock_session() for _ in range(8)]
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx_seq(sessions)),
        ):
            pool = McpSessionPool()
            entries = await asyncio.gather(
                *[pool.acquire(URL, "v", EMPTY_KEY, {}) for _ in range(8)]
            )

        assert len({id(e) for e in entries}) == 8
        assert all(s.initialize.await_count == 1 for s in sessions)

    async def test_cap_blocks_then_release_unblocks_waiter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """At MAX_SESSIONS_PER_KEY the next acquire waits; a release unblocks it
        and hands back the freed (warm) entry."""
        monkeypatch.setattr("aios.mcp.pool.MAX_SESSIONS_PER_KEY", 2)
        sessions = [_make_mock_session() for _ in range(3)]
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx_seq(sessions)),
        ):
            pool = McpSessionPool()
            e1 = await pool.acquire(URL, "v", EMPTY_KEY, {})
            await pool.acquire(URL, "v", EMPTY_KEY, {})  # now at cap (2 in use)

            waiter = asyncio.create_task(pool.acquire(URL, "v", EMPTY_KEY, {}))
            await asyncio.sleep(0)
            assert not waiter.done(), "third acquire must block at the cap"

            await pool.release(URL, "v", EMPTY_KEY, e1)
            e3 = await asyncio.wait_for(waiter, timeout=1.0)

        assert e3 is e1, "the freed (idle) entry should be reused by the waiter"
        # Only two sessions ever opened — the cap held.
        assert sessions[2].initialize.await_count == 0

    async def test_acquire_after_close_raises(self) -> None:
        """close_all latches the pool closed; a later acquire fails loudly
        rather than silently opening a session that never gets torn down."""
        pool = McpSessionPool()
        await pool.close_all()
        with pytest.raises(RuntimeError):
            await pool.acquire(URL, "v", EMPTY_KEY, {})

    async def test_close_all_closes_idle_and_in_use(self) -> None:
        """close_all tears down BOTH released (idle) and checked-out (in-use)
        entries — an in-use owner task would otherwise leak at shutdown."""
        s_idle, s_in_use = _make_mock_session(), _make_mock_session()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx_seq([s_idle, s_in_use])),
        ):
            pool = McpSessionPool()
            e_idle = await pool.acquire(URL, "a", EMPTY_KEY, {})
            await pool.release(URL, "a", EMPTY_KEY, e_idle)  # → idle
            e_in_use = await pool.acquire(URL, "b", EMPTY_KEY, {})  # stays in use

            await pool.close_all()

        assert e_idle._owner_task.done()
        assert e_in_use._owner_task.done()
        assert not pool._idle and not pool._in_use

    async def test_close_all_empty_pool(self) -> None:
        """close_all on an empty pool is a no-op."""
        pool = McpSessionPool()
        await pool.close_all()
        assert not pool._idle and not pool._in_use

    async def test_idle_reaper_closes_idle_skips_in_use(self) -> None:
        """The reaper closes a stale IDLE entry but never an in-use one (an
        in-use entry is by definition active)."""
        s1, s2 = _make_mock_session(), _make_mock_session()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx_seq([s1, s2])),
        ):
            pool = McpSessionPool()
            e1 = await pool.acquire(URL, "v", EMPTY_KEY, {})
            e2 = await pool.acquire(URL, "v", EMPTY_KEY, {})  # two distinct in-use entries
            await pool.release(URL, "v", EMPTY_KEY, e1)  # e1 → idle, e2 stays in use
            e1.last_used = 1000.0
            e2.last_used = 1000.0  # also old, but in-use → protected

            await pool._reap_idle_once(idle_timeout=300.0, now=9100.0)

        assert e1._owner_task.done(), "stale idle entry must be reaped"
        assert not e2._owner_task.done(), "in-use entry must never be reaped"

    async def test_idle_reaper_skips_entry_popped_during_close_of_another(self) -> None:
        """TOCTOU: while the reaper awaits one stale entry's close(), a
        concurrent acquire pops a different idle entry. The reaper's re-check
        under the lock must see it gone and skip it (mirrors #654)."""
        park = asyncio.Event()
        cont = asyncio.Event()
        closed: list[Any] = []

        s1, s2 = _make_mock_session(), _make_mock_session()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx_seq([s1, s2])),
        ):
            pool = McpSessionPool()
            e1 = await pool.acquire(URL, "v", EMPTY_KEY, {})
            e2 = await pool.acquire(URL, "v", EMPTY_KEY, {})
            # Release in order so the idle list is [e1, e2]; pop() returns e2.
            await pool.release(URL, "v", EMPTY_KEY, e1)
            await pool.release(URL, "v", EMPTY_KEY, e2)

            orig1 = e1.close

            async def _close1() -> None:
                closed.append(e1)
                park.set()
                await cont.wait()
                await orig1()

            orig2 = e2.close

            async def _close2() -> None:
                closed.append(e2)
                await orig2()

            e1.close = _close1  # type: ignore[method-assign]
            e2.close = _close2  # type: ignore[method-assign]

            t0 = time.monotonic()
            e1.last_used = t0 - 1000.0
            e2.last_used = t0 - 1000.0

            reap = asyncio.create_task(pool._reap_idle_once(idle_timeout=300.0, now=t0))
            try:
                await asyncio.wait_for(park.wait(), timeout=1.0)
                # Reaper parked inside e1.close(). Acquire pops an idle entry (e2).
                popped = await pool.acquire(URL, "v", EMPTY_KEY, {})
                assert popped is e2
                cont.set()
                await asyncio.wait_for(reap, timeout=1.0)
            finally:
                if not reap.done():
                    reap.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await reap

        assert e2 not in closed, "reaper must not close an entry a concurrent acquire popped"

    async def test_oauth_refresh_reuses_key_no_orphan(self) -> None:
        """A token rotation on the same vault reuses the (url, vault_id) key —
        no orphaned second entry (the #459 invariant)."""
        session = _make_mock_session()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            pool = McpSessionPool()
            e1 = await pool.acquire(URL, "vault_a", EMPTY_KEY, {"Authorization": "Bearer old"})
            await pool.release(URL, "vault_a", EMPTY_KEY, e1)
            e2 = await pool.acquire(URL, "vault_a", EMPTY_KEY, {"Authorization": "Bearer new"})

        assert e2 is e1
        assert _live_count(pool, (URL, "vault_a", EMPTY_KEY)) == 1

    async def test_cross_tenant_isolation(self) -> None:
        """Two tenants sharing an MCP URL hold independent entries; one's
        rotation reuses its OWN entry without disturbing the other."""
        s_a, s_b = _make_mock_session(), _make_mock_session()
        url = "https://shared.example/"
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx_seq([s_a, s_b])),
        ):
            pool = McpSessionPool()
            a1 = await pool.acquire(url, "vault_A", EMPTY_KEY, {"Authorization": "Bearer tokA"})
            b1 = await pool.acquire(url, "vault_B", EMPTY_KEY, {"Authorization": "Bearer tokB"})
            await pool.release(url, "vault_A", EMPTY_KEY, a1)
            a2 = await pool.acquire(url, "vault_A", EMPTY_KEY, {"Authorization": "Bearer tokA_v2"})

        assert a2 is a1, "A's session is reused across rotation"
        assert b1.session is not a1.session, "tenants hold distinct entries"

    async def test_evict_by_vault_closes_idle_entries(self) -> None:
        """evict_by_vault closes every IDLE entry keyed on the vault_id and
        drops them from the idle pool, so the next acquire opens a fresh
        session on the rotated credential (#1030)."""
        s1, s2 = _make_mock_session(), _make_mock_session()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx_seq([s1, s2])),
        ):
            pool = McpSessionPool()
            e1 = await pool.acquire(URL, "vault_a", EMPTY_KEY, {"Authorization": "Bearer old"})
            await pool.release(URL, "vault_a", EMPTY_KEY, e1)  # → idle
            assert pool._idle[(URL, "vault_a", EMPTY_KEY)] == [e1]

            await pool.evict_by_vault("vault_a")

            assert e1._owner_task.done(), "evicted idle entry's owner task must be closed"
            assert (URL, "vault_a", EMPTY_KEY) not in pool._idle
            # Next acquire opens a brand-new session (no stale reuse).
            e2 = await pool.acquire(URL, "vault_a", EMPTY_KEY, {"Authorization": "Bearer new"})
            assert e2 is not e1
            assert e2.session is not e1.session

    async def test_evict_by_vault_only_targets_matching_vault(self) -> None:
        """evict_by_vault leaves entries for OTHER vault_ids untouched — only
        the rotated vault's pooled sessions are discarded."""
        s_a, s_b = _make_mock_session(), _make_mock_session()
        url = "https://shared.example/"
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx_seq([s_a, s_b])),
        ):
            pool = McpSessionPool()
            a1 = await pool.acquire(url, "vault_A", EMPTY_KEY, {})
            b1 = await pool.acquire(url, "vault_B", EMPTY_KEY, {})
            await pool.release(url, "vault_A", EMPTY_KEY, a1)
            await pool.release(url, "vault_B", EMPTY_KEY, b1)

            await pool.evict_by_vault("vault_A")

            assert a1._owner_task.done(), "vault_A's idle entry is closed"
            assert (url, "vault_A", EMPTY_KEY) not in pool._idle
            assert not b1._owner_task.done(), "vault_B's idle entry is untouched"
            assert pool._idle[(url, "vault_B", EMPTY_KEY)] == [b1]

    async def test_evict_by_vault_flags_in_use_then_release_discards(self) -> None:
        """evict_by_vault flags an IN-USE entry so the subsequent release
        DISCARDS it (closes + drops) instead of returning it to idle — the
        old token never gets reused by a later acquire (#1030)."""
        s1, s2 = _make_mock_session(), _make_mock_session()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx_seq([s1, s2])),
        ):
            pool = McpSessionPool()
            e1 = await pool.acquire(URL, "vault_a", EMPTY_KEY, {"Authorization": "Bearer old"})
            # Rotation lands while the entry is checked out.
            await pool.evict_by_vault("vault_a")
            assert not e1._owner_task.done(), "in-use entry stays open until release"
            assert (URL, "vault_a", EMPTY_KEY) in pool._in_use

            await pool.release(URL, "vault_a", EMPTY_KEY, e1)
            # release routes a flagged entry through discard (fire-and-forget
            # close) — let the close task run.
            await asyncio.sleep(0)
            await asyncio.sleep(0)

            assert e1._owner_task.done(), "flagged entry must be CLOSED on release"
            # Not returned to idle — a stale-token session must never be reused.
            assert (URL, "vault_a", EMPTY_KEY) not in pool._idle
            assert (URL, "vault_a", EMPTY_KEY) not in pool._in_use

            e2 = await pool.acquire(URL, "vault_a", EMPTY_KEY, {"Authorization": "Bearer new"})
            assert e2 is not e1
            assert e2.session is not e1.session

    async def test_evict_by_vault_unknown_vault_is_noop(self) -> None:
        """Evicting a vault with no pooled entries is a harmless no-op."""
        pool = McpSessionPool()
        await pool.evict_by_vault("nonexistent")  # must not raise

    async def test_release_without_flag_still_returns_to_idle(self) -> None:
        """A normal release (no eviction flag) returns the entry to idle — the
        flag path must not regress warm reuse."""
        session = _make_mock_session()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            pool = McpSessionPool()
            e1 = await pool.acquire(URL, "vault_a", EMPTY_KEY, {})
            await pool.release(URL, "vault_a", EMPTY_KEY, e1)
            assert pool._idle[(URL, "vault_a", EMPTY_KEY)] == [e1]
            assert not e1._owner_task.done()

    async def test_contexts_exit_in_same_task_they_entered(self) -> None:
        """Cross-task close — regression for #425. The transport's __aenter__ and
        __aexit__ must run in the SAME task (the owner-task model guarantees it;
        the legacy stack-on-_Entry model did not)."""
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
                await pool.acquire(URL, "v", EMPTY_KEY, {"Authorization": "tok"})

            await asyncio.create_task(opener())

            async def closer() -> None:
                await pool.close_all()

            await asyncio.create_task(closer())

        assert entered_in["open"] is not None
        assert entered_in["close"] is not None
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
        """Pool path: ``initialize`` called once even across two discovers
        (the second reuses the released idle session)."""
        from aios.mcp.client import discover_mcp_tools

        session = _make_mock_session()
        runtime.mcp_session_pool = McpSessionPool()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            await discover_mcp_tools(URL, "v", {}, "srv")
            await discover_mcp_tools(URL, "v", {}, "srv")

        session.initialize.assert_awaited_once()

    async def test_discards_and_retries_on_list_tools_failure(
        self, restore_runtime_pool: None
    ) -> None:
        """First ``list_tools`` raises → pool discards, second succeeds."""
        from aios.mcp.client import discover_mcp_tools

        s_a, s_b = _make_mock_session(), _make_mock_session()
        s_a.list_tools = AsyncMock(side_effect=RuntimeError("broken pipe"))
        s_b.list_tools = AsyncMock(return_value=MagicMock(tools=[]))

        runtime.mcp_session_pool = McpSessionPool()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx_seq([s_a, s_b])),
        ):
            tools, _ = await discover_mcp_tools(URL, "v", {}, "srv")

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
            tools, _ = await discover_mcp_tools(URL, "v", {}, "srv")

        assert tools == []
        session.initialize.assert_awaited_once()


class TestCallMcpToolWithPool:
    async def test_uses_pool_when_set(self, restore_runtime_pool: None) -> None:
        """Pool path: ``initialize`` called once even across two tool calls
        (the second reuses the released idle session)."""
        from aios.mcp.client import call_mcp_tool

        session = _make_mock_session()
        session.call_tool = AsyncMock(return_value=MagicMock(content=[], isError=False))

        runtime.mcp_session_pool = McpSessionPool()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(session)),
        ):
            await call_mcp_tool(URL, "v", {}, "do_thing", {})
            await call_mcp_tool(URL, "v", {}, "do_thing", {})

        session.initialize.assert_awaited_once()

    async def test_spec_headers_segment_pool_via_call(self, restore_runtime_pool: None) -> None:
        """Two ``call_mcp_tool`` calls with DIFFERENT ``spec_headers``
        open two distinct pooled sessions (the headers_key is part of the pool
        key); the SAME spec_headers reuses one. This proves spec_headers
        threads end-to-end through ``call_mcp_tool`` into the pool key."""
        from aios.mcp.client import call_mcp_tool

        s_a, s_b = _make_mock_session(), _make_mock_session()
        runtime.mcp_session_pool = McpSessionPool()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx_seq([s_a, s_b])),
        ):
            # First header set → opens s_a, released to idle.
            await call_mcp_tool(URL, "v", {}, "t", {}, spec_headers={"X-MCP-Toolsets": "issues"})
            # Same header set → reuses s_a (no new initialize).
            await call_mcp_tool(URL, "v", {}, "t", {}, spec_headers={"X-MCP-Toolsets": "issues"})
            # Different header set → distinct key → opens s_b.
            await call_mcp_tool(
                URL, "v", {}, "t", {}, spec_headers={"X-MCP-Toolsets": "discussions"}
            )

        pool = runtime.mcp_session_pool
        assert pool is not None
        all_keys = set(pool._idle) | set(pool._in_use)
        assert s_a.initialize.await_count == 1
        assert s_b.initialize.await_count == 1
        assert (URL, "v", K1) in all_keys
        assert (URL, "v", _headers_key({"X-MCP-Toolsets": "discussions"})) in all_keys

    async def test_does_not_retry_on_call_tool_failure(self, restore_runtime_pool: None) -> None:
        """A transport failure (broken pipe, TCP reset, HTTP/2 GOAWAY) may have
        landed after the server processed the request — retrying would duplicate
        the side effect. The wrapper discards and surfaces the error; the model
        decides whether to retry."""
        from aios.mcp.client import call_mcp_tool

        s_a = _make_mock_session()
        s_a.call_tool = AsyncMock(side_effect=RuntimeError("broken"))

        runtime.mcp_session_pool = McpSessionPool()
        with (
            patch("aios.mcp.pool.streamable_http_client", return_value=_transport_mock()),
            patch("aios.mcp.pool.ClientSession", _session_ctx(s_a)),
        ):
            result = await call_mcp_tool(URL, "v", {}, "do_thing", {})

        assert result.get("code") == "transport_error"
        assert s_a.call_tool.await_count == 1

    async def test_does_not_retry_call_tool_on_timeout(
        self, restore_runtime_pool: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A timeout in ``call_tool`` may mean the request reached the server;
        retrying would duplicate the side effect (e.g. a duplicate
        ``signal_send``). The wrapper must NOT retry — discard + surface."""
        from aios.mcp import client as mcp_client
        from aios.mcp.client import call_mcp_tool

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
            result = await call_mcp_tool(URL, "v", {}, "do_thing", {})

        assert result.get("code") == "transport_error"
        assert s_a.call_tool.await_count == 1
