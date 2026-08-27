"""Integration coverage for the browser control plane (jarbot#106 §5.7).

Real Postgres (testcontainer), stubbed driver: the submit→LISTEN→execute→
resolve round-trip, the lost-NOTIFY redrive, expiry-without-execution, grant
lifecycle (one-open-per-account by construction, close + handback, TTL
expiry with the model-visible notice), and clear-state's open-grant refusal.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest

from aios.config import get_settings
from aios.db import queries
from aios.db.pool import create_pool
from aios.harness import browser_control
from aios.harness.worker import _run_browser_call_listener
from aios.ids import (
    ACCOUNT,
    AGENT,
    BROWSER_CALL,
    BROWSER_GRANT,
    ENVIRONMENT,
    SESSION,
    make_id,
)
from aios.sandbox.browser_protocol import BrowserResponse
from aios.sandbox.volumes import ensure_browser_plane_dir
from aios.services.browser_calls import submit_browser_call

pytestmark = pytest.mark.integration


async def _seed_account(conn: asyncpg.Connection[Any]) -> str:
    """A fresh CHILD account per test (the one-active-root index permits only
    a single non-archived root, shared across the whole test DB)."""
    root_id = await conn.fetchval(
        "SELECT id FROM accounts WHERE parent_account_id IS NULL AND archived_at IS NULL"
    )
    if root_id is None:
        root_id = make_id(ACCOUNT)
        await conn.execute(
            "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
            "VALUES ($1, NULL, TRUE, 'browser-test-root')",
            root_id,
        )
    account_id = make_id(ACCOUNT)
    await conn.execute(
        "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
        "VALUES ($1, $2, FALSE, $3)",
        account_id,
        root_id,
        f"acct-{account_id[-6:]}",
    )
    return account_id


async def _seed_session(conn: asyncpg.Connection[Any], account_id: str) -> str:
    env_id, agent_id, session_id = make_id(ENVIRONMENT), make_id(AGENT), make_id(SESSION)
    await conn.execute(
        "INSERT INTO environments (id, name, config, account_id) VALUES ($1, $2, '{}'::jsonb, $3)",
        env_id,
        f"env-{env_id[-6:]}",
        account_id,
    )
    await conn.execute(
        "INSERT INTO agents (id, name, model, system, tools, skills, mcp_servers, "
        "http_servers, description, metadata, litellm_extra, window_min, window_max, "
        "preempt_policy, version, account_id) "
        "VALUES ($1, $2, 'openrouter/test', '', '[]'::jsonb, '[]'::jsonb, '[]'::jsonb, "
        "'[]'::jsonb, NULL, '{}'::jsonb, '{}'::jsonb, 50000, 150000, 'wait', 1, $3)",
        agent_id,
        f"agent-{agent_id[-6:]}",
        account_id,
    )
    await conn.execute(
        "INSERT INTO sessions (id, agent_id, environment_id, agent_version, title, "
        "metadata, workspace_volume_path, env, account_id) "
        "VALUES ($1, $2, $3, 1, NULL, '{}'::jsonb, $4, '{}'::jsonb, $5)",
        session_id,
        agent_id,
        env_id,
        f"/tmp/{session_id}",
        account_id,
    )
    return session_id


def _driver_response(**overrides: Any) -> BrowserResponse:
    base: dict[str, Any] = {
        "ok": True,
        "boot": "01BOOTTEST",
        "epoch": 7,
        "url": "https://accounts.example.com/signin",
        "title": "Sign in",
        "snapshot": '- textbox "Email" [ref=e3]',
        "data": {"target": {"url": "https://accounts.example.com/signin"}},
    }
    base.update(overrides)
    return BrowserResponse.model_validate(base)


@pytest.fixture
async def plane(migrated_db_url: str, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A pool + seeded account/session + stubbed driver + running listener."""
    # aios's own pool factory: registers the jsonb codec the query layer
    # relies on (raw asyncpg pools return jsonb columns as strings).
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    async with pool.acquire() as conn:
        account_id = await _seed_account(conn)
        session_id = await _seed_session(conn, account_id)

    driver = AsyncMock(return_value=_driver_response())
    monkeypatch.setattr(browser_control, "driver_call", driver)
    registry = MagicMock()
    registry.touch_browser = MagicMock()
    registry.release_browser = AsyncMock()
    # A live container is cached: _close_takeover's peek gate must find a handle
    # to run the driver handback (peek None -> no handback, by design).
    registry.peek.return_value = MagicMock()
    # clear_state serializes its wipe under the owner lock; a real asyncio.Lock
    # is a working async context manager for the stub.
    registry.owner_lock.return_value = asyncio.Lock()

    listener = asyncio.create_task(_run_browser_call_listener(migrated_db_url, registry, pool))
    # Let the LISTEN connection establish before the first submit.
    await asyncio.sleep(0.3)
    try:
        yield {
            "pool": pool,
            "db_url": migrated_db_url,
            "account_id": account_id,
            "session_id": session_id,
            "driver": driver,
            "registry": registry,
        }
    finally:
        listener.cancel()
        with pytest.raises(asyncio.CancelledError):
            await listener
        await pool.close()


async def test_submit_round_trip_open_then_conflict_then_close(plane: dict[str, Any]) -> None:
    pool, db_url = plane["pool"], plane["db_url"]
    account_id, session_id = plane["account_id"], plane["session_id"]

    # The grant_id is minted by the caller (the API) and threaded through
    # params — the executor reads it fail-hard so a redrive re-presents the
    # SAME id (idempotent takeover_open), never a fresh one.
    grant_id = make_id(BROWSER_GRANT)
    result, is_error = await submit_browser_call(
        db_url,
        pool,
        account_id=account_id,
        method="open",
        params={"session_id": session_id, "reason": "auth", "grant_id": grant_id},
        timeout_s=10,
    )
    assert not is_error
    assert result["epoch"] == 7
    assert result["grant_id"] == grant_id
    async with pool.acquire() as conn:
        grant = await queries.get_browser_grant(conn, grant_id, account_id=account_id)
    assert grant is not None and grant["status"] == "open"
    assert grant["boot"] == "01BOOTTEST"

    # One open grant per computer, by construction (partial unique index): a
    # DIFFERENT grant_id still conflicts on the account.
    result2, is_error2 = await submit_browser_call(
        db_url,
        pool,
        account_id=account_id,
        method="open",
        params={"session_id": session_id, "reason": "auth", "grant_id": make_id(BROWSER_GRANT)},
        timeout_s=10,
    )
    assert is_error2
    assert result2["code"] == "takeover_in_progress"

    plane["driver"].return_value = _driver_response(
        data={"signed_in_hosts": ["accounts.example.com"]},
        shot_path="shots/handback.png",
    )
    result3, is_error3 = await submit_browser_call(
        db_url,
        pool,
        account_id=account_id,
        method="close",
        params={"grant_id": grant_id, "outcome": "done"},
        timeout_s=10,
    )
    assert not is_error3
    assert result3["handback"]["signed_in_hosts"] == ["accounts.example.com"]
    async with pool.acquire() as conn:
        grant = await queries.get_browser_grant(conn, grant_id, account_id=account_id)
        events = await conn.fetch(
            "SELECT data FROM events WHERE session_id = $1 AND kind = 'lifecycle'",
            session_id,
        )
    assert grant is not None and grant["status"] == "closed" and grant["outcome"] == "done"
    assert any(e["data"].get("event") == "browser_takeover_ended" for e in events)


async def test_redrive_executes_a_row_inserted_while_listener_was_down(
    plane: dict[str, Any],
) -> None:
    """The lost-NOTIFY hole: a pending row whose NOTIFY was never sent still
    executes, because a (re)connecting listener redrives durable pending rows —
    proven by inserting the row silently, then starting a fresh listener and
    watching the row resolve with no dispatch NOTIFY ever fired."""
    pool = plane["pool"]
    account_id = plane["account_id"]
    call_id = make_id(BROWSER_CALL)
    async with pool.acquire() as conn:
        await queries.insert_browser_call(
            conn,
            account_id=account_id,
            call_id=call_id,
            method="status",
            params={},
            expires_at=datetime.now(UTC) + timedelta(seconds=30),
        )
    # No NOTIFY was sent. A fresh listener redrives pending rows on connect.
    registry = MagicMock()
    registry.peek.return_value = None  # status never provisions
    task = asyncio.create_task(_run_browser_call_listener(plane["db_url"], registry, pool))
    try:
        for _ in range(50):
            async with pool.acquire() as conn:
                row = await queries.get_browser_call(conn, call_id, account_id=account_id)
            assert row is not None
            if row["status"] != "pending":
                break
            await asyncio.sleep(0.1)
        assert row is not None
        assert row["status"] == "succeeded"
        assert row["result"] == {"running": False}
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_expired_pending_row_resolves_failed_without_execution(
    plane: dict[str, Any],
) -> None:
    pool = plane["pool"]
    account_id = plane["account_id"]
    call_id = make_id(BROWSER_CALL)
    async with pool.acquire() as conn:
        await queries.insert_browser_call(
            conn,
            account_id=account_id,
            call_id=call_id,
            method="status",
            params={},
            expires_at=datetime.now(UTC) - timedelta(seconds=1),
        )
    plane["driver"].reset_mock()
    await browser_control.execute_browser_call(MagicMock(), pool, call_id)
    async with pool.acquire() as conn:
        row = await queries.get_browser_call(conn, call_id, account_id=account_id)
    assert row is not None and row["status"] == "failed"
    assert row["result"]["code"] == "expired"
    plane["driver"].assert_not_awaited()


async def test_grant_ttl_expiry_appends_the_model_visible_notice(
    plane: dict[str, Any],
) -> None:
    pool = plane["pool"]
    account_id, session_id = plane["account_id"], plane["session_id"]
    grant_id = make_id(BROWSER_GRANT)
    async with pool.acquire() as conn:
        await queries.insert_browser_grant(
            conn,
            grant_id=grant_id,
            account_id=account_id,
            session_id=session_id,
            reason="auth",
            boot="01BOOTTEST",
            epoch=3,
            target={"url": "https://example.com"},
            ttl_seconds=60,
        )
        # Lapse the heartbeat far past the TTL.
        await conn.execute(
            "UPDATE browser_grants SET heartbeat_at = now() - interval '10 minutes' WHERE id = $1",
            grant_id,
        )

    await browser_control.browser_reaper_tick(plane["registry"], pool)

    async with pool.acquire() as conn:
        grant = await queries.get_browser_grant(conn, grant_id, account_id=account_id)
        events = await conn.fetch(
            "SELECT data FROM events WHERE session_id = $1 AND kind = 'lifecycle'",
            session_id,
        )
    assert grant is not None and grant["status"] == "expired"
    ended = [e for e in events if e["data"].get("event") == "browser_takeover_ended"]
    assert ended and ended[-1]["data"]["outcome"] == "expired"


async def test_fresh_grant_heartbeat_bumps_the_container_keepalive(
    plane: dict[str, Any],
) -> None:
    pool = plane["pool"]
    account_id, session_id = plane["account_id"], plane["session_id"]
    async with pool.acquire() as conn:
        await queries.insert_browser_grant(
            conn,
            grant_id=make_id(BROWSER_GRANT),
            account_id=account_id,
            session_id=session_id,
            reason="auth",
            boot="01BOOTTEST",
            epoch=3,
            target={},
            ttl_seconds=300,
        )
    await browser_control.browser_reaper_tick(plane["registry"], pool)
    plane["registry"].touch_browser.assert_called_with(account_id)


async def test_clear_state_refused_while_a_takeover_is_open(plane: dict[str, Any]) -> None:
    pool, db_url = plane["pool"], plane["db_url"]
    account_id, session_id = plane["account_id"], plane["session_id"]
    async with pool.acquire() as conn:
        await queries.insert_browser_grant(
            conn,
            grant_id=make_id(BROWSER_GRANT),
            account_id=account_id,
            session_id=session_id,
            reason="auth",
            boot="01BOOTTEST",
            epoch=3,
            target={},
            ttl_seconds=300,
        )
    result, is_error = await submit_browser_call(
        db_url,
        pool,
        account_id=account_id,
        method="clear_state",
        params={},
        timeout_s=10,
    )
    assert is_error
    assert result["code"] == "takeover_open"
    plane["registry"].release_browser.assert_not_awaited()


async def test_open_without_grant_id_fails_hard(plane: dict[str, Any]) -> None:
    """The executor reads ``params['grant_id']`` fail-hard — NO fallback mint.
    An open missing it resolves is_error rather than silently minting a fresh id
    that a lost-NOTIFY redrive could then double-open (the CRITICAL this fix
    closes)."""
    pool, db_url = plane["pool"], plane["db_url"]
    account_id, session_id = plane["account_id"], plane["session_id"]
    result, is_error = await submit_browser_call(
        db_url,
        pool,
        account_id=account_id,
        method="open",
        params={"session_id": session_id, "reason": "auth"},  # no grant_id
        timeout_s=10,
    )
    assert is_error
    assert result["code"] == "internal"  # KeyError -> the always-resolve backstop


async def test_close_of_a_dead_container_still_closes_with_null_handback(
    plane: dict[str, Any],
) -> None:
    """A close when the container is gone (peek None) must NOT cold-provision one
    just to hand back — the grant still reaches terminal (viewer stops) with a
    null handback, and the driver is never touched (reaper-starvation fix)."""
    pool, db_url = plane["pool"], plane["db_url"]
    account_id, session_id = plane["account_id"], plane["session_id"]
    plane["registry"].peek.return_value = None  # container gone
    grant_id = make_id(BROWSER_GRANT)
    async with pool.acquire() as conn:
        await queries.insert_browser_grant(
            conn,
            grant_id=grant_id,
            account_id=account_id,
            session_id=session_id,
            reason="auth",
            boot="01BOOTTEST",
            epoch=3,
            target={},
            ttl_seconds=300,
        )
    plane["driver"].reset_mock()
    result, is_error = await submit_browser_call(
        db_url,
        pool,
        account_id=account_id,
        method="close",
        params={"grant_id": grant_id, "outcome": "done"},
        timeout_s=10,
    )
    assert not is_error
    assert result["handback"] is None
    plane["driver"].assert_not_awaited()  # never cold-provisioned to hand back
    async with pool.acquire() as conn:
        grant = await queries.get_browser_grant(conn, grant_id, account_id=account_id)
    assert grant is not None and grant["status"] == "closed"


async def test_clear_state_wipes_the_plane_and_notifies(
    plane: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no open grant, clear_state releases the container, wipes the plane
    subdirs (login state), recreates them empty, and posts browser_state_lost —
    all under the owner lock, with no ignore_errors masking a partial wipe."""
    pool, db_url = plane["pool"], plane["db_url"]
    account_id, session_id = plane["account_id"], plane["session_id"]
    monkeypatch.setattr(get_settings(), "workspace_root", tmp_path)
    plane_dir = ensure_browser_plane_dir(account_id)
    cookie = plane_dir / "profile" / "Cookies"
    cookie.write_bytes(b"secret-session-token")

    _result, is_error = await submit_browser_call(
        db_url,
        pool,
        account_id=account_id,
        method="clear_state",
        params={"session_id": session_id},
        timeout_s=10,
    )
    assert not is_error
    plane["registry"].release_browser.assert_awaited_with(account_id)
    assert not cookie.exists()  # login state actually deleted
    assert (plane_dir / "profile").is_dir()  # subdirs recreated empty
    async with pool.acquire() as conn:
        events = await conn.fetch(
            "SELECT data FROM events WHERE session_id = $1 AND kind = 'lifecycle'",
            session_id,
        )
    assert any(e["data"].get("event") == "browser_state_lost" for e in events)
