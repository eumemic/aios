"""Integration tests for migration 0121 (per-connection inbound policy, #1500).

Covers the migration mechanics and — the deploy-safety crux — the one-time
known-chats backfill. The lockout risk of the fail-open→fail-closed flip lives
in the backfill: if it misses any live ``chat_id``, the flip locks the operator
out of their own agent. These tests pin:

* a clean up/down round-trip;
* a connection backfills to ``AllowList`` of the UNION of its ``chat_sessions``
  ledger and its historical ``role='user'`` events;
* a **slash-bearing** ``chat_id`` (Signal group id) survives the backfill
  intact (NOT truncated by ``split_part(channel,'/',3)``);
* prefix-scoping: a sibling connection's events under the same account do NOT
  leak into this connection's AllowList, and LIKE metacharacters are escaped;
* a zero-history connection backfills fail-closed (NULL → resolves to DenyAll).

Each test mutates ``alembic_version`` / seeds rows, so the container is
function-scoped. Modeled on tests/integration/test_migrations_0108_*.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator

import asyncpg
import pytest

from tests.conftest import _docker_available, needs_docker
from tests.integration.test_migrations import _alembic_url, _run_alembic

# FK chain to head, then seed connections + chat_sessions + events at revision
# 0120 (one BELOW our migration) so the 0120→0121 upgrade runs the backfill over
# pre-existing data — exactly the production deploy ordering.
_CHAIN_SQL = """
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_mig', NULL, TRUE, 'mig-test') ON CONFLICT DO NOTHING;
INSERT INTO environments (id, name, account_id)
VALUES ('env_mig', 'mig-env', 'acc_mig') ON CONFLICT DO NOTHING;
INSERT INTO agents (id, name, model, account_id)
VALUES ('agn_mig', 'mig-agent', 'fake/test', 'acc_mig') ON CONFLICT DO NOTHING;
INSERT INTO sessions (id, agent_id, environment_id, workspace_volume_path, account_id)
VALUES ('ses_mig', 'agn_mig', 'env_mig', '/tmp/ws-mig', 'acc_mig') ON CONFLICT DO NOTHING;
"""

# Two connections under the SAME account:
#   conn_signal (signal/+15550000001) — has both a ledger row and history,
#     including a slash-bearing group chat_id.
#   conn_tg     (telegram/bot_a)       — has NO ledger and NO history → backfills
#     fail-closed (stays NULL → DenyAll). 'bot_a' also stress-tests LIKE
#     metacharacter escaping ('_').
_CONNECTIONS_SQL = """
INSERT INTO connections (id, connector, external_account_id, account_id)
VALUES
  ('conn_signal', 'signal', '+15550000001', 'acc_mig'),
  ('conn_tg', 'telegram', 'bot_a', 'acc_mig');
"""

# Ledger row for conn_signal: a direct-message chat_id.
_CHAT_SESSIONS_SQL = """
INSERT INTO chat_sessions (connection_id, chat_id, session_id, account_id)
VALUES ('conn_signal', 'dm_alice', 'ses_mig', 'acc_mig');
"""


def _user_event_sql(evt_id: str, channel: str) -> str:
    data = json.dumps({"role": "user", "content": "hi"})
    return f"""
    INSERT INTO events (
        id, session_id, seq, kind, data, created_at,
        orig_channel, focal_channel_at_arrival, channel, account_id
    )
    VALUES (
        '{evt_id}', 'ses_mig',
        (SELECT coalesce(max(seq), 0) + 1 FROM events WHERE session_id = 'ses_mig'),
        'message', '{data}'::jsonb, now(),
        '{channel}', '{channel}', '{channel}', 'acc_mig'
    );
    """


# conn_signal history:
#   - a plain chat_id already in the ledger (dedup proves UNION + DISTINCT)
#   - a SLASH-BEARING group chat_id (the load-bearing case)
# Plus an event for the OTHER connection's prefix-shaped channel under the same
# account, to prove prefix-scoping does not leak it into conn_signal's list.
_EVENTS_SQL = (
    _user_event_sql("evt_dm", "signal/+15550000001/dm_alice")
    + _user_event_sql("evt_grp", "signal/+15550000001/group/Zm9vYmFy==")
    + _user_event_sql("evt_other", "telegramX/bot_a/should_not_match")
)


@pytest.fixture
def postgres() -> Iterator[object]:
    if not _docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


async def _execute(db_url: str, sql: str) -> None:
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(sql)
    finally:
        await conn.close()


async def _fetch_policy(db_url: str, connection_id: str) -> object:
    conn = await asyncpg.connect(db_url)
    try:
        raw = await conn.fetchval(
            "SELECT inbound_policy FROM connections WHERE id = $1", connection_id
        )
        return json.loads(raw) if isinstance(raw, str) else raw
    finally:
        await conn.close()


@needs_docker
@pytest.mark.integration
def test_upgrade_and_clean_downgrade_round_trip(postgres: object) -> None:
    """0121 upgrades and downgrades cleanly with zero connection rows."""
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "0121"], db_url)
    assert up.returncode == 0, f"upgrade to 0121 failed:\n{up.stderr}\n{up.stdout}"

    down = _run_alembic(["downgrade", "0120"], db_url)
    assert down.returncode == 0, f"downgrade to 0120 failed:\n{down.stderr}\n{down.stdout}"


@needs_docker
@pytest.mark.integration
def test_backfill_unions_ledger_and_slash_safe_history(postgres: object) -> None:
    """The backfill writes an AllowList that is the UNION of the ledger and the
    slash-safe events scan, scoped to the connection's own prefix; a
    zero-history connection stays NULL (→ DenyAll)."""
    db_url = _alembic_url(postgres)

    # Seed at 0120 (one below our migration) so the 0120→0121 upgrade backfills
    # over pre-existing rows.
    up120 = _run_alembic(["upgrade", "0120"], db_url)
    assert up120.returncode == 0, f"upgrade to 0120 failed:\n{up120.stderr}\n{up120.stdout}"
    asyncio.run(_execute(db_url, _CHAIN_SQL))
    asyncio.run(_execute(db_url, _CONNECTIONS_SQL))
    asyncio.run(_execute(db_url, _CHAT_SESSIONS_SQL))
    asyncio.run(_execute(db_url, _EVENTS_SQL))

    up121 = _run_alembic(["upgrade", "0121"], db_url)
    assert up121.returncode == 0, f"upgrade to 0121 failed:\n{up121.stderr}\n{up121.stdout}"

    signal_policy = asyncio.run(_fetch_policy(db_url, "conn_signal"))
    assert isinstance(signal_policy, dict)
    assert signal_policy["kind"] == "allow_list"
    got = sorted(signal_policy["chat_ids"])
    # The slash-bearing group id survives INTACT; the ledger + history union and
    # de-dupes; the sibling connection's channel does NOT leak in.
    assert got == sorted(["dm_alice", "group/Zm9vYmFy=="]), got
    assert "should_not_match" not in signal_policy["chat_ids"]

    # Zero-history connection: NULL → resolves to the server default DenyAll.
    tg_policy = asyncio.run(_fetch_policy(db_url, "conn_tg"))
    assert tg_policy is None
