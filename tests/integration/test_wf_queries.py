"""B1.0 — workflow schema + queries against a real (testcontainer) Postgres.

Proves migration 0064 applies and that ``append_run_event`` is the single,
gapless, idempotent journal writer the runtime depends on.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db.queries import workflows as wf_queries
from aios.errors import ConflictError


@pytest.fixture
async def wf_conn(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Connection[Any]]:
    """A conn with a single root tenant ``acc_root`` + env ``env_root``."""
    conn = await asyncpg.connect(migrated_db_url)
    try:
        await conn.execute(
            "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
            "VALUES ('acc_root', NULL, TRUE, 'tenant-root')"
        )
        await conn.execute(
            "INSERT INTO environments (id, name, config, account_id) "
            "VALUES ('env_root', 'wf-env', '{}'::jsonb, 'acc_root')"
        )
        yield conn
    finally:
        await conn.close()


async def _seed_run(conn: asyncpg.Connection[Any]) -> str:
    wf = await wf_queries.insert_workflow(
        conn, account_id="acc_root", name="demo", script="async def main(input):\n    return 1\n"
    )
    run = await wf_queries.insert_wf_run(
        conn,
        account_id="acc_root",
        workflow_id=wf.id,
        environment_id="env_root",
        script=wf.script,
        script_sha="deadbeef",
    )
    return run.id


async def test_insert_and_get_workflow(wf_conn: asyncpg.Connection[Any]) -> None:
    wf = await wf_queries.insert_workflow(
        wf_conn,
        account_id="acc_root",
        name="demo",
        script="async def main(input):\n    return input\n",
        input_schema={"type": "object"},
    )
    assert wf.id.startswith("wf_")
    assert wf.version == 1
    fetched = await wf_queries.get_workflow(wf_conn, wf.id, account_id="acc_root")
    assert fetched.script == wf.script
    assert fetched.input_schema == {"type": "object"}


async def test_duplicate_workflow_version_conflicts(wf_conn: asyncpg.Connection[Any]) -> None:
    await wf_queries.insert_workflow(wf_conn, account_id="acc_root", name="dup", script="x")
    with pytest.raises(ConflictError):
        await wf_queries.insert_workflow(wf_conn, account_id="acc_root", name="dup", script="y")


async def test_insert_run_snapshots_script(wf_conn: asyncpg.Connection[Any]) -> None:
    wf = await wf_queries.insert_workflow(
        wf_conn, account_id="acc_root", name="demo", script="SRC-V1"
    )
    run = await wf_queries.insert_wf_run(
        wf_conn,
        account_id="acc_root",
        workflow_id=wf.id,
        environment_id="env_root",
        script=wf.script,
        script_sha="sha-v1",
    )
    assert run.id.startswith("wfr_")
    assert run.status == "pending"
    assert run.environment_id == "env_root"
    assert run.script == "SRC-V1"
    assert run.script_sha == "sha-v1"
    assert run.last_event_seq == 0
    reloaded = await wf_queries.get_run_for_step(wf_conn, run.id)
    assert reloaded is not None and reloaded.script == "SRC-V1"


async def test_append_run_event_gapless_idempotent_terminal(
    wf_conn: asyncpg.Connection[Any],
) -> None:
    run_id = await _seed_run(wf_conn)

    e1 = await wf_queries.append_run_event(
        wf_conn, account_id="acc_root", run_id=run_id, type="run_started", payload={"input": None}
    )
    assert e1 is not None and e1.seq == 1

    # NULLS NOT DISTINCT memo: a second run_started (call_key IS NULL) is deduped
    # too — bookends, not just call-keyed events — so no duplicate, no seq burned.
    dup_started = await wf_queries.append_run_event(
        wf_conn, account_id="acc_root", run_id=run_id, type="run_started", payload={"input": None}
    )
    assert dup_started is None

    key = "sha:abc#0"
    e2 = await wf_queries.append_run_event(
        wf_conn,
        account_id="acc_root",
        run_id=run_id,
        type="call_started",
        call_key=key,
        payload={"capability": "gate"},
    )
    assert e2 is not None and e2.seq == 2

    # Idempotent: same (run_id, call_key, type) → no-op, no seq consumed.
    dup = await wf_queries.append_run_event(
        wf_conn,
        account_id="acc_root",
        run_id=run_id,
        type="call_started",
        call_key=key,
        payload={"capability": "gate"},
    )
    assert dup is None

    # Different type, same key → distinct memo slot, next seq (no gap from the dup).
    e3 = await wf_queries.append_run_event(
        wf_conn,
        account_id="acc_root",
        run_id=run_id,
        type="call_result",
        call_key=key,
        payload={"result": 42},
    )
    assert e3 is not None and e3.seq == 3

    run = await wf_queries.get_run_for_step(wf_conn, run_id)
    assert run is not None and run.last_event_seq == 3
    events = await wf_queries.list_run_events(wf_conn, run_id)
    assert [e.seq for e in events] == [1, 2, 3]

    # Terminal guard: a completed run rejects further appends (no-op).
    await wf_queries.set_run_status(wf_conn, run_id, "completed", account_id="acc_root")
    late = await wf_queries.append_run_event(
        wf_conn, account_id="acc_root", run_id=run_id, type="run_completed", payload={}
    )
    assert late is None


async def test_run_signal_idempotent(wf_conn: asyncpg.Connection[Any]) -> None:
    run_id = await _seed_run(wf_conn)
    s1 = await wf_queries.insert_run_signal(
        wf_conn, run_id=run_id, call_key="sha:g#0", kind="gate_resume", result={"answer": "yes"}
    )
    assert s1.result == {"answer": "yes"}
    # Second delivery returns the existing row, not an error.
    s2 = await wf_queries.insert_run_signal(
        wf_conn, run_id=run_id, call_key="sha:g#0", kind="gate_resume", result={"answer": "OTHER"}
    )
    assert s2.result == {"answer": "yes"}
    signals = await wf_queries.list_run_signals(wf_conn, run_id)
    assert len(signals) == 1
