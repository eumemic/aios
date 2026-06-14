"""B1.0 — workflow schema + queries against a real (testcontainer) Postgres.

Proves migration 0064 applies and that ``append_run_event`` is the single,
gapless, idempotent journal writer the runtime depends on.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.api.sse import wf_run_event_stream
from aios.db.listen import open_listen_for_run_events
from aios.db.pool import create_pool, register_jsonb_codec
from aios.db.queries import workflows as wf_queries
from aios.errors import ConflictError, NotFoundError
from aios.models.agents import ToolSpec
from aios.workflows.determinism import HOST_SEMANTICS_EPOCH


@pytest.fixture
async def wf_conn(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Connection[Any]]:
    """A conn with a single root tenant ``acc_root`` + env ``env_root``."""
    conn = await asyncpg.connect(migrated_db_url)
    # Mirror the production pool: query functions read jsonb as native Python.
    await register_jsonb_codec(conn)
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
        host_semantics_epoch=HOST_SEMANTICS_EPOCH,
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


async def test_duplicate_workflow_name_conflicts(wf_conn: asyncpg.Connection[Any]) -> None:
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
        host_semantics_epoch=HOST_SEMANTICS_EPOCH,
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


# ─── Block 3 surface: account-scoped reads + the stream NOTIFY ────────────────


async def test_account_scoped_reads_isolate_tenants(wf_conn: asyncpg.Connection[Any]) -> None:
    """The public reads (get_wf_run / list_wf_runs / list_workflows /
    list_run_events_scoped) never surface another account's data."""
    # A second tenant (a child account — only one active ROOT is allowed).
    await wf_conn.execute(
        "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
        "VALUES ('acc_other', 'acc_root', FALSE, 'tenant-other')"
    )
    run_id = await _seed_run(wf_conn)  # one workflow 'demo' + one run, under acc_root
    await wf_queries.append_run_event(
        wf_conn, account_id="acc_root", run_id=run_id, type="run_started", payload={"input": None}
    )

    # Owner sees everything.
    assert (await wf_queries.get_wf_run(wf_conn, run_id, account_id="acc_root")).id == run_id
    assert [r.id for r in await wf_queries.list_wf_runs(wf_conn, account_id="acc_root")] == [run_id]
    assert len(await wf_queries.list_workflows(wf_conn, account_id="acc_root")) == 1
    assert len(await wf_queries.list_run_events_scoped(wf_conn, run_id, account_id="acc_root")) == 1

    # The other tenant sees nothing — a 404 on the point read, empty lists otherwise.
    with pytest.raises(NotFoundError):
        await wf_queries.get_wf_run(wf_conn, run_id, account_id="acc_other")
    assert await wf_queries.list_wf_runs(wf_conn, account_id="acc_other") == []
    assert await wf_queries.list_workflows(wf_conn, account_id="acc_other") == []
    assert await wf_queries.list_run_events_scoped(wf_conn, run_id, account_id="acc_other") == []


async def test_list_run_events_scoped_paginates_forward(wf_conn: asyncpg.Connection[Any]) -> None:
    run_id = await _seed_run(wf_conn)
    for i in range(3):
        await wf_queries.append_run_event(
            wf_conn,
            account_id="acc_root",
            run_id=run_id,
            type="call_started",
            call_key=f"sha:k#{i}",
            payload={"capability": "gate"},
        )
    page1 = await wf_queries.list_run_events_scoped(
        wf_conn, run_id, account_id="acc_root", after_seq=0, limit=2
    )
    assert [e.seq for e in page1] == [1, 2]
    page2 = await wf_queries.list_run_events_scoped(
        wf_conn, run_id, account_id="acc_root", after_seq=page1[-1].seq, limit=2
    )
    assert [e.seq for e in page2] == [3]


async def test_append_run_event_notifies(
    wf_conn: asyncpg.Connection[Any], migrated_db_url: str
) -> None:
    """append_run_event fires a pg_notify on wf_run_events_<run_id> with the event
    id — the load-bearing signal the /runs/{id}/stream endpoint tails."""
    run_id = await _seed_run(wf_conn)
    listener = await asyncpg.connect(migrated_db_url)
    try:
        received: asyncio.Queue[str] = asyncio.Queue()
        await listener.add_listener(
            f"wf_run_events_{run_id}",
            lambda _conn, _pid, _channel, payload: received.put_nowait(payload),
        )
        event = await wf_queries.append_run_event(
            wf_conn, account_id="acc_root", run_id=run_id, type="run_started", payload={"input": 1}
        )
        assert event is not None
        payload = await asyncio.wait_for(received.get(), timeout=5)
        assert payload == event.id

        # An idempotent no-op append (deduped) must NOT notify.
        dup = await wf_queries.append_run_event(
            wf_conn, account_id="acc_root", run_id=run_id, type="run_started", payload={"input": 1}
        )
        assert dup is None
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(received.get(), timeout=0.5)
    finally:
        await listener.close()


async def test_append_run_event_notify_defers_to_outer_commit(
    wf_conn: asyncpg.Connection[Any], migrated_db_url: str
) -> None:
    """pg_notify is transactional: when an OUTER txn wraps append_run_event (the real
    _complete_run path), the notify is NOT delivered until that txn commits, and is
    not delivered at all on rollback. This guards the NOTIFY-after-commit invariant
    against a refactor moving the notify inside the txn (which the autocommit test
    can't catch)."""
    run_id = await _seed_run(wf_conn)
    listener = await asyncpg.connect(migrated_db_url)
    try:
        received: asyncio.Queue[str] = asyncio.Queue()
        await listener.add_listener(
            f"wf_run_events_{run_id}",
            lambda _conn, _pid, _channel, payload: received.put_nowait(payload),
        )

        async with wf_conn.transaction():
            event = await wf_queries.append_run_event(
                wf_conn, account_id="acc_root", run_id=run_id, type="run_started", payload={}
            )
            assert event is not None
            # Inside the open outer txn: the notify is queued, NOT yet delivered.
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(received.get(), timeout=0.5)
        # Outer txn committed → now it arrives.
        assert await asyncio.wait_for(received.get(), timeout=5) == event.id

        # Rollback path: the insert AND the notify are both undone → nothing delivered.
        with pytest.raises(RuntimeError):
            async with wf_conn.transaction():
                await wf_queries.append_run_event(
                    wf_conn,
                    account_id="acc_root",
                    run_id=run_id,
                    type="call_started",
                    call_key="sha:g#0",
                    payload={"capability": "gate"},
                )
                raise RuntimeError("force rollback")
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(received.get(), timeout=0.5)
    finally:
        await listener.close()


async def test_wf_run_event_stream_backfills_tails_and_terminates(
    wf_conn: asyncpg.Connection[Any], migrated_db_url: str
) -> None:
    """The SSE generator: a pre-subscription event arrives via backfill, a
    post-subscription event via the live NOTIFY tail, and ``run_completed`` ends the
    stream with a ``done`` frame."""
    run_id = await _seed_run(wf_conn)
    # Appended BEFORE we subscribe → must come through the backfill.
    await wf_queries.append_run_event(
        wf_conn, account_id="acc_root", run_id=run_id, type="run_started", payload={"input": 1}
    )

    pool = await create_pool(migrated_db_url, min_size=1, max_size=2)
    subscription = await open_listen_for_run_events(migrated_db_url, run_id)
    collected: list[tuple[str, str | None]] = []
    try:

        async def _consume() -> None:
            async for msg in wf_run_event_stream(subscription, pool, run_id):
                assert msg.event is not None
                assert isinstance(msg.data, str)
                kind = json.loads(msg.data).get("type") if msg.event == "event" else None
                collected.append((msg.event, kind))
                if msg.event == "done":
                    return

        task = asyncio.create_task(_consume())
        await asyncio.sleep(0.2)  # let the backfill drain, then block on the queue
        # Appended AFTER we subscribe → must come through the live NOTIFY tail.
        await wf_queries.append_run_event(
            wf_conn,
            account_id="acc_root",
            run_id=run_id,
            type="run_completed",
            payload={"output": 2, "is_error": False},
        )
        await asyncio.wait_for(task, timeout=5)
    finally:
        subscription.terminate()
        await pool.close()

    assert collected == [
        ("event", "run_started"),  # backfill
        ("event", "run_completed"),  # live tail
        ("done", None),  # terminal
    ]


# ─── update_workflow — in-place versioned updates (optimistic concurrency) ────


async def test_update_workflow_merges_and_bumps(wf_conn: asyncpg.Connection[Any]) -> None:
    wf = await wf_queries.insert_workflow(
        wf_conn, account_id="acc_root", name="up", script="A", description="d1"
    )
    updated = await wf_queries.update_workflow(
        wf_conn,
        wf.id,
        account_id="acc_root",
        expected_version=1,
        script="B",
        tools=[ToolSpec(type="web_search")],
    )
    assert updated.version == 2
    assert updated.script == "B"
    assert [t.type for t in updated.tools] == ["web_search"]
    # Omitted fields preserved; id stable.
    assert updated.name == "up" and updated.description == "d1" and updated.id == wf.id


async def test_update_workflow_noop_keeps_version(wf_conn: asyncpg.Connection[Any]) -> None:
    wf = await wf_queries.insert_workflow(wf_conn, account_id="acc_root", name="noop", script="A")
    same = await wf_queries.update_workflow(
        wf_conn, wf.id, account_id="acc_root", expected_version=1, script="A"
    )
    assert same.version == 1  # identical values → no bump
    assert (await wf_queries.get_workflow(wf_conn, wf.id, account_id="acc_root")).version == 1


async def test_update_workflow_stale_version_conflicts(wf_conn: asyncpg.Connection[Any]) -> None:
    wf = await wf_queries.insert_workflow(wf_conn, account_id="acc_root", name="stale", script="A")
    await wf_queries.update_workflow(
        wf_conn, wf.id, account_id="acc_root", expected_version=1, script="B"
    )  # → v2
    with pytest.raises(ConflictError):
        await wf_queries.update_workflow(
            wf_conn, wf.id, account_id="acc_root", expected_version=1, script="C"
        )
    # The contract is uniform: a stale token 409s even when the values are identical
    # (the write-free no-op path must not skip token validation).
    with pytest.raises(ConflictError):
        await wf_queries.update_workflow(
            wf_conn, wf.id, account_id="acc_root", expected_version=1, script="B"
        )
    with pytest.raises(NotFoundError):
        await wf_queries.update_workflow(
            wf_conn, "wf_does_not_exist", account_id="acc_root", expected_version=1, script="C"
        )


async def test_update_workflow_rename_collision_conflicts(
    wf_conn: asyncpg.Connection[Any],
) -> None:
    await wf_queries.insert_workflow(wf_conn, account_id="acc_root", name="taken", script="A")
    other = await wf_queries.insert_workflow(
        wf_conn, account_id="acc_root", name="renameme", script="A"
    )
    with pytest.raises(ConflictError):
        await wf_queries.update_workflow(
            wf_conn, other.id, account_id="acc_root", expected_version=1, name="taken"
        )


# ─── list_wf_runs — launcher filter (the agent list_runs default scoping) ─────


async def test_list_wf_runs_filters_by_launcher_session(wf_conn: asyncpg.Connection[Any]) -> None:
    """``launcher_session_id`` scopes the list to one session's own runs (including
    terminal ones), and composes with the other equality filters; omitting it lists the
    whole account. Backs the agent ``list_runs`` builtin's default (self) vs ``account_wide``."""
    # wf_runs.launcher_session_id REFERENCES sessions(id), so seed real sessions first:
    # the FK chain is accounts → environments → agents → sessions (acc_root/env_root
    # already seeded by the fixture). Minimal raw inserts — no sandbox/crypto path.
    await wf_conn.execute(
        "INSERT INTO agents (id, name, model, account_id) "
        "VALUES ('agn_wf', 'wf-agent', 'fake/test', 'acc_root')"
    )
    for ses in ("ses_a", "ses_b"):
        await wf_conn.execute(
            "INSERT INTO sessions (id, agent_id, environment_id, workspace_volume_path, account_id) "
            "VALUES ($1, 'agn_wf', 'env_root', $2, 'acc_root')",
            ses,
            f"/tmp/ws-{ses}",
        )

    wf = await wf_queries.insert_workflow(wf_conn, account_id="acc_root", name="demo", script="S")

    async def _mk_run(launcher: str | None) -> str:
        run = await wf_queries.insert_wf_run(
            wf_conn,
            account_id="acc_root",
            workflow_id=wf.id,
            environment_id="env_root",
            script=wf.script,
            script_sha="sha",
            host_semantics_epoch=HOST_SEMANTICS_EPOCH,
            launcher_session_id=launcher,
        )
        return run.id

    run_a = await _mk_run("ses_a")
    run_b = await _mk_run("ses_b")
    run_c = await _mk_run(None)

    # The launcher filter lists only that session's runs.
    by_a = await wf_queries.list_wf_runs(
        wf_conn, account_id="acc_root", launcher_session_id="ses_a"
    )
    assert [r.id for r in by_a] == [run_a]

    # No launcher → the whole account (all three).
    everything = await wf_queries.list_wf_runs(wf_conn, account_id="acc_root")
    assert {r.id for r in everything} == {run_a, run_b, run_c}

    # The launcher composes with another equality filter (workflow_id).
    by_b = await wf_queries.list_wf_runs(
        wf_conn, account_id="acc_root", launcher_session_id="ses_b", workflow_id=wf.id
    )
    assert [r.id for r in by_b] == [run_b]

    # A launcher with no runs → empty.
    assert (
        await wf_queries.list_wf_runs(
            wf_conn, account_id="acc_root", launcher_session_id="ses_a", workflow_id="wf_nope"
        )
        == []
    )


async def test_run_children_usage_sums_direct_children_and_includes_archived(
    wf_conn: asyncpg.Connection[Any],
) -> None:
    await wf_conn.execute(
        "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
        "VALUES ('acc_other', 'acc_root', FALSE, 'other')"
    )
    run_id = await _seed_run(wf_conn)
    await wf_conn.execute(
        "INSERT INTO agents (id, name, model, system, account_id) VALUES ('agent_usage', 'usage-agent', 'm', 's', 'acc_root')"
    )
    # Named workflow children must carry a pinned agent_version (0095's
    # sessions_agent_version_pair_ck: a child with agent_id set requires a
    # non-NULL agent_version); seed the matching agent_versions row.
    await wf_conn.execute(
        "INSERT INTO agent_versions (agent_id, version, model, system, account_id) "
        "VALUES ('agent_usage', 1, 'm', 's', 'acc_root')"
    )
    await wf_conn.execute(
        """
        INSERT INTO sessions (
            id, agent_id, environment_id, agent_version, title, metadata,
            workspace_volume_path, account_id, parent_run_id, input_tokens, output_tokens,
            cache_read_input_tokens, cache_creation_input_tokens, cost_microusd, archived_at
        ) VALUES
            ('ses_child_a', 'agent_usage', 'env_root', 1, NULL, '{}'::jsonb, '/tmp/a', 'acc_root', $1, 10, 20, 3, 4, 123456, NULL),
            ('ses_child_b', 'agent_usage', 'env_root', 1, NULL, '{}'::jsonb, '/tmp/b', 'acc_root', $1, 1, 2, 5, 6, 654321, now())
        """,
        run_id,
    )
    usage = await wf_queries.run_children_usage(wf_conn, run_id, account_id="acc_root")
    assert usage.input_tokens == 11
    assert usage.output_tokens == 22
    assert usage.cache_read_input_tokens == 8
    assert usage.cache_creation_input_tokens == 10
    assert usage.cost_microusd == 777777


async def test_run_children_usage_zero_and_account_scoped(wf_conn: asyncpg.Connection[Any]) -> None:
    run_id = await _seed_run(wf_conn)
    usage = await wf_queries.run_children_usage(wf_conn, run_id, account_id="acc_root")
    assert usage == wf_queries.RunChildrenUsage(0, 0, 0, 0, 0)
