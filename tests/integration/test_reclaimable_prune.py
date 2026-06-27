"""T6 (#1461) — age-based prune for RECLAIMABLE instance ephemera.

Proves the ratified rule against a real (testcontainer) Postgres:

    prune reclaimable instance ephemera and unreferenced history past a
    retention window; NEVER prune anything a live session pins or that
    constitutes institutional memory.

The headline acceptance test (``test_prune_reclaims_ephemera_but_spares_sacred``)
asserts BOTH directions in one sweep: an unreferenced terminal+archived run (and
its ``WfRunEvent`` journal) is removed, while a live-pinned agent version and a
memory store survive. The remaining tests cover idempotency, the time window,
the live-pin guard for each definition family, and the kill-switch.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.config import get_settings
from aios.db.pool import create_pool, register_jsonb_codec
from aios.db.queries import workflows as wf_queries
from aios.db.queries.prune import (
    prune_archived_runs,
    prune_unpinned_archived_agents,
    prune_unpinned_archived_skills,
    prune_unpinned_archived_workflows,
)
from aios.harness.reclaimable_prune import sweep_reclaimable_ephemera
from aios.workflows.determinism import HOST_SEMANTICS_EPOCH


@pytest.fixture
async def conn(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[asyncpg.Connection[Any]]:
    """A conn with a single root tenant ``acc_root`` + env ``env_root``."""
    c = await asyncpg.connect(migrated_db_url)
    await register_jsonb_codec(c)
    try:
        await c.execute(
            "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
            "VALUES ('acc_root', NULL, TRUE, 'tenant-root')"
        )
        await c.execute(
            "INSERT INTO environments (id, name, config, account_id) "
            "VALUES ('env_root', 'env', '{}'::jsonb, 'acc_root')"
        )
        yield c
    finally:
        await c.close()


async def _make_archived_run(
    conn: asyncpg.Connection[Any],
    *,
    archived_age_days: int,
    seq_journal: bool = True,
) -> str:
    """Insert a workflow + a TERMINAL run, archive it, then back-date its
    ``archived_at`` by ``archived_age_days``. Returns the run id."""
    wf = await wf_queries.insert_workflow(
        conn, account_id="acc_root", name=f"wf_{archived_age_days}_{seq_journal}", script="x"
    )
    run = await wf_queries.insert_wf_run(
        conn,
        account_id="acc_root",
        workflow_id=wf.id,
        environment_id="env_root",
        script="x",
        host_semantics_epoch=HOST_SEMANTICS_EPOCH,
        script_sha="sha",
        depth=10,
    )
    if seq_journal:
        # A couple of journal rows to prove the cascade drops them.
        await conn.execute(
            "INSERT INTO wf_run_events (id, run_id, seq, type, payload) "
            "VALUES ($1, $2, 0, 'run_started', '{}'::jsonb)",
            f"{run.id}_ev0",
            run.id,
        )
        await conn.execute(
            "INSERT INTO wf_run_events (id, run_id, seq, type, payload) "
            "VALUES ($1, $2, 1, 'run_completed', '{}'::jsonb)",
            f"{run.id}_ev1",
            run.id,
        )
    await wf_queries.set_run_terminal(
        conn, run.id, status="completed", output=None, account_id="acc_root"
    )
    await wf_queries.archive_run(conn, run.id, account_id="acc_root")
    # Back-date archived_at so it is past the retention window.
    await conn.execute(
        "UPDATE wf_runs SET archived_at = now() - make_interval(days => $2) WHERE id = $1",
        run.id,
        archived_age_days,
    )
    return run.id


async def _insert_agent(
    conn: asyncpg.Connection[Any], *, agent_id: str, name: str, archived_days: int | None
) -> None:
    await conn.execute(
        "INSERT INTO agents (id, name, model, account_id) VALUES ($1, $2, 'm', 'acc_root')",
        agent_id,
        name,
    )
    await conn.execute(
        "INSERT INTO agent_versions (agent_id, version, model, account_id) "
        "VALUES ($1, 1, 'm', 'acc_root')",
        agent_id,
    )
    if archived_days is not None:
        await conn.execute(
            "UPDATE agents SET archived_at = now() - make_interval(days => $2) WHERE id = $1",
            agent_id,
            archived_days,
        )


async def _insert_session(
    conn: asyncpg.Connection[Any],
    *,
    session_id: str,
    agent_id: str,
    agent_version: int | None,
    archived: bool,
) -> None:
    await conn.execute(
        """
        INSERT INTO sessions
            (id, agent_id, environment_id, agent_version, workspace_volume_path,
             account_id, archived_at)
        VALUES ($1, $2, 'env_root', $3, '/w', 'acc_root',
                CASE WHEN $4 THEN now() ELSE NULL END)
        """,
        session_id,
        agent_id,
        agent_version,
        archived,
    )


async def _count(conn: asyncpg.Connection[Any], table: str, _id_col: str, _id: str) -> int:
    count = await conn.fetchval(f"SELECT count(*) FROM {table} WHERE {_id_col} = $1", _id)
    return int(count)


# ─── the headline acceptance test ───────────────────────────────────────────


async def test_prune_reclaims_ephemera_but_spares_sacred(
    conn: asyncpg.Connection[Any],
) -> None:
    """One sweep: an unreferenced terminal+archived run (and its journal) is
    removed, while a LIVE-PINNED agent version and a MEMORY STORE survive."""
    # Reclaimable: an aged terminal+archived run with a journal.
    run_id = await _make_archived_run(conn, archived_age_days=60)
    assert await _count(conn, "wf_run_events", "run_id", run_id) == 2

    # Sacred #1: a live session pinning agent version 1 of an ARCHIVED agent.
    # Even though the agent is archived and well past the window, the live pin
    # makes its version row sacred — replay-stability.
    await _insert_agent(conn, agent_id="ag_pinned", name="pinned", archived_days=90)
    await _insert_session(
        conn,
        session_id="ses_live",
        agent_id="ag_pinned",
        agent_version=1,
        archived=False,
    )

    # Sacred #2: a memory store. The prune must never touch memory content.
    await conn.execute(
        "INSERT INTO memory_stores (id, account_id, name) VALUES ('mem_1', 'acc_root', 'm')"
    )

    settings = get_settings()
    pruned_runs = await prune_archived_runs(conn, retention_days=settings.wf_runs_retention_days)
    pruned_agents = await prune_unpinned_archived_agents(
        conn, retention_days=settings.archived_definition_retention_days
    )

    # The unreferenced terminal+archived run is gone — and so is its journal
    # (dropped by the ON DELETE CASCADE, the unbounded-growth driver).
    assert pruned_runs == 1
    assert await _count(conn, "wf_runs", "id", run_id) == 0
    assert await _count(conn, "wf_run_events", "run_id", run_id) == 0

    # The live-pinned agent + its version survived despite being archived & old.
    assert pruned_agents == 0
    assert await _count(conn, "agents", "id", "ag_pinned") == 1
    assert (
        await conn.fetchval("SELECT count(*) FROM agent_versions WHERE agent_id = 'ag_pinned'") == 1
    )

    # The memory store is untouched.
    assert await _count(conn, "memory_stores", "id", "mem_1") == 1


# ─── runs ───────────────────────────────────────────────────────────────────


async def test_run_prune_respects_retention_window(conn: asyncpg.Connection[Any]) -> None:
    young = await _make_archived_run(conn, archived_age_days=5)
    old = await _make_archived_run(conn, archived_age_days=40)
    pruned = await prune_archived_runs(conn, retention_days=30)
    assert pruned == 1
    assert await _count(conn, "wf_runs", "id", young) == 1
    assert await _count(conn, "wf_runs", "id", old) == 0


async def test_run_prune_skips_non_archived_terminal_run(
    conn: asyncpg.Connection[Any],
) -> None:
    """A terminal but NOT-archived run is never a candidate (archive_run gates it)."""
    wf = await wf_queries.insert_workflow(conn, account_id="acc_root", name="wf_t", script="x")
    run = await wf_queries.insert_wf_run(
        conn,
        account_id="acc_root",
        workflow_id=wf.id,
        environment_id="env_root",
        script="x",
        host_semantics_epoch=HOST_SEMANTICS_EPOCH,
        script_sha="sha",
        depth=10,
    )
    await wf_queries.set_run_terminal(
        conn, run.id, status="completed", output=None, account_id="acc_root"
    )
    # Not archived → not a candidate, regardless of age.
    pruned = await prune_archived_runs(conn, retention_days=0)
    assert pruned == 0
    assert await _count(conn, "wf_runs", "id", run.id) == 1


async def test_run_prune_is_idempotent(conn: asyncpg.Connection[Any]) -> None:
    await _make_archived_run(conn, archived_age_days=60)
    first = await prune_archived_runs(conn, retention_days=30)
    second = await prune_archived_runs(conn, retention_days=30)
    assert first == 1
    assert second == 0


# ─── archived definitions: agents ───────────────────────────────────────────


async def test_agent_prune_spares_any_referenced_agent(
    conn: asyncpg.Connection[Any],
) -> None:
    """Live OR archived session reference holds the archived agent (sacred
    history + FK-safety)."""
    # Referenced by a LIVE session — held.
    await _insert_agent(conn, agent_id="ag_live", name="live", archived_days=90)
    await _insert_session(
        conn, session_id="s_live", agent_id="ag_live", agent_version=1, archived=False
    )
    # Referenced only by an ARCHIVED session (history may be memory) — held.
    await _insert_agent(conn, agent_id="ag_arch", name="arch", archived_days=90)
    await _insert_session(
        conn, session_id="s_arch", agent_id="ag_arch", agent_version=1, archived=True
    )
    # Referenced by NO session — reclaimable.
    await _insert_agent(conn, agent_id="ag_free", name="free", archived_days=90)

    pruned = await prune_unpinned_archived_agents(conn, retention_days=30)
    assert pruned == 1
    assert await _count(conn, "agents", "id", "ag_live") == 1
    assert await _count(conn, "agents", "id", "ag_arch") == 1
    assert await _count(conn, "agents", "id", "ag_free") == 0
    # The reclaimed agent's version history went with it.
    assert (
        await conn.fetchval("SELECT count(*) FROM agent_versions WHERE agent_id = 'ag_free'") == 0
    )


async def test_agent_prune_respects_window_and_archived_only(
    conn: asyncpg.Connection[Any],
) -> None:
    await _insert_agent(conn, agent_id="ag_live_def", name="livedef", archived_days=None)
    await _insert_agent(conn, agent_id="ag_young", name="young", archived_days=5)
    await _insert_agent(conn, agent_id="ag_old", name="old", archived_days=40)
    pruned = await prune_unpinned_archived_agents(conn, retention_days=30)
    assert pruned == 1
    assert await _count(conn, "agents", "id", "ag_live_def") == 1  # not archived
    assert await _count(conn, "agents", "id", "ag_young") == 1  # within window
    assert await _count(conn, "agents", "id", "ag_old") == 0


# ─── archived definitions: workflows ────────────────────────────────────────


async def test_workflow_prune_spares_referenced_workflow(
    conn: asyncpg.Connection[Any],
) -> None:
    # A workflow with a run referencing it — held (cascade-safety + replay pin).
    held = await wf_queries.insert_workflow(conn, account_id="acc_root", name="held", script="x")
    await wf_queries.insert_wf_run(
        conn,
        account_id="acc_root",
        workflow_id=held.id,
        environment_id="env_root",
        script="x",
        host_semantics_epoch=HOST_SEMANTICS_EPOCH,
        script_sha="sha",
        depth=10,
    )
    await conn.execute(
        "UPDATE workflows SET archived_at = now() - make_interval(days => 90) WHERE id = $1",
        held.id,
    )
    # A run-free archived workflow — reclaimable.
    free = await wf_queries.insert_workflow(conn, account_id="acc_root", name="free", script="x")
    await conn.execute(
        "UPDATE workflows SET archived_at = now() - make_interval(days => 90) WHERE id = $1",
        free.id,
    )

    pruned = await prune_unpinned_archived_workflows(conn, retention_days=30)
    assert pruned == 1
    assert await _count(conn, "workflows", "id", held.id) == 1
    assert await _count(conn, "workflows", "id", free.id) == 0


# ─── archived definitions: skills ───────────────────────────────────────────


async def test_skill_prune_spares_live_agent_binding(
    conn: asyncpg.Connection[Any],
) -> None:
    # Skill bound by a LIVE agent's skills JSONB — held.
    await conn.execute(
        "INSERT INTO skills (id, display_title, latest_version, account_id, archived_at) "
        "VALUES ('sk_bound', 'bound', 1, 'acc_root', now() - make_interval(days => 90))"
    )
    await conn.execute(
        "INSERT INTO skill_versions (skill_id, version, directory, name, files, account_id) "
        "VALUES ('sk_bound', 1, '/d', 'bound', '{}'::jsonb, 'acc_root')"
    )
    await conn.execute(
        "INSERT INTO agents (id, name, model, account_id, skills) "
        "VALUES ('ag_binder', 'binder', 'm', 'acc_root', "
        '\'[{"skill_id": "sk_bound", "version": 1}]\'::jsonb)'
    )
    # Unbound archived skill — reclaimable.
    await conn.execute(
        "INSERT INTO skills (id, display_title, latest_version, account_id, archived_at) "
        "VALUES ('sk_free', 'free', 1, 'acc_root', now() - make_interval(days => 90))"
    )
    await conn.execute(
        "INSERT INTO skill_versions (skill_id, version, directory, name, files, account_id) "
        "VALUES ('sk_free', 1, '/d', 'free', '{}'::jsonb, 'acc_root')"
    )

    pruned = await prune_unpinned_archived_skills(conn, retention_days=30)
    assert pruned == 1
    assert await _count(conn, "skills", "id", "sk_bound") == 1
    assert await _count(conn, "skills", "id", "sk_free") == 0
    assert (
        await conn.fetchval("SELECT count(*) FROM skill_versions WHERE skill_id = 'sk_free'") == 0
    )
    assert (
        await conn.fetchval("SELECT count(*) FROM skill_versions WHERE skill_id = 'sk_bound'") == 1
    )


# ─── the session_templates FK guard + per-family sweep isolation ─────────────


@pytest.fixture
async def pool(migrated_db_url: str, _reset_db_state: None) -> AsyncIterator[asyncpg.Pool[Any]]:
    """A pool over a DB seeded with ``acc_root`` + ``env_root`` (same scaffold as
    the ``conn`` fixture), for exercising the full ``sweep_reclaimable_ephemera``
    driver — which acquires its own connection(s) from a pool, not a bare conn."""
    p = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with p.acquire() as c:
            await c.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ('acc_root', NULL, TRUE, 'tenant-root')"
            )
            await c.execute(
                "INSERT INTO environments (id, name, config, account_id) "
                "VALUES ('env_root', 'env', '{}'::jsonb, 'acc_root')"
            )
        yield p
    finally:
        await p.close()


async def _insert_session_template(
    conn: asyncpg.Connection[Any], *, template_id: str, agent_id: str
) -> None:
    """Insert a minimal ``session_templates`` row pinning ``agent_id`` — the
    THIRD non-cascade FK to ``agents(id)`` (migration 0027, no ``ON DELETE``)."""
    await conn.execute(
        "INSERT INTO session_templates (id, name, agent_id, environment_id, account_id) "
        "VALUES ($1, $1, $2, 'env_root', 'acc_root')",
        template_id,
        agent_id,
    )


async def test_agent_prune_spares_template_pinned_agent(
    conn: asyncpg.Connection[Any],
) -> None:
    """A ``session_templates`` row pins its (archived, session-free, aged) agent.

    Without the ``session_templates`` guard the agents prune raises
    ``ForeignKeyViolationError`` on the non-cascade ``session_templates.agent_id``
    FK; with the guard the template-pinned agent is simply held. This is the
    unit-level proof of the predicate fix — it FAILS (raises) without the guard
    and PASSES with it.
    """
    # Archived, aged, NO session referencing it — but a live session_template
    # pins it. The bare ``sessions`` guard would mark it reclaimable and the
    # DELETE would FK-violate against session_templates.agent_id.
    await _insert_agent(conn, agent_id="ag_tmpl", name="tmpl", archived_days=90)
    await _insert_session_template(conn, template_id="st_pin", agent_id="ag_tmpl")
    # An unrelated archived+free agent so the prune still has real work to do.
    await _insert_agent(conn, agent_id="ag_free2", name="free2", archived_days=90)

    # No raise; the template-pinned agent survives, the free one is reclaimed.
    pruned = await prune_unpinned_archived_agents(conn, retention_days=30)
    assert pruned == 1
    assert await _count(conn, "agents", "id", "ag_tmpl") == 1
    assert (
        await conn.fetchval("SELECT count(*) FROM agent_versions WHERE agent_id = 'ag_tmpl'") == 1
    )
    assert await _count(conn, "agents", "id", "ag_free2") == 0


async def test_sweep_isolates_family_failure_and_spares_template_pinned_agent(
    pool: asyncpg.Pool[Any],
) -> None:
    """The headline regression for the silent-failure blocker.

    A template-pinned archived agent SURVIVES the full sweep (not deleted, no
    raise propagates out of ``sweep_reclaimable_ephemera``) AND the other
    families still prune in the SAME sweep. Before the fix the agents prune would
    raise ``ForeignKeyViolationError``, which — with no per-family try/except —
    skipped the workflows+skills prunes too and surfaced only as a generic
    ``tick_failed`` in the loop, silently no-op'ing the whole disk-reclaim
    mechanism. After the fix the agent is held by the predicate guard, so there
    is no raise at all; the per-family isolation is the belt to that suspenders
    (a future un-guarded reference would log + skip its own family, not the rest).
    """
    def_days = get_settings().archived_definition_retention_days

    async with pool.acquire() as conn:
        # The poison pill: an archived, session-free, aged agent pinned ONLY by a
        # session_template. Sits in the SECOND family of the sweep (agents), so a
        # raise here is exactly what would skip workflows+skills.
        await _insert_agent(conn, agent_id="ag_pinned", name="pinned", archived_days=90)
        await _insert_session_template(conn, template_id="st_pin", agent_id="ag_pinned")

        # Reclaimable work in EVERY OTHER family, all past the window, so we can
        # prove the sweep did not stop at the agents family:
        #   runs: an aged terminal+archived run (+ its journal).
        run_id = await _make_archived_run(conn, archived_age_days=def_days + 30)
        assert await _count(conn, "wf_run_events", "run_id", run_id) == 2
        #   workflows: a run-free aged archived workflow.
        free_wf = await wf_queries.insert_workflow(
            conn, account_id="acc_root", name="free_wf", script="x"
        )
        await conn.execute(
            "UPDATE workflows SET archived_at = now() - make_interval(days => $2) WHERE id = $1",
            free_wf.id,
            def_days + 30,
        )
        #   skills: an unbound aged archived skill (+ its version).
        await conn.execute(
            "INSERT INTO skills (id, display_title, latest_version, account_id, archived_at) "
            "VALUES ('sk_free', 'free', 1, 'acc_root', now() - make_interval(days => $1))",
            def_days + 30,
        )
        await conn.execute(
            "INSERT INTO skill_versions (skill_id, version, directory, name, files, account_id) "
            "VALUES ('sk_free', 1, '/d', 'free', '{}'::jsonb, 'acc_root')"
        )

    # The whole sweep — no exception escapes, and the tally proves every OTHER
    # family still pruned in the same pass.
    result = await sweep_reclaimable_ephemera(pool)
    assert result.runs == 1
    assert result.agents == 0  # the template-pinned agent is held, not deleted
    assert result.workflows == 1
    assert result.skills == 1

    async with pool.acquire() as conn:
        # The template-pinned agent (and its version) survived the sweep.
        assert await _count(conn, "agents", "id", "ag_pinned") == 1
        assert (
            await conn.fetchval("SELECT count(*) FROM agent_versions WHERE agent_id = 'ag_pinned'")
            == 1
        )
        # The session_template that pinned it is untouched (it was never a target).
        assert await _count(conn, "session_templates", "id", "st_pin") == 1
        # Every other family genuinely reclaimed its candidate.
        assert await _count(conn, "wf_runs", "id", run_id) == 0
        assert await _count(conn, "wf_run_events", "run_id", run_id) == 0
        assert await _count(conn, "workflows", "id", free_wf.id) == 0
        assert await _count(conn, "skills", "id", "sk_free") == 0


async def test_sweep_one_family_raise_does_not_disable_the_others(
    pool: asyncpg.Pool[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct coverage of the per-family ``try/except`` isolation.

    Force the AGENTS family (the second of four) to raise mid-sweep — the
    failure mode that, with no per-family isolation, would skip the workflows +
    skills prunes that follow and surface only as a generic ``tick_failed``. The
    sweep must instead: NOT propagate the raise, count that family 0, and still
    prune every OTHER family in the same pass.
    """
    def_days = get_settings().archived_definition_retention_days

    async with pool.acquire() as conn:
        run_id = await _make_archived_run(conn, archived_age_days=def_days + 30)
        free_wf = await wf_queries.insert_workflow(
            conn, account_id="acc_root", name="free_wf2", script="x"
        )
        await conn.execute(
            "UPDATE workflows SET archived_at = now() - make_interval(days => $2) WHERE id = $1",
            free_wf.id,
            def_days + 30,
        )
        await conn.execute(
            "INSERT INTO skills (id, display_title, latest_version, account_id, archived_at) "
            "VALUES ('sk_free2', 'free', 1, 'acc_root', now() - make_interval(days => $1))",
            def_days + 30,
        )
        await conn.execute(
            "INSERT INTO skill_versions (skill_id, version, directory, name, files, account_id) "
            "VALUES ('sk_free2', 1, '/d', 'free', '{}'::jsonb, 'acc_root')"
        )

    async def _boom(*_args: Any, **_kwargs: Any) -> int:
        raise RuntimeError("simulated agents-prune failure")

    monkeypatch.setattr(
        "aios.harness.reclaimable_prune.queries.prune_unpinned_archived_agents", _boom
    )

    # The raise is caught per-family — the sweep returns normally.
    result = await sweep_reclaimable_ephemera(pool)
    assert result.agents == 0  # the failed family is skipped this tick
    assert result.runs == 1  # the others still pruned
    assert result.workflows == 1
    assert result.skills == 1

    async with pool.acquire() as conn:
        assert await _count(conn, "wf_runs", "id", run_id) == 0
        assert await _count(conn, "workflows", "id", free_wf.id) == 0
        assert await _count(conn, "skills", "id", "sk_free2") == 0
