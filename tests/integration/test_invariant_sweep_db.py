"""Real-Postgres coverage for the C5 edge-owned-lifetime invariant sweep (aios#1824).

The unit suite (``tests/unit/test_invariant_sweep.py``) only asserts SQL substrings over
mocked rows — it cannot validate the joins, anti-joins, creation-edge pinning, or the
grace/ceiling timestamps that ARE the collector. These tests construct the actual
``sessions`` / ``events`` / ``wf_runs`` rows and assert which sessions the four-clause
query selects — one test per spec-§6 case (f/g/n/r/s/t/u/v) plus the grandchild-ceiling
variant — and drive the armed sweep to assert the read/actuation TOCTOU re-check
(``_revalidate_and_archive``) never force-archives a session that gained a live awaited
inbound edge between selection and actuation.

Clause reference (aios#1152 §2 C5):
  * ``revoked_lease``       — terminal parent, awaited creation edge still open / revoked,
                              no OTHER live-caller inbound (api = unconditionally LIVE).
  * ``expired_idle``        — archive_when_idle, idle, aged, zero open awaited edges.
  * ``abandoned_api_lease`` — LOG-ONLY: open api edge older than its deadline.
  * ``lease_ceiling``       — archive_when_idle, past ceiling, zero open awaited edges,
                              ACTIVITY-BLIND (the only collector for the churning zombie).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest

from aios.config import get_settings
from aios.db import queries
from aios.db.pool import create_pool
from aios.harness.invariant_sweep import (
    INVARIANT_CANDIDATES_SQL,
    _candidate_query_args,
    sweep_session_invariants,
)
from aios.models.sessions import Ok, Outcome
from aios.services import sessions as service

# ``run_tools`` first breaks the ``workflows.service`` <-> ``services.workflows`` import
# cycle (mirrors ``test_cancel_cascade_seeding``); importing it makes the module present
# for the autouse ``defer_run_wake`` patch below.
from aios.workflows import run_tools as _run_tools  # noqa: F401
from aios.workflows import service as wf_service
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.integration

_ACCOUNT = "acc_c5_sweep"


@pytest.fixture(autouse=True)
def _mock_wakes() -> Iterator[None]:
    """Archive + create_run fire procrastinate wakes; no open App in this tier, so patch
    the deferral primitives where each caller looked them up at module load."""
    with (
        patch("aios.services.sessions.defer_wake", new=AsyncMock()),
        patch("aios.services.sessions.defer_run_wake", new=AsyncMock()),
        patch("aios.workflows.service.defer_run_wake", new=AsyncMock()),
    ):
        yield


class _Fixture:
    """Seeded tenant + a shared (agent, env) the tests hang sessions off of."""

    def __init__(self, pool: asyncpg.Pool[Any], agent_id: str, env_id: str) -> None:
        self.pool = pool
        self.agent_id = agent_id
        self.env_id = env_id
        self._n = 0

    def _next(self, tag: str) -> str:
        self._n += 1
        return f"{tag}{self._n}"

    async def session(
        self,
        *,
        tag: str = "s",
        archive_when_idle: bool = False,
        parent_run_id: str | None = None,
        age_secs: float = 0.0,
        active: bool = False,
    ) -> str:
        """Insert a session, then backdate ``created_at`` and set lease/activity state.

        ``active`` raises ``last_stimulus_seq`` above ``last_reacted_seq`` so
        ``session_active_predicate`` is TRUE (the churning-servicer shape); default is the
        idle shape (both 0)."""
        async with self.pool.acquire() as conn:
            row = await queries.insert_session(
                conn,
                account_id=_ACCOUNT,
                agent_id=self.agent_id,
                environment_id=self.env_id,
                agent_version=1,
                title=None,
                metadata={},
                archive_when_idle=archive_when_idle,
            )
            await conn.execute(
                "UPDATE sessions SET created_at = now() - make_interval(secs => $2), "
                "parent_run_id = $3, "
                "last_stimulus_seq = $4, last_reacted_seq = 0 WHERE id = $1",
                row.id,
                age_secs,
                parent_run_id,
                5 if active else 0,
            )
        return row.id

    async def open_edge(
        self,
        session_id: str,
        *,
        request_id: str,
        caller: dict[str, Any],
        awaited: bool = True,
        age_secs: float = 0.0,
        deadline_seconds: str | None = None,
    ) -> None:
        """Open a ``request_opened`` edge, backdate it, and optionally stamp a per-request
        ``deadline_seconds`` override onto the frame (the abandoned-api clause)."""
        async with self.pool.acquire() as conn:
            await queries.append_request_opened(
                conn,
                session_id=session_id,
                account_id=_ACCOUNT,
                request_id=request_id,
                caller=caller,
                depth=0,
                environment_id=self.env_id,
                frozen_surface={"tools": [], "mcp_servers": [], "http_servers": []},
                vault_ids=[],
                awaited=awaited,
            )
            if deadline_seconds is not None:
                await conn.execute(
                    "UPDATE events SET data = jsonb_set(data, '{deadline_seconds}', to_jsonb($3::text)) "
                    "WHERE session_id = $1 AND kind = 'lifecycle' "
                    "AND data->>'event' = 'request_opened' AND data->>'request_id' = $2",
                    session_id,
                    request_id,
                    deadline_seconds,
                )
            await conn.execute(
                "UPDATE events SET created_at = now() - make_interval(secs => $3) "
                "WHERE session_id = $1 AND kind = 'lifecycle' "
                "AND data->>'event' = 'request_opened' AND data->>'request_id' = $2",
                session_id,
                request_id,
                age_secs,
            )

    async def answer_edge(
        self, session_id: str, *, request_id: str, outcome: Outcome, age_secs: float = 0.0
    ) -> None:
        """Close an edge with a ``request_response`` (fulfilled Ok / revoked Err) and
        backdate it — its ``created_at`` becomes the session's ``last_closure``."""
        async with self.pool.acquire() as conn:
            await queries.write_response_if_absent(
                conn, session_id, account_id=_ACCOUNT, request_id=request_id, outcome=outcome
            )
            await conn.execute(
                "UPDATE events SET created_at = now() - make_interval(secs => $3) "
                "WHERE session_id = $1 AND kind = 'lifecycle' "
                "AND data->>'event' = 'request_response' AND data->>'request_id' = $2",
                session_id,
                request_id,
                age_secs,
            )

    async def terminal_run(self, *, status: str = "completed") -> str:
        """A wf_run pinned to a terminal status (the revoked-lease parent)."""
        from aios.db.queries import workflows as wf_queries

        async with self.pool.acquire() as conn:
            wf = await wf_queries.insert_workflow(
                conn,
                account_id=_ACCOUNT,
                name=self._next("wf"),
                script="async def main(i):\n return 1\n",
            )
        run = await wf_service.create_run(
            pool=self.pool,
            account_id=_ACCOUNT,
            workflow_id=wf.id,
            environment_id=self.env_id,
            input=None,
        )
        async with self.pool.acquire() as conn:
            await conn.execute("UPDATE wf_runs SET status = $2 WHERE id = $1", run.id, status)
        return run.id

    async def selected(self) -> dict[str, str]:
        """The batch selection as ``{session_id: clause}`` — the read half, no actuation."""
        rows = await self.pool.fetch(
            INVARIANT_CANDIDATES_SQL,
            *_candidate_query_args(get_settings(), batch_size=200, session_id=None),
        )
        return {r["session_id"]: r["clause"] for r in rows}

    async def archived_at(self, session_id: str) -> Any:
        return await self.pool.fetchval(
            "SELECT archived_at FROM sessions WHERE id = $1", session_id
        )


@pytest.fixture
async def fx(migrated_db_url: str, _reset_db_state: None) -> AsyncIterator[_Fixture]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=6)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
                "VALUES ($1, NULL, TRUE, 'c5-sweep')",
                _ACCOUNT,
            )
        agent, env, _throwaway = await seed_agent_env_session(
            pool, account_id=_ACCOUNT, prefix="c5"
        )
        yield _Fixture(pool, agent.id, env.id)
    finally:
        await pool.close()


def _run_caller(run_id: str) -> dict[str, Any]:
    return {"kind": "run", "id": run_id}


def _api_caller() -> dict[str, Any]:
    return {"kind": "api", "id": _ACCOUNT}


def _session_caller(session_id: str) -> dict[str, Any]:
    return {"kind": "session", "id": session_id}


# ── (f) pre-existing orphan cohort archived on first armed pass, grace respected ──────
async def test_f_orphan_cohort_archived_on_first_armed_pass_grace_respected(fx: _Fixture) -> None:
    parent = await fx.terminal_run(status="completed")
    # Two aged revoked-lease orphans (creation edge open+awaited from the terminal parent).
    orphan_a = await fx.session(tag="orphan", parent_run_id=parent, age_secs=1200)
    orphan_b = await fx.session(tag="orphan", parent_run_id=parent, age_secs=1200)
    await fx.open_edge(orphan_a, request_id="ra", caller=_run_caller(parent))
    await fx.open_edge(orphan_b, request_id="rb", caller=_run_caller(parent))
    # A newborn (within the 10-min grace) with the same shape must be SPARED.
    newborn = await fx.session(tag="orphan", parent_run_id=parent, age_secs=60)
    await fx.open_edge(newborn, request_id="rn", caller=_run_caller(parent))

    selected = await fx.selected()
    assert selected.get(orphan_a) == "revoked_lease"
    assert selected.get(orphan_b) == "revoked_lease"
    assert newborn not in selected  # newborn guard: created_at within grace

    result = await sweep_session_invariants(fx.pool, dry_run=False)
    assert result.archived == 2
    assert await fx.archived_at(orphan_a) is not None
    assert await fx.archived_at(orphan_b) is not None
    assert await fx.archived_at(newborn) is None  # grace respected through actuation


# ── (g) Tell child + terminal parent ⇒ NOT reclaimed by the revoked-lease clause ──────
async def test_g_tell_child_of_terminal_parent_not_revoked_lease(fx: _Fixture) -> None:
    parent = await fx.terminal_run(status="completed")
    # A Tell child: its creation edge is UNAWAITED — the awaited bit IS the ownership bit,
    # so the forceful revoked-lease reclaim must not touch it. Kept under ceiling + active
    # so the graceful idle/ceiling collectors don't fire either, isolating clause 1.
    tell_child = await fx.session(
        tag="tell", parent_run_id=parent, age_secs=1200, active=True, archive_when_idle=True
    )
    await fx.open_edge(tell_child, request_id="tell", caller=_run_caller(parent), awaited=False)

    selected = await fx.selected()
    assert tell_child not in selected


# ── (n) abandoned api servicer selected by clause 3 (logged, not archived); the
#        per-request decimal deadline (finding #3) is honored ───────────────────────────
async def test_n_abandoned_api_lease_selected_but_log_only(fx: _Fixture) -> None:
    # Fractional per-request deadline (1.5s) — the finding-#3 regressor. Edge aged 5s.
    frac = await fx.session(tag="api", archive_when_idle=True, age_secs=1200)
    await fx.open_edge(
        frac, request_id="af", caller=_api_caller(), age_secs=5, deadline_seconds="1.5"
    )
    # Integer per-request deadline (2s) — the optional branch.
    integer = await fx.session(tag="api", archive_when_idle=True, age_secs=1200)
    await fx.open_edge(
        integer, request_id="ai", caller=_api_caller(), age_secs=5, deadline_seconds="2"
    )
    # No per-request override, only 5s old ⇒ falls back to the 6h global ⇒ NOT abandoned.
    fresh = await fx.session(tag="api", archive_when_idle=True, age_secs=1200)
    await fx.open_edge(fresh, request_id="ag", caller=_api_caller(), age_secs=5)

    selected = await fx.selected()
    assert selected.get(frac) == "abandoned_api_lease"  # 1.5s decimal override honored
    assert selected.get(integer) == "abandoned_api_lease"  # integer override honored
    assert fresh not in selected  # global fallback ⇒ 5s edge is not abandoned

    # LOG-ONLY: an armed pass selects but never archives the api clause.
    result = await sweep_session_invariants(fx.pool, dry_run=False)
    assert await fx.archived_at(frac) is None
    assert await fx.archived_at(integer) is None
    assert result.archived == 0


# ── (r) answered-then-churning servicer reaped by clause 4 at ceiling, + grandchild ────
async def test_r_churning_zombie_reaped_by_lease_ceiling_and_grandchild(fx: _Fixture) -> None:
    ceiling = get_settings().lease_ceiling_seconds
    over = ceiling + 600  # comfortably past the ceiling

    # A servicer that ANSWERED its inbound long ago but keeps churning (active): idle/
    # revoked can't touch it; only the activity-blind ceiling backstop can.
    zombie = await fx.session(tag="zombie", archive_when_idle=True, age_secs=over, active=True)
    await fx.open_edge(zombie, request_id="z", caller=_api_caller(), age_secs=over)
    await fx.answer_edge(zombie, request_id="z", outcome=Ok(result="done"), age_secs=over)

    # Grandchild variant: root → child → grandchild via session-caller edges; the
    # grandchild is the churning-past-ceiling zombie.
    root = await fx.session(tag="gc-root")
    child = await fx.session(tag="gc-child", age_secs=over)
    await fx.open_edge(child, request_id="c-edge", caller=_session_caller(root))
    grandchild = await fx.session(tag="gc-leaf", archive_when_idle=True, age_secs=over, active=True)
    await fx.open_edge(
        grandchild, request_id="g-edge", caller=_session_caller(child), age_secs=over
    )
    await fx.answer_edge(grandchild, request_id="g-edge", outcome=Ok(result="ok"), age_secs=over)

    selected = await fx.selected()
    assert selected.get(zombie) == "lease_ceiling"
    assert selected.get(grandchild) == "lease_ceiling"

    result = await sweep_session_invariants(fx.pool, dry_run=False)
    assert await fx.archived_at(zombie) is not None
    assert await fx.archived_at(grandchild) is not None
    assert result.archived >= 2


# ── (s) fulfilled-active child of completed run ⇒ NOT clause 1; idles → clause 2 ───────
async def test_s_fulfilled_active_child_then_idles_to_expired(fx: _Fixture) -> None:
    parent = await fx.terminal_run(status="completed")
    child = await fx.session(
        tag="ful", parent_run_id=parent, age_secs=1200, active=True, archive_when_idle=True
    )
    await fx.open_edge(child, request_id="cr", caller=_run_caller(parent))
    # Servicer answered its creation edge Ok (fulfilled) — routes AWAY from clause 1.
    await fx.answer_edge(child, request_id="cr", outcome=Ok(result="done"), age_secs=1200)

    # Phase 1: still ACTIVE ⇒ neither revoked-lease (fulfilled) nor expired-idle (active).
    assert child not in await fx.selected()

    # Phase 2: goes idle ⇒ the graceful expired-idle clause claims it (not clause 1).
    async with fx.pool.acquire() as conn:
        await conn.execute(
            "UPDATE sessions SET last_stimulus_seq = 0, last_reacted_seq = 0 WHERE id = $1", child
        )
    assert (await fx.selected()).get(child) == "expired_idle"


# ── (t) multi-edge session (Tell-created + later Ask) ⇒ clause 1 reads ONLY the
#        caller-pinned creation edge, never the later Ask ────────────────────────────────
async def test_t_multi_edge_creation_pin_ignores_later_ask(fx: _Fixture) -> None:
    parent = await fx.terminal_run(status="completed")
    # archive_when_idle=False isolates clause 1 (only the revoked-lease clause can apply).
    sess = await fx.session(tag="multi", parent_run_id=parent, age_secs=1200, active=True)
    # Earliest caller={run,parent} edge is a Tell (unawaited) — the CREATION edge.
    await fx.open_edge(sess, request_id="tell0", caller=_run_caller(parent), awaited=False)
    # A LATER awaited Ask, also from the same parent run — must NOT reclassify the Tell.
    await fx.open_edge(sess, request_id="ask1", caller=_run_caller(parent), awaited=True)

    # The creation lateral pins to the earliest run-caller edge (the Tell, awaited=false),
    # so revoked-lease's "creation awaited" gate is false ⇒ NOT selected.
    assert sess not in await fx.selected()


# ── (u) run-parented child with an open api ask + terminal parent ⇒ NOT clause 1
#        (api caller counts as unconditionally LIVE) ──────────────────────────────────────
async def test_u_open_api_ask_protects_from_revoked_lease(fx: _Fixture) -> None:
    parent = await fx.terminal_run(status="completed")
    sess = await fx.session(tag="apiask", parent_run_id=parent, age_secs=1200, active=True)
    # Creation edge open + awaited from the terminal parent (would trip revoked-lease)…
    await fx.open_edge(sess, request_id="creat", caller=_run_caller(parent), awaited=True)
    # …but a live api caller holds an open awaited edge ⇒ the live-caller guard spares it.
    await fx.open_edge(sess, request_id="apiedge", caller=_api_caller(), awaited=True, age_secs=5)

    assert sess not in await fx.selected()


# ── (v) the sweep never force-archives a fulfilled-edge ACTIVE session before ceiling ───
async def test_v_fulfilled_active_before_ceiling_never_archived(fx: _Fixture) -> None:
    parent = await fx.terminal_run(status="completed")
    # Fulfilled creation edge, active, aged past grace but last_closure well UNDER ceiling.
    sess = await fx.session(
        tag="preceil", parent_run_id=parent, age_secs=1200, active=True, archive_when_idle=True
    )
    await fx.open_edge(sess, request_id="fc", caller=_run_caller(parent))
    await fx.answer_edge(sess, request_id="fc", outcome=Ok(result="ok"), age_secs=30)  # recent

    assert sess not in await fx.selected()
    result = await sweep_session_invariants(fx.pool, dry_run=False)
    assert await fx.archived_at(sess) is None
    assert result.archived == 0


# ── finding #1: read/actuation TOCTOU — a concurrent awaited-edge open between selection
#    and archival is honored (fails-before via the raw path / passes-after via the fix) ──
async def test_toctou_concurrent_edge_open_is_revalidated(fx: _Fixture) -> None:
    settings = get_settings()

    # A clean expired-idle candidate that the batch selection DOES pick (T0 state).
    safe = await fx.session(tag="race", archive_when_idle=True, age_secs=1200)
    assert (await fx.selected()).get(safe) == "expired_idle"

    # T2: a live awaited inbound edge races in AFTER selection, BEFORE actuation.
    await fx.open_edge(safe, request_id="raced", caller=_api_caller(), awaited=True, age_secs=1)

    # PASSES-AFTER: the under-lock re-check sees the new edge and SKIPS the archive.
    from aios.harness.invariant_sweep import _revalidate_and_archive

    assert await _revalidate_and_archive(fx.pool, safe, _ACCOUNT, settings=settings) is False
    assert await fx.archived_at(safe) is None  # live caller's request preserved

    # FAILS-BEFORE contrast: the raw unconditional archive path the pre-fix sweep used
    # archives the SAME shape regardless of the fresh edge — breaking the live caller.
    unsafe = await fx.session(tag="race", archive_when_idle=True, age_secs=1200)
    assert (await fx.selected()).get(unsafe) == "expired_idle"
    await fx.open_edge(unsafe, request_id="raced2", caller=_api_caller(), awaited=True, age_secs=1)
    await service.archive_session(fx.pool, unsafe, account_id=_ACCOUNT, idempotent=True)
    assert await fx.archived_at(unsafe) is not None  # the bug the re-check closes


# ── a still-qualifying candidate IS archived through the re-validation seam (no false
#    skip: the TOCTOU gate must not starve the happy path) ────────────────────────────────
async def test_still_qualifying_candidate_is_archived_through_revalidation(fx: _Fixture) -> None:
    settings = get_settings()
    idle = await fx.session(tag="ok", archive_when_idle=True, age_secs=1200)
    assert (await fx.selected()).get(idle) == "expired_idle"

    from aios.harness.invariant_sweep import _revalidate_and_archive

    assert await _revalidate_and_archive(fx.pool, idle, _ACCOUNT, settings=settings) is True
    assert await fx.archived_at(idle) is not None
