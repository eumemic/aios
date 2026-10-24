"""#2560: ``_unresolved_tool_calls``'s result-pairing probe must cost
O(page's tool_calls), never O(account's tool-result rows).

The former ``NOT EXISTS`` anti join was flattened by the planner into a
set-at-a-time join over EVERY tool-result row in the account — a Merge Anti
Join (sort + disk spill) in the CUSTOM plan asyncpg uses for a connection's
first five executions, a Hash Anti Join in the GENERIC plan.  The planner
flips between the two, so each plan-shape test below forces
``plan_cache_mode`` explicitly.

* ``test_plan_shape_*`` — the result probe is an ``Index Scan`` on
  ``events_tool_result_idx`` (parameterized per call), there is no
  Merge/Hash Anti Join, and no node scans ``events`` sequentially.  Both
  fail on the pre-#2560 ``NOT EXISTS`` SQL (``_OLD_SQL`` below) on this
  fixture.
* ``test_differential_old_vs_new`` — on a randomized 1,200-session fixture
  (random turn counts, multi-call turns, partial resolution, null/empty
  tool_calls, empty ids, random floors, a noisy second account), the new SQL
  returns exactly the old SQL's rows, in the same order, for random pages
  with and without ``max_age``.
"""

from __future__ import annotations

import json
import random
from typing import Any

import asyncpg
import pytest

from aios.db.queries.events import UNRESOLVED_TOOL_CALLS_SQL

pytestmark = pytest.mark.integration

_ACCOUNT = "acc_2560"
_OTHER = "acc_2560_other"
_N_SESSIONS = 1200

# Verbatim pre-#2560 statement — the differential oracle.
_OLD_SQL = """
        SELECT a.session_id,
               tc.item AS tool_call,
               a.created_at
          FROM sessions s
          JOIN events a
            ON a.session_id = s.id
           AND a.account_id = $2
           AND a.kind = 'message'
           AND a.role = 'assistant'
           AND a.data ? 'tool_calls'
           AND a.seq >= s.open_tool_call_floor_seq
         CROSS JOIN LATERAL jsonb_array_elements(
               COALESCE(NULLIF(a.data->'tool_calls','null'::jsonb), '[]'::jsonb)
             ) WITH ORDINALITY AS tc(item, ord)
         WHERE s.id = ANY($1::text[])
           AND s.account_id = $2
           AND (
                 $3::bigint IS NULL
                 OR a.created_at >= now() - make_interval(secs => $3::bigint)
               )
           AND COALESCE(tc.item->>'id', '') <> ''
           AND NOT EXISTS (
                 SELECT 1
                   FROM events r
                  WHERE r.session_id = a.session_id
                    AND r.account_id = $2
                    AND r.kind = 'message'
                    AND r.role = 'tool'
                    AND r.data->>'tool_call_id' = tc.item->>'id'
               )
         ORDER BY a.session_id, a.seq ASC, tc.ord ASC
"""


def _sid(account: str, i: int) -> str:
    return f"ses_{account}_{i:05d}"


async def _seed(conn: asyncpg.Connection[Any], rng: random.Random) -> None:
    for acct, parent in ((_ACCOUNT, None), (_OTHER, _ACCOUNT)):
        await conn.execute(
            "INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name) "
            "VALUES ($1, $2, TRUE, $1)",
            acct,
            parent,
        )
        await conn.execute(
            "INSERT INTO environments (id, name, account_id) VALUES ($1, 'env', $2)",
            f"env_{acct}",
            acct,
        )
        await conn.execute(
            "INSERT INTO agents (id, name, model, account_id) VALUES ($1, 'agent', 'm', $2)",
            f"agt_{acct}",
            acct,
        )
    sessions: list[tuple[Any, ...]] = []
    events: list[tuple[Any, ...]] = []
    for acct, n in ((_ACCOUNT, _N_SESSIONS), (_OTHER, 300)):
        for i in range(n):
            sid = _sid(acct, i)
            seq = 0
            open_seqs: list[int] = []
            for _turn in range(rng.randint(0, 40)):
                seq += 1
                shape = rng.random()
                if shape < 0.03:
                    tool_calls: Any = None
                elif shape < 0.06:
                    tool_calls = []
                else:
                    tool_calls = [
                        {
                            "id": "" if rng.random() < 0.02 else f"tc_{sid}_{seq}_{k}",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        }
                        for k in range(rng.randint(1, 3))
                    ]
                age_days = rng.choice((0, 0, 0, 3, 30))
                events.append(
                    (
                        f"ev_{sid}_{seq}",
                        sid,
                        seq,
                        json.dumps(
                            {"role": "assistant", "content": "", "tool_calls": tool_calls}
                        ),
                        "assistant",
                        acct,
                        age_days,
                    )
                )
                a_seq = seq
                for tc in tool_calls or []:
                    if tc["id"] and rng.random() < 0.85:
                        seq += 1
                        events.append(
                            (
                                f"ev_{sid}_{seq}",
                                sid,
                                seq,
                                json.dumps(
                                    {"role": "tool", "tool_call_id": tc["id"], "content": "ok"}
                                ),
                                "tool",
                                acct,
                                age_days,
                            )
                        )
                    elif tc["id"]:
                        open_seqs.append(a_seq)
            # Random floor: 0 (default), at the oldest open call, or (forged)
            # anywhere — the differential only needs both SQLs to agree.
            floor = rng.choice(
                (0, 0, min(open_seqs) if open_seqs else 0, rng.randint(0, max(seq, 1)))
            )
            sessions.append((sid, acct, f"env_{acct}", f"agt_{acct}", seq, floor))
    await conn.executemany(
        "INSERT INTO sessions (id, account_id, environment_id, agent_id, agent_version, "
        "workspace_volume_path, last_event_seq, open_tool_call_floor_seq) "
        "VALUES ($1, $2, $3, $4, 1, '/w', $5, $6)",
        sessions,
    )
    await conn.executemany(
        "INSERT INTO events (id, session_id, seq, kind, data, role, account_id, created_at) "
        "VALUES ($1, $2, $3, 'message', $4::jsonb, $5, $6, "
        "now() - make_interval(days => $7::int))",
        events,
    )
    await conn.execute("ANALYZE sessions")
    await conn.execute("ANALYZE events")


@pytest.fixture
async def seeded(migrated_db_url: str, _reset_db_state: None) -> asyncpg.Connection[Any]:
    conn = await asyncpg.connect(migrated_db_url)
    await _seed(conn, random.Random(2560))
    try:
        yield conn
    finally:
        await conn.close()


def _walk(plan: dict[str, Any]) -> list[dict[str, Any]]:
    out = [plan]
    for child in plan.get("Plans", []):
        out.extend(_walk(child))
    return out


async def _explain(
    conn: asyncpg.Connection[Any], sql: str, *, mode: str, session_ids: list[str]
) -> list[dict[str, Any]]:
    await conn.execute(f"SET plan_cache_mode = {mode}")
    await conn.execute("DEALLOCATE ALL")
    await conn.execute(f"PREPARE unresolved_2560(text[], text, bigint) AS {sql}")
    try:
        # Repeat so a generic plan, if chosen, is actually the one EXPLAINed.
        # EXECUTE takes no bind parameters, so inline the (test-controlled,
        # quote-free) literals.
        assert all("'" not in x and "," not in x for x in [*session_ids, _ACCOUNT])
        ids_lit = "{" + ",".join(session_ids) + "}"
        raw: Any = None
        for _ in range(6):
            raw = await conn.fetchval(
                "EXPLAIN (FORMAT JSON) EXECUTE "
                f"unresolved_2560('{ids_lit}'::text[], '{_ACCOUNT}', NULL)"
            )
        plan = json.loads(raw) if isinstance(raw, str) else raw
        return _walk(plan[0]["Plan"])
    finally:
        await conn.execute("DEALLOCATE unresolved_2560")
        await conn.execute("RESET plan_cache_mode")


def _assert_per_call_probe(nodes: list[dict[str, Any]], mode: str) -> None:
    shapes = [(n["Node Type"], n.get("Index Name"), n.get("Join Type")) for n in nodes]
    assert not any(
        n["Node Type"] in ("Merge Join", "Hash Join") and n.get("Join Type") == "Anti"
        for n in nodes
    ), f"[{mode}] set-at-a-time anti join over the account's tool rows: {shapes}"
    assert not any(
        n["Node Type"] == "Seq Scan" and n.get("Relation Name") == "events" for n in nodes
    ), f"[{mode}] sequential scan of events: {shapes}"
    probes = [
        n
        for n in nodes
        if n["Node Type"] in ("Index Scan", "Index Only Scan")
        and n.get("Index Name") == "events_tool_result_idx"
    ]
    assert probes, f"[{mode}] result probe does not use events_tool_result_idx: {shapes}"
    for p in probes:
        cond = p.get("Index Cond", "")
        assert "session_id" in cond and "tool_call_id" in cond, (
            f"[{mode}] events_tool_result_idx probe is not keyed per call: {cond!r}"
        )


_PAGE = [_sid(_ACCOUNT, i) for i in range(100)]


class TestUnresolvedToolCallsPlan:
    async def test_plan_shape_custom_plan(self, seeded: asyncpg.Connection[Any]) -> None:
        nodes = await _explain(
            seeded, UNRESOLVED_TOOL_CALLS_SQL, mode="force_custom_plan", session_ids=_PAGE
        )
        _assert_per_call_probe(nodes, "custom")

    async def test_plan_shape_generic_plan(self, seeded: asyncpg.Connection[Any]) -> None:
        nodes = await _explain(
            seeded, UNRESOLVED_TOOL_CALLS_SQL, mode="force_generic_plan", session_ids=_PAGE
        )
        _assert_per_call_probe(nodes, "generic")

    async def test_differential_old_vs_new(self, seeded: asyncpg.Connection[Any]) -> None:
        rng = random.Random(25600)
        all_ids = [_sid(_ACCOUNT, i) for i in range(_N_SESSIONS)]
        # Pages: whole account, the first page, random pages; foreign-account
        # ids mixed in must be ignored by both.
        pages: list[list[str]] = [all_ids, _PAGE]
        for _ in range(8):
            page = rng.sample(all_ids, rng.randint(1, 200))
            page += [_sid(_OTHER, rng.randrange(300)) for _ in range(3)]
            pages.append(page)
        total_rows = 0
        for page in pages:
            for max_age in (None, 7 * 86400):
                for mode in ("force_custom_plan", "force_generic_plan"):
                    await seeded.execute(f"SET plan_cache_mode = {mode}")
                    old = await seeded.fetch(_OLD_SQL, page, _ACCOUNT, max_age)
                    new = await seeded.fetch(UNRESOLVED_TOOL_CALLS_SQL, page, _ACCOUNT, max_age)
                    old_t = [(r["session_id"], r["tool_call"], r["created_at"]) for r in old]
                    new_t = [(r["session_id"], r["tool_call"], r["created_at"]) for r in new]
                    assert new_t == old_t, (
                        f"differential mismatch (mode={mode}, max_age={max_age}, "
                        f"page size={len(page)}): old={len(old_t)} new={len(new_t)}"
                    )
                    total_rows += len(new_t)
        await seeded.execute("RESET plan_cache_mode")
        # The fixture must actually exercise unresolved calls, or the
        # differential proves nothing.
        assert total_rows > 1000
