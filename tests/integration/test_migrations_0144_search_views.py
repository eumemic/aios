"""Migration 0144 acceptance tests: the session self-read search views (#1844).

Covers the deterministic acceptance set from the issue spec, against a real
Postgres at ``alembic upgrade head`` with seeded event data:

* **denylist (redaction)** — no allowlisted view exposes a denylisted column
  (raw ``data`` JSONB, the 0022 cost/token fence, lifecycle internals), and
  seeded sentinel VALUES for every redacted payload key never appear in any
  view's output (execution-based, not schema-only).
* **seeded span derivation** — ``spans_search`` derives non-NULL
  ``span_kind`` / ``tool_call_id`` / ``tool_name`` / ``start_event_id`` /
  ``is_error`` / ``focal_channel`` from realistic writer-shaped span payloads
  (the writers link ``*_end`` rows via ``<prefix>_start_id`` keys — an
  always-NULL derivation fails here), and the documented duration self-join
  pairs end rows to their start rows.
* **help completeness** — every column of every allowlisted relation,
  ``search_views_help`` INCLUDED (the view describes itself), has exactly one
  help row; no help row describes a column that does not exist (set equality,
  both directions — this is the allowlist/help drift guard's DB-backed half).
* **EXPLAIN / index-plan** — the view predicates are served by the intended
  partial indexes (``events_assistant_tool_calls_idx`` /
  ``events_tool_result_idx`` / ``events_session_span_seq_idx`` /
  ``events_session_lifecycle_seq_idx``). Determinism: inside a rolled-back
  transaction every OTHER index on ``events`` is dropped and seqscans are
  disabled, so the plan can only use the asserted index if — and only if —
  its partial predicate actually matches the view's WHERE clause.
* **executable examples** — every example query in the tool description, the
  parameters schema, and the ``search_views_help`` idiom rows executes
  through the real ``search_events_handler`` and returns rows (the fixture is
  deliberately seeded to satisfy each documented example; adding an example
  without fixture support fails loud here).
* **carve-in scoping** — ``search_views_help`` is the documented static
  carve-in: it reads zero tables (``information_schema.view_table_usage``),
  returns zero rows when the ``app.session_id`` GUC is unset (fail-closed),
  and returns byte-identical rows for two different sessions (it carries no
  tenant data to scope).
* **session scoping** — the three data views never leak another session's
  rows (the same invariant ``events_search`` enforces).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Iterator
from typing import Any

import asyncpg
import pytest

from aios.tools.search_events import _ALLOWED_RELATIONS
from tests.conftest import _docker_available, needs_docker
from tests.integration.test_migrations import _alembic_url, _run_alembic

# ---------------------------------------------------------------------------
# Seed: two tenants, two sessions. Session X carries the full menagerie —
# messages (shaped to satisfy every documented example), multi-call assistant
# events, paired tool results (success + failure), writer-shaped span pairs
# (tool_execute and model_request, the latter loaded with cost sentinels),
# and lifecycle events (allowlisted kinds, one non-allowlisted kind, and
# redacted-key sentinels). Session Y carries markers that must never appear
# in session X's reads.
# ---------------------------------------------------------------------------

_SEED_SQL = """
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_root', NULL, TRUE, 'root'),
       ('acc_a', 'acc_root', FALSE, 'tenant-a'),
       ('acc_b', 'acc_root', FALSE, 'tenant-b');

INSERT INTO agents (id, account_id, name, version, model, system, window_min, window_max)
VALUES ('agent_a', 'acc_a', 'a', 1, 'openrouter/test', '', 50000, 150000),
       ('agent_b', 'acc_b', 'b', 1, 'openrouter/test', '', 50000, 150000);

INSERT INTO environments (id, name, config, account_id)
VALUES ('env_a', 'env-a', '{}'::jsonb, 'acc_a'),
       ('env_b', 'env-b', '{}'::jsonb, 'acc_b');

INSERT INTO sessions
    (id, account_id, agent_id, environment_id, agent_version, workspace_volume_path)
VALUES ('sess_x', 'acc_a', 'agent_a', 'env_a', 1, '/tmp/sess_x'),
       ('sess_y', 'acc_b', 'agent_b', 'env_b', 1, '/tmp/sess_y');
"""

# Redaction sentinels: seeded into payload keys the views must NOT project.
# Each is a distinctive string that would be plainly visible in any leaked
# rendering of the payload.
_SENTINEL_MODEL = "sentinel-redacted-model-zzz"
_SENTINEL_USAGE = "sentinel_usage_marker_zzz"
_SENTINEL_TOKENS = 271828182845
_SENTINEL_COST = 314159.2653
_SENTINEL_FROZEN = "sentinel_frozen_surface_zzz"
_SENTINEL_SCHEMA = "sentinel_output_schema_zzz"
_SENTINEL_VAULT = "vlt_sentinel_zzz"
_SENTINEL_ENV = "env_sentinel_zzz"
_SENTINEL_CUMULATIVE_TOKENS = 161803398874
_SENTINEL_CUMULATIVE_CLASS_MASS = "sentinel_cumulative_class_mass_zzz"
# Positive control: a NON-redacted lifecycle payload key that MUST survive
# into detail_text (proves the leak assertions are not vacuously passing
# against an empty rendering).
_VISIBLE_DETAIL = "visible_detail_marker_zzz"

_ARGS_SMALL = '{"cmd": "ls"}'
_ARGS_READ = '{"path": "/tmp/x"}'
# > 16 KiB so the arguments_text cell cap + `…[truncated]` marker engage.
_ARGS_BIG = '{"cmd": "' + "a" * 20000 + '"}'


def _call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


# (id, session, seq, kind, data, role, channel, tool_name, is_error,
#  sender_name, focal_channel_at_arrival) — promoted columns stamped the way
# the real writer (queries.append_event) stamps them.
_EVENTS: list[
    tuple[
        str,
        str,
        int,
        str,
        dict[str, Any],
        str | None,
        str | None,
        str | None,
        bool | None,
        str | None,
        str | None,
    ]
] = [
    # Messages shaped to satisfy every documented example query.
    (
        "evt_x_1",
        "sess_x",
        1,
        "message",
        {"role": "user", "content": "kick off the docker deploy please"},
        "user",
        "telegram:chat/12345",
        None,
        None,
        "Matt",
        None,
    ),
    (
        "evt_x_2",
        "sess_x",
        2,
        "message",
        {"role": "user", "content": "hello from slack"},
        "user",
        "slack:C0123ABCD",
        None,
        None,
        "alice",
        None,
    ),
    # Multi-call assistant event: sibling calls share one seq — the reason the
    # pagination cursor is (seq, call_ordinal).
    (
        "evt_x_3",
        "sess_x",
        3,
        "message",
        {
            "role": "assistant",
            "content": "INTERNAL_MONOLOGUE: run it",
            "tool_calls": [
                _call("call_1", "bash", _ARGS_SMALL),
                _call("call_2", "read", _ARGS_READ),
            ],
        },
        "assistant",
        "telegram:chat/12345",
        "bash",
        None,
        None,
        "telegram:chat/12345",
    ),
    (
        "evt_x_4",
        "sess_x",
        4,
        "message",
        {"role": "tool", "tool_call_id": "call_1", "name": "bash", "content": "ok files listed"},
        "tool",
        "telegram:chat/12345",
        "bash",
        None,
        None,
        None,
    ),
    (
        "evt_x_5",
        "sess_x",
        5,
        "message",
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "name": "read",
            "content": "read failed: no such file",
        },
        "tool",
        "telegram:chat/12345",
        "read",
        True,
        None,
        None,
    ),
    # Single-call assistant event with oversized arguments (cap + marker).
    (
        "evt_x_6",
        "sess_x",
        6,
        "message",
        {"role": "assistant", "content": "", "tool_calls": [_call("call_3", "bash", _ARGS_BIG)]},
        "assistant",
        "telegram:chat/12345",
        "bash",
        None,
        None,
        "telegram:chat/12345",
    ),
    (
        "evt_x_7",
        "sess_x",
        7,
        "message",
        {"role": "tool", "tool_call_id": "call_3", "name": "bash", "content": "done"},
        "tool",
        "telegram:chat/12345",
        "bash",
        None,
        None,
        None,
    ),
    # Span pair, writer-shaped (tool_dispatch.py): the end row links via
    # ``tool_execute_start_id`` — NOT a literal ``start_event_id`` key.
    (
        "evt_span_start_1",
        "sess_x",
        8,
        "span",
        {"event": "tool_execute_start", "tool_call_id": "call_1", "tool_name": "bash"},
        None,
        None,
        None,
        None,
        None,
        "telegram:chat/12345",
    ),
    (
        "evt_span_end_1",
        "sess_x",
        9,
        "span",
        {
            "event": "tool_execute_end",
            "tool_execute_start_id": "evt_span_start_1",
            "tool_call_id": "call_1",
            "tool_name": "bash",
            "is_error": False,
        },
        None,
        None,
        None,
        None,
        None,
        "telegram:chat/12345",
    ),
    # Second span pair, different linking key (loop.py model_request writer),
    # loaded with every cost-fence sentinel the view must redact.
    (
        "evt_mrs_1",
        "sess_x",
        10,
        "span",
        {"event": "model_request_start"},
        None,
        None,
        None,
        None,
        None,
        "telegram:chat/12345",
    ),
    (
        "evt_mre_1",
        "sess_x",
        11,
        "span",
        {
            "event": "model_request_end",
            "model_request_start_id": "evt_mrs_1",
            "is_error": False,
            "model": _SENTINEL_MODEL,
            "model_usage": {"marker": _SENTINEL_USAGE, "input_tokens": _SENTINEL_TOKENS},
            "local_tokens": _SENTINEL_TOKENS,
            "local_tokens_by_class": {"marker": _SENTINEL_USAGE},
            "cost_usd": _SENTINEL_COST,
        },
        None,
        None,
        None,
        None,
        None,
        "telegram:chat/12345",
    ),
    # Lifecycle events: the four seeded allowlisted kinds, redaction
    # sentinels on request_opened, plus one NON-allowlisted kind that must be
    # invisible (kind-allowlist is fail-closed).
    (
        "evt_lc_1",
        "sess_x",
        12,
        "lifecycle",
        {
            "event": "request_opened",
            "request_id": "req_1",
            "summary": "do the thing",
            "awaited": True,
            "caller": {"kind": "session"},
            "frozen_surface": _SENTINEL_FROZEN,
            "output_schema": {"marker": _SENTINEL_SCHEMA},
            "vault_ids": [_SENTINEL_VAULT],
            "environment_id": _SENTINEL_ENV,
            "cumulative_tokens": _SENTINEL_CUMULATIVE_TOKENS,
            "cumulative_class_mass": {"marker": _SENTINEL_CUMULATIVE_CLASS_MASS},
            "custom_note": _VISIBLE_DETAIL,
        },
        None,
        None,
        None,
        None,
        None,
        None,
    ),
    (
        "evt_lc_2",
        "sess_x",
        13,
        "lifecycle",
        {"event": "request_response", "request_id": "req_1", "is_error": False},
        None,
        None,
        None,
        None,
        None,
        None,
    ),
    (
        "evt_lc_3",
        "sess_x",
        14,
        "lifecycle",
        {"event": "turn_ended", "status": "completed"},
        None,
        None,
        None,
        None,
        None,
        None,
    ),
    (
        "evt_lc_4",
        "sess_x",
        15,
        "lifecycle",
        {"event": "tool_confirmed", "request_id": "req_conf"},
        None,
        None,
        None,
        None,
        None,
        None,
    ),
    (
        "evt_lc_5",
        "sess_x",
        16,
        "lifecycle",
        {"event": "wake_deferred", "cause": "message"},
        None,
        None,
        None,
        None,
        None,
        None,
    ),
    # Tenant Y: one row of each shape, all carrying markers that must never
    # surface in session X's reads.
    (
        "evt_y_1",
        "sess_y",
        1,
        "message",
        {
            "role": "assistant",
            "content": "tenant_y_content_marker",
            "tool_calls": [_call("call_y1", "bash", '{"secret": "tenant_y_args_marker"}')],
        },
        "assistant",
        "signal:group/999",
        "bash",
        None,
        None,
        "signal:group/999",
    ),
    (
        "evt_y_2",
        "sess_y",
        2,
        "message",
        {
            "role": "tool",
            "tool_call_id": "call_y1",
            "name": "bash",
            "content": "tenant_y_result_marker",
        },
        "tool",
        "signal:group/999",
        "bash",
        None,
        None,
        None,
    ),
    (
        "evt_y_3",
        "sess_y",
        3,
        "span",
        {"event": "tool_execute_start", "tool_call_id": "call_y1", "tool_name": "bash"},
        None,
        None,
        None,
        None,
        None,
        "signal:group/999",
    ),
    (
        "evt_y_4",
        "sess_y",
        4,
        "lifecycle",
        {"event": "request_opened", "request_id": "req_y", "summary": "tenant_y_summary_marker"},
        None,
        None,
        None,
        None,
        None,
        None,
    ),
]

_TENANT_Y_MARKERS = (
    "tenant_y_content_marker",
    "tenant_y_args_marker",
    "tenant_y_result_marker",
    "tenant_y_summary_marker",
    "call_y1",
    "req_y",
)

# Column names no allowlisted view may expose: the raw payload (would defeat
# key-level redaction), the 0022 cost/token fence, the events-table token
# counters, and the lifecycle internals the issue names.
_DENYLISTED_COLUMNS = frozenset(
    {
        "data",
        "model",
        "model_usage",
        "local_tokens",
        "local_tokens_by_class",
        "cost_usd",
        "cumulative_tokens",
        "cumulative_class_mass",
        "frozen_surface",
        "output_schema",
        "vault_ids",
        "environment_id",
    }
)

# Payload-value sentinels that must never appear in any view output.
_LEAK_SENTINELS = (
    _SENTINEL_MODEL,
    _SENTINEL_USAGE,
    str(_SENTINEL_TOKENS),
    str(_SENTINEL_COST),
    _SENTINEL_FROZEN,
    _SENTINEL_SCHEMA,
    _SENTINEL_VAULT,
    _SENTINEL_ENV,
    str(_SENTINEL_CUMULATIVE_TOKENS),
    _SENTINEL_CUMULATIVE_CLASS_MASS,
)


@pytest.fixture(scope="module")
def db_url() -> Iterator[str]:
    """One container + one ``upgrade head`` + one seed for the whole module —
    every test below is read-only (the EXPLAIN test's DDL is rolled back)."""
    if not _docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        url = _alembic_url(pg)
        result = _run_alembic(["upgrade", "head"], url)
        assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}\n{result.stdout}"
        asyncio.run(_seed(url))
        yield url


async def _seed(db_url: str) -> None:
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(_SEED_SQL)
        for ev_id, session, seq, kind, data, role, channel, tool, is_err, sender, focal in _EVENTS:
            account = "acc_a" if session == "sess_x" else "acc_b"
            await conn.execute(
                "INSERT INTO events (id, session_id, seq, kind, data, account_id, role,"
                " channel, tool_name, is_error, sender_name, focal_channel_at_arrival)"
                " VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8, $9, $10, $11, $12)",
                ev_id,
                session,
                seq,
                kind,
                json.dumps(data),
                account,
                role,
                channel,
                tool,
                is_err,
                sender,
                focal,
            )
    finally:
        await conn.close()


async def _scoped_fetch(db_url: str, session_id: str, sql: str) -> list[asyncpg.Record]:
    """Fetch with the ``app.session_id`` GUC set the way the tool sets it."""
    conn = await asyncpg.connect(db_url)
    try:
        async with conn.transaction(readonly=True):
            await conn.execute("SELECT set_config('app.session_id', $1, true)", session_id)
            return list(await conn.fetch(sql))
    finally:
        await conn.close()


async def _bare_fetch(db_url: str, sql: str, *args: object) -> list[asyncpg.Record]:
    conn = await asyncpg.connect(db_url)
    try:
        return list(await conn.fetch(sql, *args))
    finally:
        await conn.close()


def _render(rows: list[asyncpg.Record]) -> str:
    """Flatten rows to one text blob for substring (leak) assertions."""
    return "\n".join(" | ".join(str(v) for v in row.values()) for row in rows)


# ---------------------------------------------------------------------------
# Schema: views + span index exist.
# ---------------------------------------------------------------------------


@needs_docker
@pytest.mark.integration
def test_views_and_span_index_exist(db_url: str) -> None:
    views = asyncio.run(
        _bare_fetch(
            db_url,
            "SELECT table_name FROM information_schema.views WHERE table_schema = 'public'",
        )
    )
    names = {r["table_name"] for r in views}
    assert names >= _ALLOWED_RELATIONS, f"missing views: {_ALLOWED_RELATIONS - names}"

    idx = asyncio.run(
        _bare_fetch(
            db_url,
            "SELECT indexdef FROM pg_indexes"
            " WHERE tablename = 'events' AND indexname = 'events_session_span_seq_idx'",
        )
    )
    assert len(idx) == 1, "events_session_span_seq_idx missing"
    assert "WHERE (kind = 'span'::text)" in idx[0]["indexdef"]


# ---------------------------------------------------------------------------
# Denylist: redacted columns and payload values never surface.
# ---------------------------------------------------------------------------


@needs_docker
@pytest.mark.integration
def test_denylisted_columns_not_exposed(db_url: str) -> None:
    """No allowlisted view exposes a denylisted column name. Checked against
    the LIVE catalog so a view edit that projects ``data`` (or any cost/token
    column) fails here regardless of what the migration source claims."""
    for relation in sorted(_ALLOWED_RELATIONS):
        cols = asyncio.run(
            _bare_fetch(
                db_url,
                "SELECT column_name FROM information_schema.columns WHERE table_name = $1",
                relation,
            )
        )
        names = {r["column_name"] for r in cols}
        assert names, f"view {relation} has no columns?"
        leaked = names & _DENYLISTED_COLUMNS
        assert not leaked, f"view {relation} exposes denylisted columns: {sorted(leaked)}"


@needs_docker
@pytest.mark.integration
def test_denylisted_payload_values_never_leak(db_url: str) -> None:
    """Execution-based redaction proof: sentinel VALUES seeded into every
    redacted payload key (model name, usage, token counts, cost, frozen
    surface, output schema, vault ids, environment id) appear NOWHERE in a
    full ``SELECT *`` of any allowlisted view — while a non-redacted payload
    key DOES flow into ``lifecycle_search.detail_text`` (positive control, so
    an empty/broken rendering can't vacuously pass)."""
    blobs: list[str] = []
    for relation in sorted(_ALLOWED_RELATIONS):
        rows = asyncio.run(_scoped_fetch(db_url, "sess_x", f"SELECT * FROM {relation}"))
        blobs.append(_render(rows))
    combined = "\n".join(blobs)

    for sentinel in _LEAK_SENTINELS:
        assert sentinel not in combined, f"redacted payload value leaked: {sentinel}"

    detail = asyncio.run(
        _scoped_fetch(
            db_url,
            "sess_x",
            "SELECT detail_text FROM lifecycle_search WHERE lifecycle_kind = 'request_opened'",
        )
    )
    assert len(detail) == 1
    assert _VISIBLE_DETAIL in detail[0]["detail_text"], (
        "positive control failed: non-redacted payload key missing from detail_text — "
        "the leak assertions above may be vacuous"
    )
    # detail_len reports the FULL redacted-payload length (pre-cap honesty).
    assert _SENTINEL_FROZEN not in detail[0]["detail_text"]


# ---------------------------------------------------------------------------
# Seeded span derivation (non-NULL columns; linking-id self-join).
# ---------------------------------------------------------------------------


@needs_docker
@pytest.mark.integration
def test_spans_search_seeded_derivation(db_url: str) -> None:
    """Span rows are ``data->>…`` derivations (no promoted columns to lean
    on): prove every derived column is non-NULL on writer-shaped rows, and
    that ``start_event_id`` resolves the writers' ``<prefix>_start_id``
    linking keys (``tool_execute_start_id`` AND ``model_request_start_id`` —
    an always-NULL derivation, e.g. reading a literal ``start_event_id`` key,
    fails this test)."""
    rows = asyncio.run(_scoped_fetch(db_url, "sess_x", "SELECT * FROM spans_search ORDER BY seq"))
    by_id = {r["event_id"]: r for r in rows}
    assert set(by_id) == {"evt_span_start_1", "evt_span_end_1", "evt_mrs_1", "evt_mre_1"}

    end = by_id["evt_span_end_1"]
    assert end["span_kind"] == "tool_execute_end"
    assert end["tool_call_id"] == "call_1"
    assert end["tool_name"] == "bash"
    assert end["start_event_id"] == "evt_span_start_1"
    assert end["is_error"] is False
    assert end["focal_channel"] == "telegram:chat/12345"

    mre = by_id["evt_mre_1"]
    assert mre["span_kind"] == "model_request_end"
    assert mre["start_event_id"] == "evt_mrs_1"

    start = by_id["evt_span_start_1"]
    assert start["span_kind"] == "tool_execute_start"
    assert start["start_event_id"] is None
    assert start["is_error"] is None

    # The documented duration self-join pairs each *_end to its *_start.
    pairs = asyncio.run(
        _scoped_fetch(
            db_url,
            "sess_x",
            "SELECT e.event_id, s.event_id AS start_id,"
            " e.created_at - s.created_at AS duration"
            " FROM spans_search e JOIN spans_search s ON s.event_id = e.start_event_id",
        )
    )
    assert {(p["event_id"], p["start_id"]) for p in pairs} == {
        ("evt_span_end_1", "evt_span_start_1"),
        ("evt_mre_1", "evt_mrs_1"),
    }


# ---------------------------------------------------------------------------
# tool_calls_search: ordinality, pairing, cap + hash honesty.
# ---------------------------------------------------------------------------


@needs_docker
@pytest.mark.integration
def test_tool_calls_search_pairing_ordinality_and_cap(db_url: str) -> None:
    rows = asyncio.run(
        _scoped_fetch(
            db_url, "sess_x", "SELECT * FROM tool_calls_search ORDER BY seq, call_ordinal"
        )
    )
    assert [(r["tool_call_id"], r["call_ordinal"]) for r in rows] == [
        ("call_1", 1),
        ("call_2", 2),
        ("call_3", 1),
    ]
    by_call = {r["tool_call_id"]: r for r in rows}

    c1 = by_call["call_1"]
    assert c1["tool_name"] == "bash"
    assert c1["arguments_text"] == _ARGS_SMALL
    assert c1["args_len"] == len(_ARGS_SMALL.encode())
    assert c1["args_sha256"] == hashlib.sha256(_ARGS_SMALL.encode()).hexdigest()
    assert c1["result_seq"] == 4
    assert "ok files listed" in c1["result_text"]
    assert c1["result_is_error"] is None  # success results are NULL, never FALSE

    c2 = by_call["call_2"]
    assert c2["result_is_error"] is True
    assert "read failed" in c2["result_text"]

    # Oversized arguments: cell-capped with the explicit marker, while
    # args_len/args_sha256 stay pre-truncation (truncation always detectable,
    # equality survives it).
    c3 = by_call["call_3"]
    assert c3["arguments_text"].endswith("…[truncated]")
    assert len(c3["arguments_text"].encode()) <= 16384
    assert c3["args_len"] == len(_ARGS_BIG.encode())
    assert c3["args_len"] > 16384
    assert c3["args_sha256"] == hashlib.sha256(_ARGS_BIG.encode()).hexdigest()


# ---------------------------------------------------------------------------
# lifecycle_search: kind allowlist is fail-closed; status mapping.
# ---------------------------------------------------------------------------


@needs_docker
@pytest.mark.integration
def test_lifecycle_kind_allowlist_fail_closed(db_url: str) -> None:
    rows = asyncio.run(
        _scoped_fetch(db_url, "sess_x", "SELECT * FROM lifecycle_search ORDER BY seq")
    )
    kinds = [r["lifecycle_kind"] for r in rows]
    assert kinds == ["request_opened", "request_response", "turn_ended", "tool_confirmed"]
    assert "wake_deferred" not in kinds  # non-allowlisted kind stays invisible

    by_kind = {r["lifecycle_kind"]: r for r in rows}
    opened = by_kind["request_opened"]
    assert opened["request_id"] == "req_1"
    assert opened["summary"] == "do the thing"
    assert opened["awaited"] is True
    assert opened["caller_kind"] == "session"

    # status carries turn_ended.status / request_response.is_error.
    assert by_kind["turn_ended"]["status"] == "completed"
    assert by_kind["request_response"]["status"] == "false"


# ---------------------------------------------------------------------------
# Session scoping of the three data views.
# ---------------------------------------------------------------------------


@needs_docker
@pytest.mark.integration
def test_new_views_are_session_scoped(db_url: str) -> None:
    """Session X must not see any of session Y's rows through the new views —
    and vice versa (the ``events_search`` isolation invariant)."""
    x_blob = "\n".join(
        _render(asyncio.run(_scoped_fetch(db_url, "sess_x", f"SELECT * FROM {rel}")))
        for rel in ("tool_calls_search", "spans_search", "lifecycle_search")
    )
    for marker in _TENANT_Y_MARKERS:
        assert marker not in x_blob, f"session X sees session Y data: {marker}"

    y_calls = asyncio.run(_scoped_fetch(db_url, "sess_y", "SELECT * FROM tool_calls_search"))
    assert [r["tool_call_id"] for r in y_calls] == ["call_y1"]
    y_spans = asyncio.run(_scoped_fetch(db_url, "sess_y", "SELECT * FROM spans_search"))
    assert [r["event_id"] for r in y_spans] == ["evt_y_3"]
    y_lc = asyncio.run(_scoped_fetch(db_url, "sess_y", "SELECT * FROM lifecycle_search"))
    assert [r["request_id"] for r in y_lc] == ["req_y"]


# ---------------------------------------------------------------------------
# Help completeness: every column of every allowlisted relation — the help
# view included — has exactly one help row, and no help row describes a
# column that does not exist.
# ---------------------------------------------------------------------------


@needs_docker
@pytest.mark.integration
def test_help_rows_cover_every_allowlisted_column_exactly(db_url: str) -> None:
    help_rows = asyncio.run(
        _scoped_fetch(
            db_url,
            "sess_x",
            "SELECT relation_name, column_name FROM search_views_help"
            " WHERE column_name NOT LIKE '\\_\\_%'",
        )
    )
    documented: dict[str, set[str]] = {}
    for r in help_rows:
        documented.setdefault(r["relation_name"], set()).add(r["column_name"])

    # Help rows only ever describe allowlisted relations.
    assert set(documented) == set(_ALLOWED_RELATIONS)

    for relation in sorted(_ALLOWED_RELATIONS):
        cols = asyncio.run(
            _bare_fetch(
                db_url,
                "SELECT column_name FROM information_schema.columns WHERE table_name = $1",
                relation,
            )
        )
        actual = {r["column_name"] for r in cols}
        assert documented[relation] == actual, (
            f"help drift for {relation}: undocumented columns "
            f"{sorted(actual - documented[relation])}, phantom help rows "
            f"{sorted(documented[relation] - actual)}"
        )

    # Idiom/example rows carry executable SQL (the executable-example test
    # runs them); column rows carry empty example_sql.
    dunder = asyncio.run(
        _scoped_fetch(
            db_url,
            "sess_x",
            "SELECT example_sql FROM search_views_help WHERE column_name LIKE '\\_\\_%'",
        )
    )
    assert dunder, "expected idiom/example help rows"
    assert all(r["example_sql"].lstrip().upper().startswith("SELECT") for r in dunder)


# ---------------------------------------------------------------------------
# search_views_help: the documented static carve-in, tested directly.
# ---------------------------------------------------------------------------


@needs_docker
@pytest.mark.integration
def test_search_views_help_carve_in_contract(db_url: str) -> None:
    """The help view's ``current_setting(...) IS NOT NULL`` predicate is a
    fail-closed gate, not row scoping — safe only because the view carries no
    tenant data. Assert all three legs of that carve-in contract directly:

    1. the view reads ZERO tables (``information_schema.view_table_usage`` is
       empty — it is a pure VALUES catalog, so there is nothing to scope);
    2. with the GUC unset it returns zero rows (fail-closed on any connection
       that skipped the tool's ``set_config`` discipline);
    3. two different sessions see byte-identical rows (session-independent,
       like ``information_schema``)."""
    usage = asyncio.run(
        _bare_fetch(
            db_url,
            "SELECT table_name FROM information_schema.view_table_usage"
            " WHERE view_name = 'search_views_help'",
        )
    )
    assert usage == [], f"search_views_help reads tables: {usage} — the carve-in is void"

    unset = asyncio.run(_bare_fetch(db_url, "SELECT count(*) AS n FROM search_views_help"))
    assert unset[0]["n"] == 0, "help view must return nothing without the app.session_id GUC"

    as_x = asyncio.run(
        _scoped_fetch(db_url, "sess_x", "SELECT * FROM search_views_help ORDER BY 1, 2, 3, 4, 5")
    )
    as_y = asyncio.run(
        _scoped_fetch(db_url, "sess_y", "SELECT * FROM search_views_help ORDER BY 1, 2, 3, 4, 5")
    )
    assert as_x, "help view returned no rows under a set GUC"
    assert [tuple(r.values()) for r in as_x] == [tuple(r.values()) for r in as_y]


# ---------------------------------------------------------------------------
# EXPLAIN: the intended partial indexes serve the view predicates.
# ---------------------------------------------------------------------------


async def _explain_isolated(db_url: str, session_id: str, sql: str, keep_index: str) -> str:
    """EXPLAIN ``sql`` with ``keep_index`` as the ONLY surviving index on
    ``events`` (every other index and the (session_id, seq) unique constraint
    are dropped inside a transaction that is rolled back) and seqscans
    disabled. On a fixture-sized table the planner's costs are all ties, so
    index CHOICE is not meaningful — index USABILITY is: with no competitors
    and seqscan off, the asserted index appears in the plan if and only if
    its partial predicate actually matches the view's WHERE clause (e.g. the
    promoted ``role`` column, not ``data->>'role'``)."""
    conn = await asyncpg.connect(db_url)
    try:
        tr = conn.transaction()
        await tr.start()
        try:
            constraints = await conn.fetch(
                "SELECT conname FROM pg_constraint"
                " WHERE conrelid = 'events'::regclass AND contype = 'u'"
            )
            for row in constraints:
                await conn.execute(f'ALTER TABLE events DROP CONSTRAINT "{row["conname"]}"')
            indexes = await conn.fetch(
                "SELECT indexname FROM pg_indexes WHERE tablename = 'events'"
                " AND indexname NOT IN ($1, 'events_pkey')",
                keep_index,
            )
            for row in indexes:
                await conn.execute(f'DROP INDEX "{row["indexname"]}"')
            await conn.execute("SELECT set_config('app.session_id', $1, true)", session_id)
            await conn.execute("SET LOCAL enable_seqscan = off")
            plan = await conn.fetch(f"EXPLAIN {sql}")
            return "\n".join(r[0] for r in plan)
        finally:
            await tr.rollback()
    finally:
        await conn.close()


@needs_docker
@pytest.mark.integration
def test_explain_view_predicates_use_intended_partial_indexes(db_url: str) -> None:
    cases = [
        # Driving arm of tool_calls_search: must match the 0023 partial index
        # (kind='message' AND role='assistant' AND data ? 'tool_calls' on the
        # PROMOTED role column — the reason the view filters a.role).
        (
            "SELECT event_id, tool_name FROM tool_calls_search ORDER BY seq, call_ordinal",
            "events_assistant_tool_calls_idx",
        ),
        # Result-join arm: the unique (session_id, data->>'tool_call_id')
        # partial index over role='tool' rows.
        (
            "SELECT event_id, result_seq FROM tool_calls_search ORDER BY seq, call_ordinal",
            "events_tool_result_idx",
        ),
        # spans_search windowed read: the NEW 0144 span partial index.
        (
            "SELECT * FROM spans_search WHERE seq BETWEEN 1 AND 100",
            "events_session_span_seq_idx",
        ),
        # lifecycle_search: the 0135 lifecycle partial index (single-kind
        # predicate — a kind IN (…) list would defeat it).
        (
            "SELECT * FROM lifecycle_search ORDER BY seq DESC LIMIT 10",
            "events_session_lifecycle_seq_idx",
        ),
    ]
    for sql, index in cases:
        plan = asyncio.run(_explain_isolated(db_url, "sess_x", sql, index))
        assert index in plan, (
            f"expected {index} to serve:\n  {sql}\nplan was:\n{plan}\n"
            f"(the view predicate no longer matches the index's partial predicate)"
        )


# ---------------------------------------------------------------------------
# Executable examples: every documented example runs — and returns rows —
# through the real tool handler.
# ---------------------------------------------------------------------------


def _extract_sql_examples(text: str) -> list[str]:
    """Pull example queries out of prose: a line matching ``^  SELECT ``
    starts an example; deeper-indented non-blank lines continue it; anything
    else ends it."""
    examples: list[str] = []
    current: list[str] | None = None
    for line in text.split("\n"):
        if line.startswith("  SELECT "):
            if current is not None:
                examples.append(" ".join(current))
            current = [line.strip()]
        elif current is not None and line.startswith("  ") and line.strip():
            current.append(line.strip())
        elif current is not None:
            examples.append(" ".join(current))
            current = None
    if current is not None:
        examples.append(" ".join(current))
    return examples


@needs_docker
@pytest.mark.integration
def test_every_documented_example_executes_and_returns_rows(db_url: str) -> None:
    """The fixture is deliberately seeded so every example in the tool
    description, the parameters schema, and the ``search_views_help`` idiom
    rows returns at least one row. Non-empty (not merely non-error) is the
    bar: it proves each example is self-contained against real data, and it
    forces whoever adds a new example to seed fixture support for it."""
    from aios.db.pool import create_pool
    from aios.harness import runtime
    from aios.tools.search_events import (
        SEARCH_EVENTS_DESCRIPTION,
        SEARCH_EVENTS_PARAMETERS_SCHEMA,
        search_events_handler,
    )

    description_examples = _extract_sql_examples(SEARCH_EVENTS_DESCRIPTION)
    schema_examples = _extract_sql_examples(
        SEARCH_EVENTS_PARAMETERS_SCHEMA["properties"]["query"]["description"]
    )
    # Extractor sanity: a formatting change that hides the examples from the
    # extractor must fail here, not silently shrink coverage.
    assert len(description_examples) >= 6, description_examples
    assert len(schema_examples) >= 3, schema_examples

    help_examples = [
        r["example_sql"]
        for r in asyncio.run(
            _scoped_fetch(
                db_url,
                "sess_x",
                "SELECT example_sql FROM search_views_help"
                " WHERE column_name LIKE '\\_\\_%' AND example_sql <> ''",
            )
        )
    ]
    assert len(help_examples) >= 5, help_examples

    async def _run_all() -> list[tuple[str, str]]:
        failures: list[tuple[str, str]] = []
        pool = await create_pool(db_url, min_size=1, max_size=2)
        prev = runtime.pool
        runtime.pool = pool
        try:
            for query in [*description_examples, *schema_examples, *help_examples]:
                try:
                    result = await search_events_handler("sess_x", {"query": query})
                except Exception as exc:
                    failures.append((query, f"raised: {exc}"))
                    continue
                if result["result"] == "No results.":
                    failures.append((query, "returned no rows against the seeded fixture"))
        finally:
            runtime.pool = prev
            await pool.close()
        return failures

    failures = asyncio.run(_run_all())
    assert not failures, "documented examples failed:\n" + "\n".join(
        f"  {reason}: {query}" for query, reason in failures
    )
