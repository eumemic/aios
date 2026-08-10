"""Migration 0117 normalizes the retired ``cancel_run`` builtin tool name in persisted ``tools``.

Exercises the shared ``_aios_normalize_legacy_tools`` helper on ``workflows`` (the minimal seed;
the same helper rewrites all six surface tables): the 1:1 ``cancel_run``→``stop_task`` rename,
order + custom-tool preservation, the dedupe path (a row carrying BOTH ``cancel_run`` and
``stop_task`` collapses to one), and the legacy-EXISTS gate (a clean row is left untouched).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any, cast

import asyncpg
import pytest

from tests.conftest import _docker_available, needs_docker
from tests.integration.test_migrations import _alembic_url, _run_alembic

# wf_cxl: a legacy cancel_run amid a plain builtin + a custom tool. wf_both: already carries
# stop_task too (the rename would duplicate it → dedupe to one). wf_clean: no legacy name.
_SEED_SQL = """
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_root', NULL, TRUE, 'root');
INSERT INTO workflows (id, account_id, name, version, script, tools)
VALUES
  ('wf_cxl', 'acc_root', 'cxl', 1, 'S',
   '[{"type":"bash"},{"type":"cancel_run"},
     {"type":"custom","name":"foo","description":"d","input_schema":{}}]'::jsonb),
  ('wf_both', 'acc_root', 'both', 1, 'S',
   '[{"type":"cancel_run"},{"type":"stop_task"}]'::jsonb),
  ('wf_clean', 'acc_root', 'clean', 1, 'S',
   '[{"type":"bash"},{"type":"stop_task"}]'::jsonb);
"""


@pytest.fixture
def postgres() -> Iterator[object]:
    if not _docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


async def _fetch_tools(db_url: str, wf_id: str) -> list[dict[str, Any]]:
    conn = await asyncpg.connect(db_url)
    try:
        raw = await conn.fetchval("SELECT tools FROM workflows WHERE id = $1", wf_id)
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        return cast("list[dict[str, Any]]", parsed)
    finally:
        await conn.close()


async def _execute(db_url: str, sql: str) -> None:
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(sql)
    finally:
        await conn.close()


@needs_docker
@pytest.mark.integration
def test_cancel_run_tool_name_normalized_and_deduped(postgres: object) -> None:
    db_url = _alembic_url(postgres)

    # Seed AFTER 0116 (which only touches the invoke* names) so the cancel_run rows are pristine.
    up = _run_alembic(["upgrade", "0116"], db_url)
    assert up.returncode == 0, f"upgrade to 0116 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _SEED_SQL))

    up = _run_alembic(["upgrade", "0117"], db_url)
    assert up.returncode == 0, f"upgrade to 0117 failed:\n{up.stderr}\n{up.stdout}"

    # Legacy row: cancel_run → stop_task, bash + custom untouched, original order preserved.
    cxl = asyncio.run(_fetch_tools(db_url, "wf_cxl"))
    assert [t["type"] for t in cxl] == ["bash", "stop_task", "custom"]
    assert cxl[2]["name"] == "foo"  # custom spec preserved verbatim

    # Both row: cancel_run + stop_task collapse to ONE stop_task (first wins).
    both = asyncio.run(_fetch_tools(db_url, "wf_both"))
    assert [t["type"] for t in both] == ["stop_task"]

    # Clean row: no legacy names → left byte-untouched by the EXISTS gate.
    clean = asyncio.run(_fetch_tools(db_url, "wf_clean"))
    assert [t["type"] for t in clean] == ["bash", "stop_task"]
