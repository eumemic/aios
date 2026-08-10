"""Migration 0116 normalizes pre-#1419 builtin tool names in persisted ``tools`` JSONB.

Exercises the shared ``_aios_normalize_legacy_tools`` helper on ``workflows`` (the minimal
seed; the same helper rewrites all six surface tables): the rename, the
``invoke_workflow``/``create_run`` collapse-and-dedupe, order preservation, custom-tool
preservation, and the legacy-EXISTS gate (a clean row is left untouched).
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

# A workflow whose declared surface mixes a plain builtin, two legacy names that collapse
# to call_workflow, a renamed builtin, and a custom tool — plus a CLEAN workflow the gate
# must leave alone.
_SEED_SQL = """
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_root', NULL, TRUE, 'root');
INSERT INTO workflows (id, account_id, name, version, script, tools)
VALUES
  ('wf_legacy', 'acc_root', 'legacy', 1, 'S',
   '[{"type":"bash"},{"type":"invoke_workflow"},{"type":"create_run"},
     {"type":"invoke_agent"},
     {"type":"custom","name":"foo","description":"d","input_schema":{}}]'::jsonb),
  ('wf_clean', 'acc_root', 'clean', 1, 'S',
   '[{"type":"bash"},{"type":"call_workflow"}]'::jsonb);
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
def test_legacy_tool_names_normalized_and_deduped(postgres: object) -> None:
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "0115"], db_url)
    assert up.returncode == 0, f"upgrade to 0115 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _SEED_SQL))

    up = _run_alembic(["upgrade", "0116"], db_url)
    assert up.returncode == 0, f"upgrade to 0116 failed:\n{up.stderr}\n{up.stdout}"

    # Legacy row: invoke_workflow + create_run collapse to ONE call_workflow (first wins),
    # invoke_agent → call_agent, bash + custom untouched, original order preserved.
    legacy = asyncio.run(_fetch_tools(db_url, "wf_legacy"))
    assert [t["type"] for t in legacy] == ["bash", "call_workflow", "call_agent", "custom"]
    assert legacy[3]["name"] == "foo"  # custom spec preserved verbatim

    # Clean row: no legacy names → left byte-untouched by the EXISTS gate.
    clean = asyncio.run(_fetch_tools(db_url, "wf_clean"))
    assert [t["type"] for t in clean] == ["bash", "call_workflow"]
