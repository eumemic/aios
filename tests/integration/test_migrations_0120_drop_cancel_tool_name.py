"""Migration 0120 drops the retired ``cancel`` builtin tool name from persisted ``tools``.

Exercises ``_aios_drop_cancel_tool`` on ``workflows`` (the minimal seed; the same helper
scrubs all six surface tables): the ``cancel`` element is removed with order + custom-tool
preservation, a ``[cancel]``-only list collapses to ``[]``, and the EXISTS gate leaves a
clean row byte-untouched.
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

# wf_cxl: cancel amid a plain builtin + a custom tool. wf_only: cancel alone → collapses to [].
# wf_clean: no cancel (left untouched by the EXISTS gate).
_SEED_SQL = """
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_root', NULL, TRUE, 'root');
INSERT INTO workflows (id, account_id, name, version, script, tools)
VALUES
  ('wf_cxl', 'acc_root', 'cxl', 1, 'S',
   '[{"type":"bash"},{"type":"cancel"},
     {"type":"custom","name":"foo","description":"d","input_schema":{}}]'::jsonb),
  ('wf_only', 'acc_root', 'only', 1, 'S',
   '[{"type":"cancel"}]'::jsonb),
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
def test_cancel_tool_name_dropped(postgres: object) -> None:
    db_url = _alembic_url(postgres)

    # Seed AFTER 0119 (the pre-0120 head) so the cancel rows are pristine.
    up = _run_alembic(["upgrade", "0119"], db_url)
    assert up.returncode == 0, f"upgrade to 0119 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _SEED_SQL))

    up = _run_alembic(["upgrade", "0120"], db_url)
    assert up.returncode == 0, f"upgrade to 0120 failed:\n{up.stderr}\n{up.stdout}"

    # cancel removed; bash + custom untouched, original order preserved.
    cxl = asyncio.run(_fetch_tools(db_url, "wf_cxl"))
    assert [t["type"] for t in cxl] == ["bash", "custom"]
    assert cxl[1]["name"] == "foo"  # custom spec preserved verbatim

    # A cancel-only list collapses to [].
    only = asyncio.run(_fetch_tools(db_url, "wf_only"))
    assert only == []

    # Clean row: no cancel → left byte-untouched by the EXISTS gate.
    clean = asyncio.run(_fetch_tools(db_url, "wf_clean"))
    assert [t["type"] for t in clean] == ["bash", "stop_task"]
