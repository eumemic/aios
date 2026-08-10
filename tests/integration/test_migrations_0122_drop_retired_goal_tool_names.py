"""Migration 0122 drops the retired ``complete_goal``/``fail_goal`` builtin tool names.

#1525 removed ``complete_goal``/``fail_goal`` from ``BuiltinToolType`` + the registry without a
read shim or a data migration, so any persisted ``tools`` JSONB still carrying them poisons
``ToolSpec`` validation on read (the kedalion-ultron wedge, #1562). Migration 0122 scrubs both
retired ``type`` values from every surface table; this exercises ``_aios_drop_retired_goal_tools``
on ``workflows`` (the minimal seed; the same helper scrubs all six surface tables): the retired
elements are removed with order + custom-tool preservation, a retired-only list collapses to
``[]``, and the EXISTS gate leaves a clean row byte-untouched.
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

# wf_goal: both retired verbs amid a plain builtin + a custom tool.
# wf_only: complete_goal + fail_goal alone → collapses to [].
# wf_clean: no retired verb (left untouched by the EXISTS gate).
_SEED_SQL = """
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_root', NULL, TRUE, 'root');
INSERT INTO workflows (id, account_id, name, version, script, tools)
VALUES
  ('wf_goal', 'acc_root', 'goal', 1, 'S',
   '[{"type":"bash"},{"type":"complete_goal"},
     {"type":"custom","name":"foo","description":"d","input_schema":{}},
     {"type":"fail_goal"}]'::jsonb),
  ('wf_only', 'acc_root', 'only', 1, 'S',
   '[{"type":"complete_goal"},{"type":"fail_goal"}]'::jsonb),
  ('wf_clean', 'acc_root', 'clean', 1, 'S',
   '[{"type":"bash"},{"type":"create_goal"}]'::jsonb);
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
def test_retired_goal_tool_names_dropped(postgres: object) -> None:
    db_url = _alembic_url(postgres)

    # Seed AFTER 0121 (the pre-0122 head) so the retired rows are pristine.
    up = _run_alembic(["upgrade", "0121"], db_url)
    assert up.returncode == 0, f"upgrade to 0121 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _SEED_SQL))

    up = _run_alembic(["upgrade", "0122"], db_url)
    assert up.returncode == 0, f"upgrade to 0122 failed:\n{up.stderr}\n{up.stdout}"

    # Both retired verbs removed; bash + custom untouched, original order preserved.
    goal = asyncio.run(_fetch_tools(db_url, "wf_goal"))
    assert [t["type"] for t in goal] == ["bash", "custom"]
    assert goal[1]["name"] == "foo"  # custom spec preserved verbatim

    # A retired-only list collapses to [].
    only = asyncio.run(_fetch_tools(db_url, "wf_only"))
    assert only == []

    # Clean row: no retired verb → left byte-untouched by the EXISTS gate.
    clean = asyncio.run(_fetch_tools(db_url, "wf_clean"))
    assert [t["type"] for t in clean] == ["bash", "create_goal"]
