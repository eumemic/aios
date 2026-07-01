"""Integration test: a connector ``tools_schema`` is validated at the
``PUT /v1/connectors/{connector}/tools_schema`` authoring boundary (#1652).

Each published entry is read back at step time by
``compute_step_prelude`` (step_context.py) via ``ToolSpec.model_validate``.
A malformed entry raises there — lazily, at step time — crashing the
prelude for EVERY session bound to a connection of that connector type,
because the ``connectors.tools_schema`` row is connector-type-wide. That
is a fail-late DoS gated on operator/author-controlled input.

The fix validates each entry against ``ToolSpec`` at the PUT boundary
(service layer, mirroring the exact model and site the prelude uses) so a
bad schema fails fast at the operator/author edge (a 422 ``ValidationError``)
rather than wedging live sessions. The root-only auth gate still runs
FIRST, so a child tenant sees Forbidden — the validation error never
leaks the write path to an unauthorized caller.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.db.pool import create_pool
from aios.errors import ForbiddenError, ValidationError
from aios.services import connectors as connectors_service

pytestmark = pytest.mark.integration


@pytest.fixture
async def root_and_child(
    migrated_db_url: str, _reset_db_state: None
) -> AsyncIterator[tuple[asyncpg.Pool[Any], str, str]]:
    """Yield ``(pool, root_account_id, child_account_id)`` with a
    seeded ``echo`` connector row."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
                VALUES ('acc_root',  NULL,      TRUE,  'tenant-root'),
                       ('acc_child', 'acc_root', FALSE, 'tenant-child')
                """
            )
            await conn.execute(
                "INSERT INTO connectors (connector) VALUES ('echo') ON CONFLICT DO NOTHING"
            )
        yield pool, "acc_root", "acc_child"
    finally:
        await pool.close()


# The exact shape ``compute_step_prelude`` reads back and feeds to
# ``ToolSpec.model_validate``; what the SDK's ``derive_tool_spec`` publishes.
_VALID_ENTRY = {
    "type": "custom",
    "name": "send",
    "description": "send a message",
    "input_schema": {"type": "object", "properties": {}},
}


async def test_malformed_schema_rejected_at_put_boundary(
    root_and_child: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """A malformed entry — one that would raise in the prelude's
    ``ToolSpec.model_validate`` — must be rejected here, before it is
    persisted, and NOT wedge every bound session at step time."""
    pool, root_id, _child_id = root_and_child

    with pytest.raises(ValidationError) as excinfo:
        await connectors_service.update_tools_schema(
            pool,
            connector="echo",
            account_id=root_id,
            # Missing the required ``type`` discriminator (and carrying an
            # unknown ``parameters`` key) — exactly what crashes the prelude.
            tools_schema=[{"name": "send", "description": "d", "parameters": {}}],
        )

    err = excinfo.value
    assert err.status_code == 422
    assert err.detail.get("connector") == "echo"
    # The offending entry is pinpointed for the operator.
    assert err.detail.get("index") == 0
    assert err.detail.get("errors")

    # Fail-fast means the poisoned row is never written — the global
    # ``connectors.tools_schema`` still reads back as the empty default.
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT tools_schema FROM connectors WHERE connector = 'echo'")
    assert row is not None
    assert row["tools_schema"] == []


async def test_second_entry_malformed_reports_its_index(
    root_and_child: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """A valid first entry followed by a malformed one is still rejected,
    with the index of the offending entry surfaced (and nothing written)."""
    pool, root_id, _child_id = root_and_child

    with pytest.raises(ValidationError) as excinfo:
        await connectors_service.update_tools_schema(
            pool,
            connector="echo",
            account_id=root_id,
            tools_schema=[_VALID_ENTRY, {"type": "custom", "name": None}],
        )

    assert excinfo.value.detail.get("index") == 1

    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT tools_schema FROM connectors WHERE connector = 'echo'")
    assert row is not None
    assert row["tools_schema"] == []


async def test_valid_schema_is_persisted(
    root_and_child: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """A well-formed schema — the shape the SDK publishes — passes the
    boundary check and lands on the row."""
    pool, root_id, _child_id = root_and_child

    await connectors_service.update_tools_schema(
        pool,
        connector="echo",
        account_id=root_id,
        tools_schema=[_VALID_ENTRY],
    )

    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT tools_schema FROM connectors WHERE connector = 'echo'")
    assert row is not None
    raw = row["tools_schema"]
    import json

    schema = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    assert any(t.get("name") == "send" for t in schema)


async def test_auth_gate_runs_before_validation(
    root_and_child: tuple[asyncpg.Pool[Any], str, str],
) -> None:
    """A child tenant publishing a malformed schema must see Forbidden —
    the root-only gate runs FIRST, so validation never leaks the write
    path to an unauthorized caller."""
    pool, _root_id, child_id = root_and_child

    with pytest.raises(ForbiddenError):
        await connectors_service.update_tools_schema(
            pool,
            connector="echo",
            account_id=child_id,
            tools_schema=[{"name": "malicious", "description": "injection", "parameters": {}}],
        )
