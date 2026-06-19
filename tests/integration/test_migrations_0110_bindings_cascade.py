"""Migration 0110 restores ``ON DELETE CASCADE`` on ``bindings.session_id``
in the original single-column form ``session_id REFERENCES sessions(id) ON
DELETE CASCADE`` (the shape the 0015 table declared, before the 0033
redesign dropped the cascade). The composite ``(session_id, account_id)``
form is deliberately *not* used here because ``bindings`` is the one
session-child whose ``account_id`` is rewritten independently of its
``session_id`` by ``reparent_connection``.

The cascade existed in the original ``bindings`` table (0015) and was
dropped when the 0033 connector redesign recreated the table. These tests
assert the constraint *definition* swaps correctly on upgrade and is
restored to the bare cascade-less form on downgrade. The end-to-end cascade
behaviour (a raw ``DELETE FROM sessions`` cleaning up bindings) is covered
by ``test_delete_session_with_binding.py``, which builds the row chain
through the service/query layer.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import asyncpg
import pytest

from tests.conftest import _docker_available, needs_docker
from tests.integration.test_migrations import _alembic_url, _run_alembic


@pytest.fixture
def postgres() -> Iterator[object]:
    """Fresh function-scoped Postgres; each test mutates alembic_version."""
    if not _docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


async def _fetchval(db_url: str, sql: str) -> object:
    conn = await asyncpg.connect(db_url)
    try:
        return await conn.fetchval(sql)
    finally:
        await conn.close()


def _bindings_fk_def(db_url: str) -> str:
    """Return the ``pg_get_constraintdef`` text of the (sole) FK on
    ``bindings`` that references ``sessions``."""
    return str(
        asyncio.run(
            _fetchval(
                db_url,
                """
                SELECT pg_get_constraintdef(c.oid)
                  FROM pg_constraint c
                  JOIN pg_class t       ON t.oid = c.conrelid
                  JOIN pg_class reffed  ON reffed.oid = c.confrelid
                 WHERE t.relname = 'bindings'
                   AND reffed.relname = 'sessions'
                   AND c.contype = 'f'
                """,
            )
        )
    )


@needs_docker
@pytest.mark.integration
def test_upgrade_swaps_to_cascade_fk(postgres: object) -> None:
    db_url = _alembic_url(postgres)

    # Before 0110: the bare single-column FK with no ON DELETE action.
    up = _run_alembic(["upgrade", "0108"], db_url)
    assert up.returncode == 0, f"upgrade to 0108 failed:\n{up.stderr}\n{up.stdout}"
    before = _bindings_fk_def(db_url)
    assert "ON DELETE CASCADE" not in before, f"unexpected cascade pre-0110: {before}"

    up = _run_alembic(["upgrade", "head"], db_url)
    assert up.returncode == 0, f"upgrade to head failed:\n{up.stderr}\n{up.stdout}"
    after = _bindings_fk_def(db_url)
    assert "ON DELETE CASCADE" in after, f"cascade missing after 0110: {after}"
    # Single-column form, matching the original 0015 shape. A tenant-scoped
    # composite FK is deliberately avoided: ``bindings.account_id`` is
    # rewritten independently of ``session_id`` by ``reparent_connection``,
    # which a composite ``(session_id, account_id)`` FK would break.
    assert "FOREIGN KEY (session_id)" in after, f"not single-column: {after}"
    assert "REFERENCES sessions(id)" in after, f"wrong target: {after}"
    assert "account_id" not in after, f"unexpectedly composite/tenant-scoped: {after}"


@needs_docker
@pytest.mark.integration
def test_downgrade_restores_bare_fk(postgres: object) -> None:
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "head"], db_url)
    assert up.returncode == 0, f"upgrade to head failed:\n{up.stderr}\n{up.stdout}"

    down = _run_alembic(["downgrade", "0108"], db_url)
    assert down.returncode == 0, f"downgrade to 0108 failed:\n{down.stderr}\n{down.stdout}"

    restored = _bindings_fk_def(db_url)
    assert "ON DELETE CASCADE" not in restored, f"cascade lingered post-downgrade: {restored}"
    assert "(session_id)" in restored, f"bare single-column FK not restored: {restored}"
