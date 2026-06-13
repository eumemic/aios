"""`aios rekey` must cover EVERY encrypted column in the schema.

A vault-key rotation re-encrypts a hand-maintained list of ``(table,
ciphertext)`` columns (``ops._REKEY_COLUMNS``). If an encrypted column is added
to the schema but not to that list, rotation silently leaves it encrypted under
the OLD key — and once ``AIOS_VAULT_KEY_PREVIOUS`` is dropped after the
rotation, those rows are permanently undecryptable (data loss).

This pins the invariant by deriving the truth from the migrated schema rather
than a second hand-maintained list: every ``*ciphertext*`` column the
migrations create must have a ``_REKEY_COLUMNS`` entry. It caught
``oauth_flows`` (the Connect-flow state table, migration 0061) being omitted,
and it fails loudly the next time an encrypted table is added without wiring
rekey.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import asyncpg
import pytest

from aios.cli.commands.ops import _REKEY_COLUMNS
from tests.conftest import _docker_available, needs_docker
from tests.integration.test_migrations import _alembic_url, _run_alembic

# The legacy ``credentials`` table (migration 0001) was the one historically
# un-rekeyable encrypted column — no ``account_id`` for the per-account subkey,
# read/written by no current code. Migration 0099 dropped it, so the coverage
# assertion below is now TOTAL: every ``ciphertext`` column in the live schema
# must be covered by ``_REKEY_COLUMNS``, with no exceptions.


@pytest.fixture(scope="module")
def postgres() -> Iterator[object]:
    if not _docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


@needs_docker
@pytest.mark.integration
def test_rekey_covers_every_encrypted_column(postgres: object) -> None:
    db_url = _alembic_url(postgres)
    result = _run_alembic(["upgrade", "head"], db_url)
    assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}\n{result.stdout}"

    async def check() -> None:
        conn = await asyncpg.connect(db_url)
        try:
            # The vault crypto stores every encrypted value as a
            # ``<name>ciphertext`` + ``<name>nonce`` pair, so a ``ciphertext``
            # column is the reliable schema-level marker of an encrypted column.
            rows = await conn.fetch(
                "SELECT table_name, column_name FROM information_schema.columns "
                "WHERE table_schema = 'public' AND column_name LIKE '%ciphertext%'"
            )
        finally:
            await conn.close()

        schema_cols = {(r["table_name"], r["column_name"]) for r in rows}
        rekey_cols = {(c.table, c.ciphertext) for c in _REKEY_COLUMNS}
        missing = schema_cols - rekey_cols
        assert not missing, (
            "encrypted columns present in the schema but NOT re-encrypted by "
            "`aios rekey` — a vault-key rotation would orphan these rows under "
            f"the old key (permanent data loss): {sorted(missing)}. "
            "Add an `_EncryptedColumn` entry for each to ops._REKEY_COLUMNS."
        )

    asyncio.run(check())
