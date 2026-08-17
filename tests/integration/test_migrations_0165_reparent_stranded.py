"""Migration 0165 repairs databases that 0154 already stranded.

0154 ran in production on 2026-07-31 and alembic will never re-run it, so
correcting 0154 in place only helps *fresh* databases.  These checks exercise
the case that matters: a database migrated by the **original** 0154 (the
pre-fix text, reconstructed here) still carries the defect, and 0165 corrects
it.

The four properties, each derived from the live account tree rather than from
a fixture list:

* **red/green** -- pre-existing root children are stranded after the original
  0154, and resolve again after 0165;
* **idempotence** -- a second 0165 run moves nothing and changes nothing;
* **positive control** -- a database that never had the defect (fresh DB
  migrated by the corrected 0154) is left byte-identical by 0165;
* **guard** -- 0165 declines to touch an account tree with no
  migration-owned Eumemic child.
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import asyncpg
import pytest
from nacl.secret import SecretBox

from tests.conftest import _docker_available, needs_docker
from tests.integration.test_migrations import PROJECT_ROOT, _alembic_url

pytestmark = pytest.mark.integration

_MIGRATION_0154 = PROJECT_ROOT / "migrations" / "versions" / "0154_rehome_fleet_to_eumemic_child.py"


@pytest.fixture
def postgres() -> Iterator[Any]:
    if not _docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


def _run_alembic(args: list[str], db_url: str, key: bytes) -> subprocess.CompletedProcess[str]:
    uv = shutil.which("uv")
    if uv is None:
        raise FileNotFoundError("uv not found on PATH")
    return subprocess.run(
        [uv, "run", "alembic", *args],
        cwd=PROJECT_ROOT,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin"),
            "HOME": str(Path.home()),
            "AIOS_DB_URL": db_url,
            "AIOS_VAULT_KEY": base64.b64encode(key).decode(),
        },
        capture_output=True,
        text=True,
        check=False,
    )


def _original_0154_text() -> str:
    """Reconstruct the 0154 that production actually ran.

    The fix added three call sites to 0154's ``upgrade``/``downgrade``: the
    reparenting itself plus the snapshot/assert pair that guards it.  Removing
    all of them reproduces the pre-fix behaviour.  Stripping only the
    reparenting call is NOT enough -- the retained assertion then aborts the
    migration and the "already-migrated database" can never be built.

    The strict count assertions below fail loudly if the fix is ever reshaped,
    so this test can never silently degrade into "run the fixed migration
    twice" -- which is exactly the test that would hide the production bug.
    ``test_prefix_reconstruction_has_no_reparenting`` pins the result.
    """
    text = _MIGRATION_0154.read_text()
    for pattern, expected in (
        (r"^ *_reparent_children\([^)]*\)\n", 2),
        (r"^ *_snapshot_provider_resolution\([^)]*\)\n", 1),
        (r"^ *_assert_provider_resolution_preserved\([^)]*\)\n", 1),
    ):
        text, count = re.subn(pattern, "", text, flags=re.MULTILINE)
        assert count == expected, (
            f"expected {expected} call site(s) matching {pattern!r} in 0154, found {count}; "
            "update this reconstruction of the pre-fix migration"
        )
    return text


class _PreFix0154:
    """Swap 0154's file for its pre-fix text for the duration of the block."""

    def __enter__(self) -> _PreFix0154:
        self._original = _MIGRATION_0154.read_text()
        _MIGRATION_0154.write_text(_original_0154_text())
        return self

    def __exit__(self, *exc: object) -> None:
        _MIGRATION_0154.write_text(self._original)


async def _seed(db_url: str, key: bytes) -> None:
    """A root with pre-existing children and a root-owned provider."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("m0154_seed", _MIGRATION_0154)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ciphertext, nonce = module._Box(key).account("acc_root").encrypt("secret:model_providers")

    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(
            "INSERT INTO accounts (id,display_name,can_mint_children,spent_microusd) "
            "VALUES ('acc_root','root',true,4242)"
        )
        await conn.execute(
            "INSERT INTO accounts (id,parent_account_id,display_name,can_mint_children) VALUES "
            "('acc_tenantA','acc_root','Tenant A',true),"
            "('acc_tenantB','acc_root','Tenant B',true)"
        )
        await conn.execute(
            "INSERT INTO accounts (id,parent_account_id,display_name) "
            "VALUES ('acc_grandchild','acc_tenantA','Grandchild A1')"
        )
        await conn.execute(
            "INSERT INTO account_keys (key_id,account_id,hash,label) VALUES "
            "('key_keep','acc_root',$1,'admin'),('key_move','acc_root',$2,'fleet')",
            os.urandom(32),
            os.urandom(32),
        )
        await conn.execute(
            "INSERT INTO model_providers (id,account_id,provider,ciphertext,nonce) "
            "VALUES ('mp_root','acc_root','test',$1,$2)",
            ciphertext,
            nonce,
        )
    finally:
        await conn.close()


async def _resolving_accounts(db_url: str) -> set[str]:
    """Accounts that can reach a live provider by nearest-ancestor walk."""
    conn = await asyncpg.connect(db_url)
    try:
        rows = await conn.fetch(
            """
            WITH RECURSIVE ancestry(account_id, ancestor_id) AS (
                SELECT id, id FROM accounts WHERE archived_at IS NULL
                UNION ALL
                SELECT ancestry.account_id, accounts.parent_account_id
                FROM ancestry
                JOIN accounts ON accounts.id = ancestry.ancestor_id
                WHERE accounts.parent_account_id IS NOT NULL
            )
            SELECT DISTINCT ancestry.account_id
            FROM ancestry
            JOIN model_providers ON model_providers.account_id = ancestry.ancestor_id
            WHERE model_providers.archived_at IS NULL
            """
        )
        return {row["account_id"] for row in rows}
    finally:
        await conn.close()


async def _account_tree(db_url: str) -> list[tuple[str, str | None, str]]:
    conn = await asyncpg.connect(db_url)
    try:
        rows = await conn.fetch(
            "SELECT id,parent_account_id,display_name FROM accounts ORDER BY id"
        )
        return [(r["id"], r["parent_account_id"], r["display_name"]) for r in rows]
    finally:
        await conn.close()


@needs_docker
def test_0165_repairs_a_database_the_original_0154_stranded(postgres: Any) -> None:
    """RED then GREEN on the state production is actually in."""
    db_url = _alembic_url(postgres)
    key = os.urandom(SecretBox.KEY_SIZE)

    assert _run_alembic(["upgrade", "0152"], db_url, key).returncode == 0
    asyncio.run(_seed(db_url, key))
    before = asyncio.run(_resolving_accounts(db_url))
    assert {"acc_tenantA", "acc_tenantB", "acc_grandchild"} <= before

    # Apply the migration production actually ran, then stop short of 0165.
    with _PreFix0154():
        result = _run_alembic(["upgrade", "0159"], db_url, key)
        assert result.returncode == 0, result.stderr

    # RED: exactly the accounts the issue names have lost provider resolution.
    stranded = asyncio.run(_resolving_accounts(db_url))
    assert {"acc_tenantA", "acc_tenantB", "acc_grandchild"} & stranded == set(), (
        "expected the pre-fix 0154 to strand root's pre-existing children; "
        f"still resolving: {sorted(stranded)}"
    )

    # A direct-root account created after 0154 is legitimate, not stranded.
    async def _add_legitimate_post_cutover_account() -> None:
        conn = await asyncpg.connect(db_url)
        try:
            await conn.execute(
                "INSERT INTO accounts (id,parent_account_id,display_name) "
                "VALUES ('acc_legitimate_root','acc_root','Legitimate root child')"
            )
        finally:
            await conn.close()

    asyncio.run(_add_legitimate_post_cutover_account())

    # GREEN: the forward migration restores only accounts present at cutover.
    result = _run_alembic(["upgrade", "0165"], db_url, key)
    assert result.returncode == 0, result.stderr
    repaired = asyncio.run(_resolving_accounts(db_url))
    assert before - {"acc_root"} <= repaired, (
        f"0165 failed to restore resolution; missing: {sorted(before - repaired - {'acc_root'})}"
    )

    async def _check_topology() -> None:
        conn = await asyncpg.connect(db_url)
        try:
            child = await conn.fetchval(
                "SELECT id FROM accounts WHERE display_name='Eumemic' "
                "AND parent_account_id='acc_root'"
            )
            assert child
            for stranded_id in ("acc_tenantA", "acc_tenantB"):
                assert (
                    await conn.fetchval(
                        "SELECT parent_account_id FROM accounts WHERE id=$1", stranded_id
                    )
                    == child
                )
            assert (
                await conn.fetchval(
                    "SELECT parent_account_id FROM accounts WHERE id='acc_legitimate_root'"
                )
                == "acc_root"
            )
            # Deeper descendants keep their own parent -- only root's direct
            # children move.
            assert (
                await conn.fetchval(
                    "SELECT parent_account_id FROM accounts WHERE id='acc_grandchild'"
                )
                == "acc_tenantA"
            )
        finally:
            await conn.close()

    asyncio.run(_check_topology())

    # The repair's downgrade is deliberately non-destructive; a full round
    # trip must preserve both the repaired and legitimate placements.
    result = _run_alembic(["downgrade", "0159"], db_url, key)
    assert result.returncode == 0, result.stderr
    asyncio.run(_check_topology())


@needs_docker
def test_0165_is_idempotent(postgres: Any) -> None:
    """Re-running the forward migration changes nothing."""
    db_url = _alembic_url(postgres)
    key = os.urandom(SecretBox.KEY_SIZE)

    assert _run_alembic(["upgrade", "0152"], db_url, key).returncode == 0
    asyncio.run(_seed(db_url, key))
    with _PreFix0154():
        assert _run_alembic(["upgrade", "0159"], db_url, key).returncode == 0
    assert _run_alembic(["upgrade", "0165"], db_url, key).returncode == 0

    tree_once = asyncio.run(_account_tree(db_url))
    resolved_once = asyncio.run(_resolving_accounts(db_url))

    # Re-run the same forward migration against the already-repaired database.
    assert _run_alembic(["downgrade", "0159"], db_url, key).returncode == 0
    result = _run_alembic(["upgrade", "0165"], db_url, key)
    assert result.returncode == 0, result.stderr

    assert asyncio.run(_account_tree(db_url)) == tree_once
    assert asyncio.run(_resolving_accounts(db_url)) == resolved_once


@needs_docker
def test_0165_leaves_a_never_defective_database_untouched(postgres: Any) -> None:
    """Positive control: a fresh DB migrated by the corrected 0154."""
    db_url = _alembic_url(postgres)
    key = os.urandom(SecretBox.KEY_SIZE)

    assert _run_alembic(["upgrade", "0152"], db_url, key).returncode == 0
    asyncio.run(_seed(db_url, key))
    # The corrected 0154 (working tree) already reparents; stop before 0165.
    assert _run_alembic(["upgrade", "0159"], db_url, key).returncode == 0

    tree_before = asyncio.run(_account_tree(db_url))
    resolved_before = asyncio.run(_resolving_accounts(db_url))

    result = _run_alembic(["upgrade", "0165"], db_url, key)
    assert result.returncode == 0, result.stderr

    assert asyncio.run(_account_tree(db_url)) == tree_before
    assert asyncio.run(_resolving_accounts(db_url)) == resolved_before


@needs_docker
def test_0165_noops_without_a_migration_owned_eumemic_child(postgres: Any) -> None:
    """An operator-created account merely named Eumemic is not the marker."""
    db_url = _alembic_url(postgres)
    key = os.urandom(SecretBox.KEY_SIZE)

    assert _run_alembic(["upgrade", "0159"], db_url, key).returncode == 0

    async def _seed_lookalike() -> None:
        conn = await asyncpg.connect(db_url)
        try:
            await conn.execute(
                "INSERT INTO accounts (id,display_name,can_mint_children) "
                "VALUES ('acc_root','root',true)"
            )
            await conn.execute(
                "INSERT INTO accounts (id,parent_account_id,display_name) VALUES "
                "('acc_lookalike','acc_root','Eumemic'),"
                "('acc_other','acc_root','Other')"
            )
        finally:
            await conn.close()

    asyncio.run(_seed_lookalike())
    tree_before = asyncio.run(_account_tree(db_url))

    result = _run_alembic(["upgrade", "0165"], db_url, key)
    assert result.returncode == 0, result.stderr
    assert asyncio.run(_account_tree(db_url)) == tree_before


def test_prefix_reconstruction_has_no_reparenting() -> None:
    """The reconstructed pre-fix 0154 must genuinely lack the fix.

    Guards the reconstruction itself: if this ever retained the reparenting
    (or the assertion that aborts without it), the "already-migrated
    database" these tests build would silently become a *fixed* database and
    the red test would stop being red.
    """
    text = _original_0154_text()
    upgrade_body = text.split("def upgrade()")[1].split("def downgrade()")[0]
    assert "_reparent_children" not in upgrade_body
    assert "_snapshot_provider_resolution" not in upgrade_body
    assert "_assert_provider_resolution_preserved" not in upgrade_body
    # ...while the shipped migration does have it (the fix is real).
    shipped = _MIGRATION_0154.read_text().split("def upgrade()")[1].split("def downgrade()")[0]
    assert "_reparent_children" in shipped
