"""Operator subcommands: ``api``, ``worker``, ``migrate``.

These are lifted almost verbatim from the old ``__main__.py`` implementation.
They start long-running processes (uvicorn, procrastinate worker) or run
migrations — they do NOT talk to the HTTP API.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import asyncpg
import typer

from aios.crypto.vault import CryptoBox, EncryptedBlob
from aios.errors import CryptoDecryptError


def _run_api() -> int:
    import uvicorn

    from aios.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "aios.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        log_config=None,  # we configure structlog ourselves
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
    return 0


def _run_worker() -> int:
    from aios.harness.worker import worker_main
    from aios.logging import get_logger

    try:
        asyncio.run(worker_main())
    except KeyboardInterrupt:
        pass
    except SystemExit:
        raise
    except BaseException:
        get_logger("aios.worker").exception("worker.unexpected_exit")
        raise
    return 0


@dataclass(frozen=True, slots=True)
class _EncryptedColumn:
    table: str
    ciphertext: str
    nonce: str
    where: str
    account_expr: str = "account_id"


_REKEY_COLUMNS = (
    _EncryptedColumn("vault_credentials", "ciphertext", "nonce", "archived_at IS NULL"),
    _EncryptedColumn(
        "connections", "secrets_ciphertext", "secrets_nonce", "secrets_ciphertext IS NOT NULL"
    ),
    _EncryptedColumn("session_github_repositories", "ciphertext", "nonce", "TRUE"),
    _EncryptedColumn(
        "accounts",
        "placeholder_salt_ciphertext",
        "placeholder_salt_nonce",
        "placeholder_salt_ciphertext IS NOT NULL",
        "id",
    ),
)


def _decrypt_with_current_then_previous(
    current: CryptoBox, previous: CryptoBox | None, account_id: str, blob: EncryptedBlob
) -> str:
    for box in (current, previous):
        if box is None:
            continue
        try:
            return box.derive_account_subkey(account_id).decrypt(blob)
        except CryptoDecryptError:
            if box is current and previous is not None:
                continue
            raise
    raise RuntimeError("unreachable")


async def _run_rekey_async() -> int:
    from aios.config import get_settings
    from aios.db.pool import create_pool

    settings = get_settings()
    previous_secret = settings.vault_key_previous
    if previous_secret is None:
        raise RuntimeError("AIOS_VAULT_KEY_PREVIOUS must be set while running `aios rekey`")
    current = CryptoBox.from_base64(settings.vault_key.get_secret_value())
    previous = CryptoBox.from_base64(
        previous_secret.get_secret_value(), env_name="AIOS_VAULT_KEY_PREVIOUS"
    )
    pool = await create_pool(settings.db_url, min_size=1, max_size=1)
    total = 0
    try:
        async with pool.acquire() as conn, conn.transaction():
            for col in _REKEY_COLUMNS:
                total += await _rekey_column(conn, col, current, previous)
    finally:
        await pool.close()
    typer.echo(f"aios rekey re-encrypted {total} encrypted rows with AIOS_VAULT_KEY")
    return 0


async def _rekey_column(
    conn: asyncpg.Connection[Any], col: _EncryptedColumn, current: CryptoBox, previous: CryptoBox
) -> int:
    rows = await conn.fetch(
        f"SELECT id, {col.account_expr} AS account_id, {col.ciphertext} AS ct, {col.nonce} AS nn "
        f"FROM {col.table} WHERE {col.where}"
    )
    for row in rows:
        plaintext = _decrypt_with_current_then_previous(
            current, previous, row["account_id"], EncryptedBlob(row["ct"], row["nn"])
        )
        blob = current.derive_account_subkey(row["account_id"]).encrypt(plaintext)
        await conn.execute(
            f"UPDATE {col.table} SET {col.ciphertext} = $1, {col.nonce} = $2 WHERE id = $3",
            blob.ciphertext,
            blob.nonce,
            row["id"],
        )
    return len(rows)


def _run_rekey() -> int:
    return asyncio.run(_run_rekey_async())


def _run_migrate() -> int:
    from aios.config import get_settings
    from aios.db.migrations import apply_procrastinate_schema, upgrade_to_head

    db_url = get_settings().db_url
    upgrade_to_head(db_url)
    asyncio.run(apply_procrastinate_schema(db_url, verbose=True))
    return 0


def register(app: typer.Typer) -> None:
    """Attach the operator commands to the root app."""

    @app.command("api", help="Run the aios HTTP API server (uvicorn).")
    def api() -> None:
        raise typer.Exit(_run_api())

    @app.command("worker", help="Run the aios worker (procrastinate).")
    def worker() -> None:
        raise typer.Exit(_run_worker())

    @app.command(
        "migrate",
        help="Apply alembic migrations and the procrastinate schema if missing.",
    )
    def migrate() -> None:
        raise typer.Exit(_run_migrate())

    @app.command("rekey", help="Re-encrypt encrypted rows after AIOS_VAULT_KEY rotation.")
    def rekey() -> None:
        raise typer.Exit(_run_rekey())
