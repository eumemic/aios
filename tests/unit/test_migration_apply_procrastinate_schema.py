"""Unit tests for ``apply_procrastinate_schema`` DSN normalization.

The ``aios migrate`` CLI and the session-scoped ``migrated_db_url`` test
fixture both feed ``settings.db_url`` verbatim into
:func:`apply_procrastinate_schema`. An operator may set ``AIOS_DB_URL`` with a
SQLAlchemy driver suffix (``postgresql+asyncpg://`` / ``postgresql+psycopg://``),
which the rest of ``aios migrate`` tolerates — ``_migration_admission`` rewrites
it via ``_sync_db_url`` and ``migrations/env.py`` rewrites it again for Alembic —
so the suffixed URL survives all the way to this helper.

asyncpg rejects the suffix at *DSN parse time* (``ClientConfigurationError``)
before opening a connection, and procrastinate's psycopg ``PsycopgConnector``
rejects it on ``open_async()``. The helper must therefore normalize the URL via
:func:`aios.db.pool.normalize_dsn` first — mirroring ``aios.db.pool.create_pool``
and ``aios.jobs.app._sync_dsn``. Without normalization, ``aios migrate`` commits
every Alembic migration and then crashes before creating ``procrastinate_jobs``
and the lock-release trigger, blocking worker startup (introduced by 3704f9c8).

These tests mock the I/O layer (``asyncpg.connect``, ``procrastinate.App`` /
``PsycopgConnector``) so no Postgres is required; the real ``normalize_dsn`` runs
end-to-end, making the assertions exercise the actual fix.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.db.migrations import apply_procrastinate_schema


def _fake_conn(*, present: object) -> MagicMock:
    """A stand-in asyncpg connection whose ``fetchval`` reports schema presence.

    ``present`` is what ``SELECT to_regclass('procrastinate_jobs')`` returns:
    ``None`` → schema absent (apply path); anything non-None → idempotent skip.
    """
    conn = MagicMock()
    conn.fetchval = AsyncMock(return_value=present)
    # Catalog probe for the lock-release trigger: nothing installed → DDL runs.
    conn.fetchrow = AsyncMock(
        return_value={
            "functiondef": None,
            "triggerdef": None,
            "function_schema": None,
            "table_schema": None,
        }
    )
    conn.execute = AsyncMock()
    conn.close = AsyncMock()
    conn.transaction = MagicMock(return_value=AsyncMock())
    return conn


def _ddl_calls(conn: MagicMock) -> list[str]:
    from aios.db.procrastinate_extensions import LOCK_RELEASE_TRIGGER_DDL

    return [
        c.args[0] for c in conn.execute.await_args_list if c.args[0] == LOCK_RELEASE_TRIGGER_DDL
    ]


_SUFFIXED_ASYNCPG = "postgresql+asyncpg://u:p@host:5432/db"
_SUFFIXED_PSYCOPG = "postgresql+psycopg://u:p@host:5432/db"
_BARE = "postgresql://u:p@host:5432/db"


@pytest.mark.asyncio
async def test_strips_asyncpg_suffix_before_asyncpg_connect() -> None:
    """A ``postgresql+asyncpg://`` URL reaches asyncpg.connect as bare scheme.

    Without the fix, asyncpg raises ``ClientConfigurationError`` on the raw
    suffixed URL before any DB work — the reported ``aios migrate`` crash.
    """
    conn = _fake_conn(present="procrastinate_jobs")
    with patch("asyncpg.connect", new=AsyncMock(return_value=conn)) as mock_connect:
        await apply_procrastinate_schema(_SUFFIXED_ASYNCPG)

    mock_connect.assert_awaited_once_with(_BARE)


@pytest.mark.asyncio
async def test_strips_psycopg_suffix_before_asyncpg_connect() -> None:
    """A ``postgresql+psycopg://`` URL reaches asyncpg.connect as bare scheme."""
    conn = _fake_conn(present="procrastinate_jobs")
    with patch("asyncpg.connect", new=AsyncMock(return_value=conn)) as mock_connect:
        await apply_procrastinate_schema(_SUFFIXED_PSYCOPG)

    mock_connect.assert_awaited_once_with(_BARE)


@pytest.mark.asyncio
async def test_bare_url_passes_through_unchanged() -> None:
    """A bare ``postgresql://`` URL is a no-op for normalize_dsn — no regression.

    Every existing deployment and test pins the bare form, so the fix must not
    alter the happy path. ``normalize_dsn`` is idempotent on already-bare URLs.
    """
    conn = _fake_conn(present="procrastinate_jobs")
    with patch("asyncpg.connect", new=AsyncMock(return_value=conn)) as mock_connect:
        await apply_procrastinate_schema(_BARE)

    mock_connect.assert_awaited_once_with(_BARE)


@pytest.mark.asyncio
async def test_normalizes_conninfo_for_psycopg_connector_when_schema_missing() -> None:
    """The latent defect: ``PsycopgConnector(conninfo=...)`` must also be normalized.

    When ``to_regclass('procrastinate_jobs')`` is NULL the helper builds a
    throwaway procrastinate ``App`` over a ``PsycopgConnector``. The connector's
    ``conninfo`` is the same normalized local ``db_url`` as the asyncpg connect,
    so it must be the bare scheme — otherwise ``open_async()`` would fail at
    libpq conninfo parse time once asyncpg stopped failing first.
    """
    conn = _fake_conn(present=None)

    captured: dict[str, Any] = {}

    def _capture_connector(**kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return MagicMock()

    tmp_app = AsyncMock()
    with (
        patch("asyncpg.connect", new=AsyncMock(return_value=conn)) as mock_connect,
        patch("procrastinate.PsycopgConnector", side_effect=_capture_connector),
        patch("procrastinate.App", return_value=tmp_app) as mock_app,
    ):
        await apply_procrastinate_schema(_SUFFIXED_ASYNCPG)

    # asyncpg got the bare scheme ...
    mock_connect.assert_awaited_once_with(_BARE)
    # ... and so did the PsycopgConnector conninfo.
    assert captured == {"conninfo": _BARE}, captured
    # The throwaway App opened, applied the schema, and closed.
    mock_app.assert_called_once()
    tmp_app.open_async.assert_awaited_once()
    tmp_app.schema_manager.apply_schema_async.assert_awaited_once()
    tmp_app.close_async.assert_awaited_once()
    # Fresh schema → the lock-release trigger is missing, so its DDL runs.
    assert len(_ddl_calls(conn)) == 1
    conn.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_idempotent_path_skips_app_and_still_runs_trigger_ddl() -> None:
    """When the schema already exists, no App/connector is built; trigger DDL runs.

    The ``to_regclass`` guard is the idempotency mechanism (``apply_schema_async``
    has no ``IF NOT EXISTS``). This confirms a re-run after a successful migrate
    — including the documented recovery (correct ``AIOS_DB_URL`` then re-run) —
    re-enters the apply guard and completes without touching the connector.
    """
    conn = _fake_conn(present="procrastinate_jobs")
    with (
        patch("asyncpg.connect", new=AsyncMock(return_value=conn)) as mock_connect,
        patch("procrastinate.App") as mock_app,
        patch("procrastinate.PsycopgConnector") as mock_connector,
    ):
        await apply_procrastinate_schema(_SUFFIXED_ASYNCPG, verbose=True)

    mock_connect.assert_awaited_once_with(_BARE)
    mock_app.assert_not_called()
    mock_connector.assert_not_called()
    assert len(_ddl_calls(conn)) == 1  # trigger absent in the probe → DDL runs
    conn.close.assert_awaited_once()


# --- #2539: never take AccessExclusive on procrastinate_jobs needlessly -------


def _current_state(schema: str = "public") -> dict[str, str]:
    from aios.db.procrastinate_extensions import expected_functiondef, expected_triggerdef

    return {
        "functiondef": expected_functiondef(schema),
        "triggerdef": expected_triggerdef(schema),
        "function_schema": schema,
        "table_schema": schema,
    }


@pytest.mark.asyncio
async def test_current_trigger_skips_ddl_entirely() -> None:
    """An unchanged DB takes no lock: neither lock_timeout nor DDL is issued."""
    conn = _fake_conn(present="procrastinate_jobs")
    conn.fetchrow = AsyncMock(return_value=_current_state())
    with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
        await apply_procrastinate_schema(_BARE, verbose=True)

    conn.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_drifted_trigger_runs_ddl_under_lock_timeout() -> None:
    """A drifted function body re-applies the DDL, preceded by SET LOCAL lock_timeout."""
    from aios.db.procrastinate_extensions import LOCK_RELEASE_TRIGGER_DDL

    state = _current_state()
    state["functiondef"] = state["functiondef"].replace("RETURN NEW", "RETURN NULL")
    conn = _fake_conn(present="procrastinate_jobs")
    conn.fetchrow = AsyncMock(return_value=state)
    with patch("asyncpg.connect", new=AsyncMock(return_value=conn)):
        await apply_procrastinate_schema(_BARE)

    statements = [c.args[0] for c in conn.execute.await_args_list]
    # set_config(..., is_local => true) is SET LOCAL: scoped to the DDL's transaction.
    assert conn.execute.await_args_list[0].args == (
        "SELECT set_config('lock_timeout', $1, true)",
        "5s",
    )
    assert statements[1] == LOCK_RELEASE_TRIGGER_DDL


@pytest.mark.asyncio
async def test_lock_timeout_retries_then_succeeds() -> None:
    import asyncpg

    from aios.db.procrastinate_extensions import LOCK_RELEASE_TRIGGER_DDL

    conn = _fake_conn(present="procrastinate_jobs")
    failures = iter([asyncpg.exceptions.LockNotAvailableError("lock timeout")] * 2)

    async def _execute(sql: str, *args: object) -> None:
        if sql == LOCK_RELEASE_TRIGGER_DDL:
            err = next(failures, None)
            if err is not None:
                raise err

    conn.execute = AsyncMock(side_effect=_execute)
    with (
        patch("asyncpg.connect", new=AsyncMock(return_value=conn)),
        patch("asyncio.sleep", new=AsyncMock()) as sleep,
    ):
        await apply_procrastinate_schema(_BARE, max_attempts=5, retry_sleep_seconds=0.25)

    assert len(_ddl_calls(conn)) == 3
    assert sleep.await_count == 2
    sleep.assert_awaited_with(0.25)
    conn.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_lock_timeout_exhaustion_fails_instead_of_waiting() -> None:
    """Exhausted retries raise (failing ``aios migrate``) rather than stalling the queue."""
    import asyncpg

    from aios.db.procrastinate_extensions import LOCK_RELEASE_TRIGGER_DDL

    conn = _fake_conn(present="procrastinate_jobs")

    async def _execute(sql: str, *args: object) -> None:
        if sql == LOCK_RELEASE_TRIGGER_DDL:
            raise asyncpg.exceptions.LockNotAvailableError("lock timeout")

    conn.execute = AsyncMock(side_effect=_execute)
    with (
        patch("asyncpg.connect", new=AsyncMock(return_value=conn)),
        patch("asyncio.sleep", new=AsyncMock()) as sleep,
        pytest.raises(asyncpg.exceptions.LockNotAvailableError),
    ):
        await apply_procrastinate_schema(_BARE, max_attempts=3, retry_sleep_seconds=0)

    assert len(_ddl_calls(conn)) == 3
    assert sleep.await_count == 2
    conn.close.assert_awaited_once()


@pytest.mark.parametrize(
    "field",
    ["functiondef", "triggerdef", "function_schema", "table_schema"],
)
def test_is_current_false_when_any_piece_missing(field: str) -> None:
    from aios.db.procrastinate_extensions import lock_release_trigger_is_current

    state: dict[str, str | None] = dict(_current_state())
    assert lock_release_trigger_is_current(**state)
    state[field] = None
    assert not lock_release_trigger_is_current(**state)


def test_is_current_false_when_trigger_definition_differs() -> None:
    from aios.db.procrastinate_extensions import lock_release_trigger_is_current

    state: dict[str, str | None] = dict(_current_state())
    state["triggerdef"] = str(state["triggerdef"]).replace("AFTER UPDATE", "BEFORE UPDATE")
    assert not lock_release_trigger_is_current(**state)
