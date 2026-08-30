"""Tests for operator subcommands (api, worker, migrate)."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from psycopg.errors import LockNotAvailable
from sqlalchemy.exc import OperationalError
from typer.testing import CliRunner

from aios.cli.app import app
from aios.cli.commands.ops import _run_migrate

runner = CliRunner()


def test_api_command_passes_proxy_headers_to_uvicorn(monkeypatch):
    """The `api` command must pass proxy_headers=True and forwarded_allow_ips="*"
    to uvicorn.run so that X-Forwarded-For headers are trusted when running
    behind a reverse proxy."""
    monkeypatch.setenv("AIOS_API_KEY", "test")
    monkeypatch.setenv("AIOS_DB_URL", "postgresql://localhost/test")
    # The shared-DB guard fires from a linked worktree (the mandated dev
    # workflow) because db name "test" is not aios_dev_*. Disable it here so we
    # exercise uvicorn wiring, not the guard. (CI runs from a main checkout
    # where the guard is a no-op, so it would never catch this regression.)
    monkeypatch.setattr("aios.cli.commands.dev.is_linked_worktree", lambda: False)

    with patch("uvicorn.run") as mock_run:
        runner.invoke(app, ["api"])

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs.get("proxy_headers") is True, "uvicorn.run must be called with proxy_headers=True"
    assert kwargs.get("forwarded_allow_ips") == "*", (
        'uvicorn.run must be called with forwarded_allow_ips="*"'
    )


def test_api_command_aborts_before_uvicorn_on_shared_worktree(monkeypatch):
    """`api` from a linked worktree on the shared DB aborts before uvicorn.run."""
    monkeypatch.setenv("AIOS_API_KEY", "test")
    monkeypatch.setenv("AIOS_DB_URL", "postgresql://localhost/aios")
    monkeypatch.delenv("AIOS_ALLOW_SHARED_DB", raising=False)
    monkeypatch.setattr("aios.cli.commands.dev.is_linked_worktree", lambda: True)

    with patch("uvicorn.run") as mock_run:
        result = runner.invoke(app, ["api"])

    mock_run.assert_not_called()
    assert result.exit_code == 1


def test_worker_command_aborts_before_worker_main_on_shared_worktree(monkeypatch):
    """`worker` from a linked worktree on the shared DB aborts before worker_main."""
    monkeypatch.setenv("AIOS_API_KEY", "test")
    monkeypatch.setenv("AIOS_DB_URL", "postgresql://localhost/aios")
    monkeypatch.delenv("AIOS_ALLOW_SHARED_DB", raising=False)
    monkeypatch.setattr("aios.cli.commands.dev.is_linked_worktree", lambda: True)

    with patch("aios.harness.worker.worker_main") as mock_worker_main:
        result = runner.invoke(app, ["worker"])

    mock_worker_main.assert_not_called()
    assert result.exit_code == 1


def test_migrate_configures_logging_before_running_migrations(monkeypatch):
    """`migrate` must call configure_logging BEFORE upgrade_to_head so that
    migration-emitted audit records (e.g. the 0130 auto-disable WARNING) are
    visible on the prod path. Without this, `import alembic`'s NullHandler on
    the 'alembic' parent logger satisfies the handler-search and shadows
    logging.lastResort, silently swallowing every migration warning (#1678 F1).
    Assert both that configure_logging is called and that it precedes the
    migration run (order matters — a migration warning emitted before logging
    is configured would still be lost)."""
    monkeypatch.setenv("AIOS_API_KEY", "test")
    monkeypatch.setenv("AIOS_DB_URL", "postgresql://localhost/test")

    # Attach every collaborator to one parent so mock_calls records global order.
    parent = Mock()
    parent.get_settings.return_value = SimpleNamespace(
        db_url="postgresql://localhost/test", log_level="INFO"
    )
    parent.apply_procrastinate_schema = AsyncMock()

    with (
        patch("aios.config.get_settings", parent.get_settings),
        patch("aios.logging.configure_logging", parent.configure_logging),
        patch("aios.db.migrations.upgrade_to_head", parent.upgrade_to_head),
        patch("aios.db.migrations.apply_procrastinate_schema", parent.apply_procrastinate_schema),
        patch("aios.db.migrations._migration_admission", return_value=nullcontext(True)),
    ):
        result = runner.invoke(app, ["migrate"])

    assert result.exit_code == 0, result.output
    parent.configure_logging.assert_called_once_with("INFO")
    parent.upgrade_to_head.assert_called_once_with("postgresql://localhost/test")
    # configure_logging must run before the migration.
    called = [c[0] for c in parent.mock_calls]
    assert called.index("configure_logging") < called.index("upgrade_to_head"), (
        "configure_logging must be called before upgrade_to_head"
    )


def _migrate_patches(upgrade_to_head: Mock) -> tuple[Any, ...]:
    settings = SimpleNamespace(db_url="postgresql://localhost/test", log_level="INFO")
    return (
        patch("aios.config.get_settings", return_value=settings),
        patch("aios.logging.configure_logging"),
        patch("aios.logging.get_logger"),
        patch("aios.db.migrations.upgrade_to_head", upgrade_to_head),
        patch("aios.db.migrations.apply_procrastinate_schema", new_callable=AsyncMock),
        patch("aios.db.migrations._migration_admission", return_value=nullcontext(True)),
        patch("time.sleep"),
    )


def test_migrate_skips_alembic_for_rollback_image() -> None:
    upgrade_to_head = Mock()
    patches = _migrate_patches(upgrade_to_head)
    patches = (
        *patches[:5],
        patch("aios.db.migrations._migration_admission", return_value=nullcontext(False)),
        patches[6],
    )

    with (
        patches[0],
        patches[1],
        patches[2] as logger,
        patches[3],
        patches[4] as procrastinate,
        patches[5],
        patches[6],
    ):
        assert _run_migrate() == 0

    upgrade_to_head.assert_not_called()
    procrastinate.assert_not_awaited()
    logger.return_value.info.assert_called_once_with("migration.rollback_image_admitted")


def test_migrate_retries_lock_errors_then_succeeds():
    lock_error = LockNotAvailable("lock timeout")
    wrapped_lock_error = OperationalError("ALTER TABLE events", {}, lock_error)
    upgrade_to_head = Mock(side_effect=[lock_error, wrapped_lock_error, None])

    patches = _migrate_patches(upgrade_to_head)
    with (
        patches[0],
        patches[1],
        patches[2] as logger,
        patches[3],
        patches[4],
        patches[5],
        patches[6] as sleep,
    ):
        assert _run_migrate() == 0

    assert upgrade_to_head.call_count == 3
    assert sleep.call_count == 2
    assert logger.return_value.warning.call_count == 2


def test_migrate_does_not_retry_non_lock_error():
    migration_error = RuntimeError("broken migration")
    upgrade_to_head = Mock(side_effect=migration_error)

    patches = _migrate_patches(upgrade_to_head)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6] as sleep,
        pytest.raises(RuntimeError, match="broken migration") as raised,
    ):
        _run_migrate()

    assert raised.value is migration_error
    upgrade_to_head.assert_called_once_with("postgresql://localhost/test")
    sleep.assert_not_called()


def test_migrate_command_exits_nonzero_when_lock_retries_exhausted():
    upgrade_to_head = Mock(side_effect=LockNotAvailable("lock timeout"))
    patches = _migrate_patches(upgrade_to_head)

    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
        result = runner.invoke(app, ["migrate"])

    assert result.exit_code != 0
    assert isinstance(result.exception, LockNotAvailable)


def test_migrate_propagates_last_lock_error_after_attempts_exhausted():
    lock_errors = [LockNotAvailable(f"lock timeout {attempt}") for attempt in range(10)]
    upgrade_to_head = Mock(side_effect=lock_errors)

    patches = _migrate_patches(upgrade_to_head)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6] as sleep,
        pytest.raises(LockNotAvailable) as raised,
    ):
        _run_migrate()

    assert raised.value is lock_errors[-1]
    assert upgrade_to_head.call_count == 10
    assert sleep.call_count == 9
