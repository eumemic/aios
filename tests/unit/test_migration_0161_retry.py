"""Regression tests for the hot-table lock retry in migration 0161."""

from __future__ import annotations

import importlib.util
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest

_PATH = Path(__file__).resolve().parents[2] / "migrations/versions/0161_image_token_baseline_v2.py"
_SPEC = importlib.util.spec_from_file_location("migration_0161", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
migration = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migration)


class _Bind:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.alter_attempts = 0
        self.statements: list[str] = []

    def begin_nested(self) -> Any:
        return nullcontext()

    def exec_driver_sql(self, sql: str) -> None:
        self.statements.append(sql)
        if sql.startswith("ALTER TABLE events"):
            self.alter_attempts += 1
            if self.alter_attempts <= self.failures:
                raise RuntimeError("lock timeout")


def test_events_alter_retries_lock_timeout_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    bind = _Bind(failures=2)
    monkeypatch.setattr(migration.op, "get_bind", lambda: bind)
    monkeypatch.setattr(migration.op, "execute", lambda _sql: None)
    monkeypatch.setattr(migration.time, "sleep", lambda _seconds: None)

    migration.upgrade()

    assert bind.alter_attempts == 3
    assert sum(sql.startswith("SET LOCAL lock_timeout") for sql in bind.statements) == 3


def test_events_alter_fails_after_bounded_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    bind = _Bind(failures=migration._MAX_ATTEMPTS)
    monkeypatch.setattr(migration.op, "get_bind", lambda: bind)
    monkeypatch.setattr(migration.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="lock timeout"):
        migration.upgrade()

    assert bind.alter_attempts == migration._MAX_ATTEMPTS
