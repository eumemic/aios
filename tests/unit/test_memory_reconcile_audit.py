"""Unit tests for the uncorrelated memory-reconcile audit (#1748 §Uncorrelated detector).

Pure in-memory: the DB pool/connection are mocked directly (no Postgres), the
host directories live under ``tmp_path``. These tests exercise the audit's
independent walk + unconditional hash + DB-comparison logic — never the
prefilter's candidate classification, by design (the whole point of this
detector is that it does NOT reuse that code path).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.harness import memory_reconcile_audit as audit

STORE_A = "memstore_01AUDITSTOREA00000000000001"
ACCOUNT_A = "acct_01AUDITACCOUNT0000000000001"


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _fake_conn(
    *,
    store_rows: list[tuple[str, str]],
    db_content: dict[str, str],
) -> MagicMock:
    conn = MagicMock()

    async def _fetch(sql: str, *args: Any) -> list[dict[str, Any]]:
        assert "memory_stores" in sql
        return [{"id": sid, "account_id": aid} for sid, aid in store_rows]

    conn.fetch = AsyncMock(side_effect=_fetch)
    return conn


def _fake_pool(conn: MagicMock) -> MagicMock:
    pool = MagicMock()

    class _Cm:
        async def __aenter__(self) -> Any:
            return conn

        async def __aexit__(self, *_a: Any) -> None:
            return None

    pool.acquire.return_value = _Cm()
    return pool


@pytest.fixture(autouse=True)
def _patch_list_active(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[tuple[str, str]]]:
    """Patch ``list_active_memory_paths_and_content`` per-store via a mutable dict."""
    db_state: dict[str, list[tuple[str, str]]] = {}

    async def _fake_list_active(
        conn: Any, store_id: str, *, account_id: str
    ) -> list[tuple[str, str]]:
        return db_state.get(store_id, [])

    monkeypatch.setattr(audit.queries, "list_active_memory_paths_and_content", _fake_list_active)
    return db_state


def _seed_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    db_state: dict[str, list[tuple[str, str]]],
    *,
    store_id: str = STORE_A,
    disk_files: dict[str, str] | None = None,
    db_files: dict[str, str] | None = None,
    materialized: bool = True,
) -> Path:
    host_dir = tmp_path / store_id
    host_dir.mkdir(parents=True, exist_ok=True)
    if materialized:
        (host_dir / ".materialized").touch()
    for path, content in (disk_files or {}).items():
        target = host_dir / path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    db_state[store_id] = list((db_files or {}).items())

    monkeypatch.setattr(audit, "memory_store_host_dir", lambda sid, _host_dir=host_dir: _host_dir)
    return host_dir


class TestCleanAudit:
    async def test_matching_disk_and_db_produce_no_divergence(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        _patch_list_active: dict[str, list[tuple[str, str]]],
    ) -> None:
        content = "hello world\n"
        _seed_store(
            tmp_path,
            monkeypatch,
            _patch_list_active,
            disk_files={"/foo.md": content},
            db_files={"/foo.md": content},
        )
        conn = _fake_conn(store_rows=[(STORE_A, ACCOUNT_A)], db_content={})
        pool = _fake_pool(conn)

        result = await audit.run_memory_reconcile_audit(pool)

        assert result.clean
        assert result.stores_checked == 1
        assert result.files_hashed == 1


class TestDivergenceDetection:
    async def test_content_mismatch_detected(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        _patch_list_active: dict[str, list[tuple[str, str]]],
    ) -> None:
        """The core scenario this detector exists for: on-disk content differs
        from the DB's row — a prefilter false-negative would produce exactly
        this divergence, undetectably from inside the prefilter's own
        substrate. This audit's independent hash must catch it."""
        _seed_store(
            tmp_path,
            monkeypatch,
            _patch_list_active,
            disk_files={"/foo.md": "disk content\n"},
            db_files={"/foo.md": "stale db content\n"},
        )
        conn = _fake_conn(store_rows=[(STORE_A, ACCOUNT_A)], db_content={})
        pool = _fake_pool(conn)

        result = await audit.run_memory_reconcile_audit(pool)

        assert not result.clean
        assert len(result.divergences) == 1
        d = result.divergences[0]
        assert d.reason == "content_mismatch"
        assert d.store_id == STORE_A
        assert d.store_path == "/foo.md"
        assert d.disk_sha256 == _sha256("disk content\n")
        assert d.db_sha256 == _sha256("stale db content\n")

    async def test_missing_in_db_detected(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        _patch_list_active: dict[str, list[tuple[str, str]]],
    ) -> None:
        """A file present on disk but absent from the DB (e.g. a create that
        never durably landed) is a divergence, not silently ignored."""
        _seed_store(
            tmp_path,
            monkeypatch,
            _patch_list_active,
            disk_files={"/orphan.md": "orphaned content\n"},
            db_files={},
        )
        conn = _fake_conn(store_rows=[(STORE_A, ACCOUNT_A)], db_content={})
        pool = _fake_pool(conn)

        result = await audit.run_memory_reconcile_audit(pool)

        assert not result.clean
        assert result.divergences[0].reason == "missing_in_db"

    async def test_missing_on_disk_detected(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        _patch_list_active: dict[str, list[tuple[str, str]]],
    ) -> None:
        """A file the DB believes is live but which vanished from disk (e.g. a
        prefilter false-negative on a delete) is a divergence."""
        _seed_store(
            tmp_path,
            monkeypatch,
            _patch_list_active,
            disk_files={},
            db_files={"/ghost.md": "ghost content\n"},
        )
        conn = _fake_conn(store_rows=[(STORE_A, ACCOUNT_A)], db_content={})
        pool = _fake_pool(conn)

        result = await audit.run_memory_reconcile_audit(pool)

        assert not result.clean
        assert result.divergences[0].reason == "missing_on_disk"

    async def test_unreadable_file_detected(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        _patch_list_active: dict[str, list[tuple[str, str]]],
    ) -> None:
        host_dir = _seed_store(
            tmp_path,
            monkeypatch,
            _patch_list_active,
            disk_files={"/locked.md": "content\n"},
            db_files={"/locked.md": "content\n"},
        )
        real_read_bytes = Path.read_bytes
        target = host_dir / "locked.md"

        def _fake_read_bytes(self: Path) -> bytes:
            if self == target:
                raise OSError("permission denied")
            return real_read_bytes(self)

        monkeypatch.setattr(Path, "read_bytes", _fake_read_bytes)

        conn = _fake_conn(store_rows=[(STORE_A, ACCOUNT_A)], db_content={})
        pool = _fake_pool(conn)

        result = await audit.run_memory_reconcile_audit(pool)

        assert not result.clean
        assert result.divergences[0].reason == "unreadable"


class TestSkipsUnmaterializedOrMissingStores:
    async def test_store_without_host_dir_is_skipped(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        _patch_list_active: dict[str, list[tuple[str, str]]],
    ) -> None:
        missing_dir = tmp_path / "does_not_exist"
        monkeypatch.setattr(audit, "memory_store_host_dir", lambda sid: missing_dir)

        conn = _fake_conn(store_rows=[(STORE_A, ACCOUNT_A)], db_content={})
        pool = _fake_pool(conn)

        result = await audit.run_memory_reconcile_audit(pool)

        assert result.stores_checked == 0
        assert result.clean

    async def test_store_without_materialized_marker_is_skipped(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        _patch_list_active: dict[str, list[tuple[str, str]]],
    ) -> None:
        _seed_store(
            tmp_path,
            monkeypatch,
            _patch_list_active,
            disk_files={"/foo.md": "content\n"},
            db_files={},
            materialized=False,
        )
        conn = _fake_conn(store_rows=[(STORE_A, ACCOUNT_A)], db_content={})
        pool = _fake_pool(conn)

        result = await audit.run_memory_reconcile_audit(pool)

        assert result.stores_checked == 0
        assert result.clean


class TestSymlinkRejection:
    """The audit walks the same shared host directory bash writes into and
    must reject the same confused-deputy symlink pattern the reconcile
    prefilter rejects — even though the two walks are independent code."""

    async def test_symlink_never_hashed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        _patch_list_active: dict[str, list[tuple[str, str]]],
    ) -> None:
        sensitive = tmp_path / "host_secret.txt"
        sensitive.write_text("HOST-ONLY SECRET")

        host_dir = _seed_store(
            tmp_path,
            monkeypatch,
            _patch_list_active,
            disk_files={},
            db_files={},
        )
        (host_dir / "leak.txt").symlink_to(sensitive)

        conn = _fake_conn(store_rows=[(STORE_A, ACCOUNT_A)], db_content={})
        pool = _fake_pool(conn)

        result = await audit.run_memory_reconcile_audit(pool)

        # The symlink is invisible to the audit entirely (not even a
        # divergence) — it must never be walked or hashed.
        assert result.files_hashed == 0
        assert result.clean


class TestIndependenceFromPrefilter:
    """The audit module must not import or call into
    ``aios.tools.bash_memory_reconcile`` at all — a same-substrate verdict is
    worthless as a backstop (per the issue's design constraint)."""

    def test_module_does_not_import_bash_memory_reconcile(self) -> None:
        import ast
        import inspect

        source = inspect.getsource(audit)
        tree = ast.parse(source)
        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)

        assert not any("bash_memory_reconcile" in m for m in imported_modules), (
            f"the uncorrelated audit must never import the prefilter module "
            f"it's meant to independently verify; found imports: {imported_modules!r}"
        )
