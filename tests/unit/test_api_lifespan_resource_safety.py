"""Regression: API lifespan must close pool on pre-yield startup exception (#2064).

The pool is created before workspace-root validation runs. If validation
(or any other pre-yield step) raises, the pool must be closed so the
process does not leak asyncpg connections.

These tests read the source file directly rather than importing the module,
because ``aios.api.app`` has a module-level ``app = create_app()`` that
triggers MCP mount side effects requiring a full dependency graph.
"""

from __future__ import annotations

from pathlib import Path


def _read_app_source() -> str:
    """Read the app.py source without triggering module-level side effects."""
    app_path = Path(__file__).resolve().parents[2] / "src" / "aios" / "api" / "app.py"
    return app_path.read_text()


def test_lifespan_wraps_pre_yield_in_try_except() -> None:
    """The lifespan's pre-yield startup code must be wrapped in
    try/except BaseException that closes the pool on failure.
    """
    source = _read_app_source()
    assert "except BaseException:" in source, (
        "lifespan must catch BaseException to close pool on startup failure"
    )
    assert "await pool.close()" in source, (
        "lifespan must call pool.close() in the except branch"
    )
    # pool.close() must appear at least twice: once in except, once in finally
    assert source.count("await pool.close()") >= 2, (
        "pool.close() must appear in both the except-branch and the finally-branch"
    )


def test_pool_create_precedes_try_block() -> None:
    """``create_pool`` must be called BEFORE the try block so the pool
    object exists for the except branch to close it.
    """
    source = _read_app_source()
    pool_create_pos = source.index("create_pool(")
    try_pos = source.index("try:", pool_create_pos)
    except_pos = source.index("except BaseException:", try_pos)
    pool_close_pos = source.index("await pool.close()", except_pos)
    # Ordering: create_pool → try → except BaseException → pool.close()
    assert pool_create_pos < try_pos < except_pos < pool_close_pos


def test_validation_inside_try_block() -> None:
    """``validate_workspace_root_against_sessions`` must be called INSIDE
    the try block so a validation failure triggers pool cleanup.
    """
    source = _read_app_source()
    try_pos = source.index("try:")
    except_pos = source.index("except BaseException:")
    validate_pos = source.index("validate_workspace_root_against_sessions")
    assert try_pos < validate_pos < except_pos, (
        "validation must run between try and except so failures close the pool"
    )
