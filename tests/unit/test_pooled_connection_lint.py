from __future__ import annotations

import ast
from pathlib import Path

from scripts.pooled_connection_lint import check_source


def _messages(source: str) -> list[str]:
    return [violation.message for violation in check_source(source, filename="example.py")]


def test_flags_foreign_await_while_pool_connection_is_held() -> None:
    messages = _messages(
        """
async def work(pool, client):
    async with pool.acquire() as conn:
        await conn.fetchrow("SELECT 1")
        await client.post("https://example.com")
"""
    )

    assert messages == ["pooled connection 'conn' held across non-DB await"]


def test_allows_db_helpers_receiving_the_held_connection() -> None:
    assert not _messages(
        """
async def work(pool):
    async with pool.acquire() as conn, conn.transaction():
        await queries.append_event(conn, "hello")
        await conn.execute("SELECT 1")
"""
    )


def test_transaction_scope_on_connection_parameter_is_checked() -> None:
    assert _messages(
        """
async def work(conn, model):
    async with conn.transaction():
        await model.stream()
"""
    )


def test_linked_pragma_allows_a_reviewed_exception() -> None:
    assert not _messages(
        """
async def work(pool, client):
    async with pool.acquire() as conn:
        await client.post("https://example.com")  # pooled-connection-await: allow https://github.com/eumemic/aios/issues/123
"""
    )


def test_unlinked_pragma_does_not_suppress_violation() -> None:
    assert _messages(
        """
async def work(pool, client):
    async with pool.acquire() as conn:
        await client.post("https://example.com")  # pooled-connection-await: allow
"""
    )


def test_synthetic_tree_violation_is_reported(tmp_path: Path) -> None:
    source = """
async def work(pool):
    async with pool.acquire() as conn:
        await asyncio.sleep(1)
"""
    path = tmp_path / "bad.py"
    path.write_text(source)

    tree = ast.parse(path.read_text(), filename=str(path))
    assert tree.body
    assert _messages(source)
