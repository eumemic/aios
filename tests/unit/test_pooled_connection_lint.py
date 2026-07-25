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


def test_passing_conn_to_arbitrary_call_is_not_db_io() -> None:
    # Reviewer's bypass: classification must be by called object, not arguments.
    assert _messages(
        """
async def work(pool, client):
    async with pool.acquire() as conn:
        await client.post(conn, "https://example.com")
"""
    )


def test_fabricated_repository_pragma_does_not_suppress_violation() -> None:
    assert _messages(
        """
async def work(pool, client):
    async with pool.acquire() as conn:
        await client.post("https://example.com")  # pooled-connection-await: allow evil/aios#123
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
        await client.post("https://example.com")  # pooled-connection-await: allow eumemic/aios#123
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


def test_name_heuristics_are_not_db_helpers() -> None:
    for call in ("get_weather(conn)", "send_network_conn(conn)", "service.post(conn)"):
        assert _messages(f"""
async def work(pool):
    async with pool.acquire() as conn:
        await {call}
""")


def test_attribute_connection_does_not_allow_other_self_awaits() -> None:
    assert _messages("""
async def work(self):
    async with self.conn.transaction():
        await self.http_client.post("https://example.com")
""")


def test_qualified_non_db_names_are_not_blanket_allowed() -> None:
    for call in (
        "queries.get_weather(conn)",
        "queries.send_network_conn(conn)",
        "_queries.post(conn)",
        "wf_queries.http(conn)",
        "trace_q.fetch_weather(conn)",
    ):
        assert _messages(f"""
async def work(pool):
    async with pool.acquire() as conn:
        await {call}
""")
