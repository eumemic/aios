from __future__ import annotations

import ast
from pathlib import Path

import pytest

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


def test_allows_database_stats_helpers_receiving_the_held_connection() -> None:
    assert not _messages(
        """
async def work(pool, tables):
    async with pool.acquire() as conn, conn.transaction(readonly=True):
        await db_stats_queries.database_size(conn)
        await db_stats_queries.table_storage_stats(conn)
        await db_stats_queries.monthly_buckets(conn, tables)
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


def test_iter_exemption_refs_sees_markers_the_old_ci_grep_missed(tmp_path: Path) -> None:
    """The em-dash blind spot: aios#2143.

    CI used to enumerate exemptions with an inline grep requiring the issue ref to
    follow ``allow`` IMMEDIATELY::

        grep -RhoE 'pooled-connection-await: allow eumemic/aios#[0-9]+' src

    A real marker read ``allow — eumemic/aios#919``. The grep never matched it, so
    that exemption cited a CLOSED issue for two months while the check reported
    success. This pins the parser against every punctuation variant, so the check
    can never again be blind to the formatting it did not anticipate.
    """
    import re

    from scripts.pooled_connection_lint import iter_exemption_refs

    variants = [
        ("plain", "allow eumemic/aios#111"),
        ("em_dash", "allow — eumemic/aios#222"),
        ("hyphen", "allow - eumemic/aios#333"),
        ("colon", "allow: eumemic/aios#444"),
        ("parens", "allow (eumemic/aios#555)"),
        ("trailing_prose", "allow eumemic/aios#666 — load-bearing, see thread"),
    ]
    src = tmp_path / "src"
    src.mkdir()
    for name, marker in variants:
        (src / f"{name}.py").write_text(
            "async def f(pool):\n"
            "    async with pool.acquire() as conn:\n"
            f"        return await conn.fetch('x')  # pooled-connection-await: {marker}\n",
            encoding="utf-8",
        )

    found = {issue for _path, _line, issue in iter_exemption_refs(str(src))}
    assert found == {111, 222, 333, 444, 555, 666}, (
        f"parser missed a marker variant: got {sorted(found)}"
    )

    # And prove the OLD grep semantics really were blind, so this test documents the
    # bug rather than merely asserting current behaviour.
    old_grep = re.compile(r"pooled-connection-await: allow eumemic/aios#(\d+)")
    old_found = {
        int(m.group(1))
        for _n, marker in variants
        for m in [old_grep.search(f"# pooled-connection-await: {marker}")]
        if m
    }
    assert 222 not in old_found, "the em-dash case must be what the old grep missed"
    assert old_found < found, "the new parser must strictly dominate the old grep"


def test_enumeration_refuses_to_report_clean_when_it_cannot_look(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Partial discovery failure must RAISE, never return a short list.

    The per-file read guard was necessary and NOT sufficient: it covers files that
    were discovered but unreadable. ``Path.rglob`` SUPPRESSES OSErrors raised while
    scanning directories, so an unreadable subtree simply does not appear in its
    results -- making partial discovery failure indistinguishable from "that subtree
    has no exemptions".

    That is aios#2138's class one level below the read: a non-empty result is not
    evidence of a COMPLETE result. The non-empty guard in CI cannot catch it, because
    the list is non-empty -- just short.
    """
    import os

    from scripts.pooled_connection_lint import iter_exemption_refs

    # A root that is not a directory yields zero markers from rglob, silently.
    not_a_dir = tmp_path / "file.py"
    not_a_dir.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not a directory"):
        iter_exemption_refs(str(not_a_dir))

    # A directory that cannot be scanned must escalate, not be skipped.
    src = tmp_path / "src"
    src.mkdir()
    (src / "ok.py").write_text(
        "async def f(pool):\n"
        "    async with pool.acquire() as conn:\n"
        "        return await conn.fetch('x')  # pooled-connection-await: allow eumemic/aios#1\n",
        encoding="utf-8",
    )
    real_walk = os.walk

    def exploding_walk(top, **kwargs):  # type: ignore[no-untyped-def]
        onerror = kwargs.get("onerror")
        if onerror is not None:
            onerror(OSError(13, "Permission denied", str(src / "locked")))
        return iter(())

    monkeypatch.setattr(os, "walk", exploding_walk)
    with pytest.raises(RuntimeError, match="cannot enumerate"):
        iter_exemption_refs(str(src))


def test_unreadable_file_raises_rather_than_reporting_no_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file we cannot READ is a file whose exemptions we do not know about.

    Committed because the original verification induced the OSError by hand and was
    correctly judged insufficient: it proved the handler propagates, not that the
    behaviour is pinned. ``chmod 000`` is useless here -- the test may run as root,
    and root can read anything, so the guard would appear to pass while never firing.
    """
    from pathlib import Path as _Path

    from scripts.pooled_connection_lint import iter_exemption_refs

    src = tmp_path / "src"
    src.mkdir()
    target = src / "unreadable.py"
    target.write_text("x = 1\n", encoding="utf-8")

    real_read_text = _Path.read_text

    def boom(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self.name == "unreadable.py":
            raise OSError(13, "Permission denied")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(_Path, "read_text", boom)
    with pytest.raises(RuntimeError, match="cannot read"):
        iter_exemption_refs(str(src))
