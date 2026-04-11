"""Unit tests for the search tool handler."""

from __future__ import annotations

from typing import Any

import pytest

from aios.tools import file_session
from aios.tools.search import SearchArgumentError, search_handler
from aios.vendor.hermes_files.file_operations import SearchMatch, SearchResult


class _StubFileOps:
    def __init__(self) -> None:
        self.search_calls: list[dict[str, Any]] = []
        self._result = SearchResult(
            matches=[SearchMatch(path="/workspace/a.py", line_number=1, content="def foo():")],
            total_count=1,
        )

    def set_result(self, result: SearchResult) -> None:
        self._result = result

    async def search(self, pattern: str, **kwargs: Any) -> SearchResult:
        kwargs["pattern"] = pattern
        self.search_calls.append(kwargs)
        return self._result

    async def _exec(self, command: str, **kwargs: Any) -> Any:
        class _R:
            stdout = ""
            exit_code = 0

        return _R()

    def _escape_shell_arg(self, arg: str) -> str:
        return "'" + arg.replace("'", "'\"'\"'") + "'"


@pytest.fixture(autouse=True)
def _reset_file_sessions() -> None:
    file_session._sessions.clear()
    yield
    file_session._sessions.clear()


@pytest.fixture
def stub_fileops() -> _StubFileOps:
    return _StubFileOps()


@pytest.fixture
def stub_session(stub_fileops: _StubFileOps) -> file_session.FileToolSession:
    sess = file_session.FileToolSession(file_ops=stub_fileops)  # type: ignore[arg-type]
    file_session._sessions["sess_01TEST"] = sess
    return sess


class TestArguments:
    async def test_missing_pattern_raises(self, stub_session: file_session.FileToolSession) -> None:
        with pytest.raises(SearchArgumentError):
            await search_handler("sess_01TEST", {})

    async def test_empty_pattern_raises(self, stub_session: file_session.FileToolSession) -> None:
        with pytest.raises(SearchArgumentError):
            await search_handler("sess_01TEST", {"pattern": "  "})

    async def test_invalid_target_raises(self, stub_session: file_session.FileToolSession) -> None:
        with pytest.raises(SearchArgumentError):
            await search_handler("sess_01TEST", {"pattern": "x", "target": "wat"})

    async def test_invalid_output_mode_raises(
        self, stub_session: file_session.FileToolSession
    ) -> None:
        with pytest.raises(SearchArgumentError):
            await search_handler("sess_01TEST", {"pattern": "x", "output_mode": "ugly"})

    async def test_negative_offset_raises(self, stub_session: file_session.FileToolSession) -> None:
        with pytest.raises(SearchArgumentError):
            await search_handler("sess_01TEST", {"pattern": "x", "offset": -1})


class TestContentMode:
    async def test_happy_path(
        self, stub_fileops: _StubFileOps, stub_session: file_session.FileToolSession
    ) -> None:
        result = await search_handler("sess_01TEST", {"pattern": r"def \w+"})
        assert "matches" in result
        assert stub_fileops.search_calls[0]["pattern"] == r"def \w+"
        assert stub_fileops.search_calls[0]["target"] == "content"

    async def test_file_glob_propagates(
        self, stub_fileops: _StubFileOps, stub_session: file_session.FileToolSession
    ) -> None:
        await search_handler("sess_01TEST", {"pattern": "x", "file_glob": "*.py"})
        assert stub_fileops.search_calls[0]["file_glob"] == "*.py"

    async def test_output_mode_count_propagates(
        self, stub_fileops: _StubFileOps, stub_session: file_session.FileToolSession
    ) -> None:
        stub_fileops.set_result(SearchResult(counts={"/workspace/a.py": 3}, total_count=3))
        result = await search_handler("sess_01TEST", {"pattern": "x", "output_mode": "count"})
        assert "counts" in result
        assert result["counts"] == {"/workspace/a.py": 3}


class TestFilesMode:
    async def test_happy_path(
        self, stub_fileops: _StubFileOps, stub_session: file_session.FileToolSession
    ) -> None:
        stub_fileops.set_result(
            SearchResult(files=["/workspace/a.py", "/workspace/b.py"], total_count=2)
        )
        result = await search_handler("sess_01TEST", {"pattern": "*.py", "target": "files"})
        assert result["files"] == ["/workspace/a.py", "/workspace/b.py"]
        assert stub_fileops.search_calls[0]["target"] == "files"


class TestConsecutiveLoop:
    async def test_three_searches_warn(
        self, stub_fileops: _StubFileOps, stub_session: file_session.FileToolSession
    ) -> None:
        r1 = await search_handler("sess_01TEST", {"pattern": "x"})
        r2 = await search_handler("sess_01TEST", {"pattern": "x"})
        r3 = await search_handler("sess_01TEST", {"pattern": "x"})
        assert "_warning" not in r1
        assert "_warning" not in r2
        assert "_warning" in r3
        assert "consecutively" in r3["_warning"]

    async def test_four_searches_block(
        self, stub_fileops: _StubFileOps, stub_session: file_session.FileToolSession
    ) -> None:
        for _ in range(3):
            await search_handler("sess_01TEST", {"pattern": "x"})
        r4 = await search_handler("sess_01TEST", {"pattern": "x"})
        assert "error" in r4
        assert "BLOCKED" in r4["error"]

    async def test_different_pattern_resets_counter(
        self, stub_fileops: _StubFileOps, stub_session: file_session.FileToolSession
    ) -> None:
        await search_handler("sess_01TEST", {"pattern": "x"})
        await search_handler("sess_01TEST", {"pattern": "x"})
        await search_handler("sess_01TEST", {"pattern": "x"})
        r = await search_handler("sess_01TEST", {"pattern": "y"})
        assert "error" not in r
        assert "_warning" not in r


class TestPaginationHint:
    async def test_truncated_result_adds_next_offset_hint(
        self, stub_fileops: _StubFileOps, stub_session: file_session.FileToolSession
    ) -> None:
        stub_fileops.set_result(
            SearchResult(
                matches=[SearchMatch(path="/workspace/a.py", line_number=1, content="x")],
                total_count=100,
                truncated=True,
            )
        )
        result = await search_handler("sess_01TEST", {"pattern": "x", "limit": 10})
        assert "_next_offset_hint" in result
        assert "offset=10" in result["_next_offset_hint"]
