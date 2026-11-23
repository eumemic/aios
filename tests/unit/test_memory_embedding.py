"""Unit tests for hybrid memory recall's embedding seam (#1629). Embedder stubbed."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from aios.services import memory_embedding as me
from aios.services import memory_stores as svc
from aios.services.memory_embedding import (
    EMBEDDING_DIM,
    RRF_K,
    Embedded,
    Skipped,
    embed_memory_content,
    rrf_score,
    vector_literal,
)
from aios.tools import memory_search as ms


def _settings(model: str | None) -> Any:
    return lambda: SimpleNamespace(memory_embedding_model=model)


class TestEmbedMemoryContent:
    async def test_calls_aembedding_with_configured_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(me, "get_settings", _settings("text-embedding-3-small"))
        vec = [0.5] * EMBEDDING_DIM
        stub = AsyncMock(return_value={"data": [{"embedding": vec}]})
        monkeypatch.setattr("litellm.aembedding", stub)
        out = await embed_memory_content("the customer churned")
        assert out == Embedded(vector=vec, model="text-embedding-3-small")
        stub.assert_awaited_once_with(
            model="text-embedding-3-small", input=["the customer churned"]
        )

    async def test_provider_error_is_skipped_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(me, "get_settings", _settings("m"))
        monkeypatch.setattr("litellm.aembedding", AsyncMock(side_effect=RuntimeError("down")))
        out = await embed_memory_content("x")
        assert isinstance(out, Skipped)
        assert "down" in out.reason

    async def test_wrong_dimension_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(me, "get_settings", _settings("m"))
        stub = AsyncMock(return_value={"data": [{"embedding": [0.1, 0.2]}]})
        monkeypatch.setattr("litellm.aembedding", stub)
        assert isinstance(await embed_memory_content("x"), Skipped)

    async def test_unconfigured_model_is_skipped_without_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(me, "get_settings", _settings(None))
        stub = AsyncMock()
        monkeypatch.setattr("litellm.aembedding", stub)
        assert isinstance(await embed_memory_content("x"), Skipped)
        stub.assert_not_awaited()


class TestRrf:
    def test_first_fts_third_vec(self) -> None:
        assert RRF_K == 60
        assert rrf_score(1, 3) == pytest.approx(1 / 61 + 1 / 63)

    def test_absent_arm_contributes_zero(self) -> None:
        assert rrf_score(None, 2) == pytest.approx(1 / 62)
        assert rrf_score(None, None) == 0

    def test_sql_uses_same_k(self) -> None:
        assert f"1.0 / ({RRF_K} + fts.r)" in ms._SEARCH_SQL
        assert f"1.0 / ({RRF_K} + vec.r)" in ms._SEARCH_SQL

    def test_sql_guards_null_query_vector(self) -> None:
        # A NULL $2 must empty the vec arm (``<=> NULL`` alone would not).
        assert "$2::vector IS NOT NULL" in ms._SEARCH_SQL

    def test_vector_literal(self) -> None:
        assert vector_literal([1, 0.5]) == "[1.0,0.5]"


class _Conn:
    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows
        self.fetch_args: tuple[Any, ...] = ()

    async def execute(self, *a: Any) -> str:
        return "OK"

    async def fetch(self, *a: Any) -> list[Any]:
        self.fetch_args = a
        return self.rows

    def transaction(self, **kw: Any) -> Any:
        return _Ctx(None)


class _Ctx:
    def __init__(self, v: Any) -> None:
        self.v = v

    async def __aenter__(self) -> Any:
        return self.v

    async def __aexit__(self, *a: Any) -> None:
        return None


class _Pool:
    def __init__(self, conn: _Conn) -> None:
        self.conn = conn

    def acquire(self) -> Any:
        return _Ctx(self.conn)


class TestReadPathDegrade:
    async def test_query_embed_failure_binds_null_vector(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(me, "get_settings", _settings("m"))
        monkeypatch.setattr("litellm.aembedding", AsyncMock(side_effect=RuntimeError("down")))
        fts_rows = [{"path": "/a"}]
        conn = _Conn(fts_rows)
        rows, truncated = await ms._execute_query(_Pool(conn), "sess", "deploy")
        assert rows == fts_rows and truncated is False
        assert conn.fetch_args[1:] == ("deploy", None)

    async def test_query_embed_success_binds_vector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            ms, "embed_memory_content", AsyncMock(return_value=Embedded([1.0, 0.0], "m"))
        )
        conn = _Conn([])
        await ms._execute_query(_Pool(conn), "sess", "q")
        assert conn.fetch_args[2] == "[1.0,0.0]"


class TestWritePathEmbedding:
    def _patch(self, monkeypatch: pytest.MonkeyPatch, *, prior_path: str = "/a") -> AsyncMock:
        prior = SimpleNamespace(path=prior_path, content="old")
        monkeypatch.setattr("aios.db.queries.get_memory", AsyncMock(return_value=prior))
        monkeypatch.setattr(
            "aios.db.queries.update_memory_with_version",
            AsyncMock(
                side_effect=lambda conn, **kw: SimpleNamespace(
                    id="mem_1", path=kw["new_path"] or prior_path
                )
            ),
        )
        monkeypatch.setattr(svc, "_mirror_to_host", lambda *a: None)
        monkeypatch.setattr(svc, "_mirror_delete_from_host", lambda *a: None)
        embed = AsyncMock(return_value=Skipped("stub"))
        monkeypatch.setattr(me, "embed_and_store", embed)
        return embed

    async def test_pure_rename_does_not_embed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        embed = self._patch(monkeypatch)
        await svc.update_memory(
            _Pool(_Conn([])),
            account_id="acc",
            store_id="st",
            memory_id="mem_1",
            new_path="/b",
            actor=svc.SessionActor(session_id="sess_1"),
        )
        embed.assert_not_awaited()

    async def test_content_change_embeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        embed = self._patch(monkeypatch)
        await svc.update_memory(
            _Pool(_Conn([])),
            account_id="acc",
            store_id="st",
            memory_id="mem_1",
            new_content="new body",
            actor=svc.SessionActor(session_id="sess_1"),
        )
        embed.assert_awaited_once()
        assert embed.await_args is not None
        kw = embed.await_args.kwargs
        assert kw["content"] == "new body"
        assert kw["content_sha256"] == svc._sha256_hex("new body")

    async def test_create_embeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        embed = self._patch(monkeypatch)
        monkeypatch.setattr(
            "aios.db.queries.insert_memory_with_version",
            AsyncMock(return_value=SimpleNamespace(id="mem_9", path="/x")),
        )
        await svc.create_memory(
            _Pool(_Conn([])),
            account_id="acc",
            store_id="st",
            path="/x",
            content="hello",
            actor=svc.SessionActor(session_id="sess_1"),
        )
        embed.assert_awaited_once()
        assert embed.await_args is not None
        assert embed.await_args.kwargs["memory_id"] == "mem_9"

    async def test_embed_and_store_skipped_does_not_write(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(me, "embed_memory_content", AsyncMock(return_value=Skipped("x")))
        write = AsyncMock()
        monkeypatch.setattr("aios.db.queries.update_memory_embedding", write)
        out = await me.embed_and_store(
            _Pool(_Conn([])),
            account_id="a",
            memory_id="m",
            content="c",
            content_sha256="s",
        )
        assert isinstance(out, Skipped)
        write.assert_not_awaited()
