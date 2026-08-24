"""``GET /v1/sessions?view=lite&ids=`` skips fat hydration (vaults/echoes/triggers/obligations)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from aios.db.queries import sessions as session_queries
from aios.models.sessions import AwaitingToolCall, Session
from aios.services import sessions as sessions_service


class _CapturingConn:
    def __init__(self) -> None:
        self.sql: str | None = None
        self.args: tuple[Any, ...] = ()
        self.fetched = False

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self.sql = sql
        self.args = args
        self.fetched = True
        return []


def _session(sid: str = "sess_1") -> Session:
    return Session(
        id=sid,
        agent_id="agt_1",
        environment_id="env_1",
        agent_version=1,
        title=None,
        metadata={},
        status="idle",
        stop_reason=None,
        last_event_seq=0,
        created_at=datetime(2026, 8, 23, tzinfo=UTC),
        updated_at=datetime(2026, 8, 23, tzinfo=UTC),
        last_event_at=datetime(2026, 8, 23, 12, tzinfo=UTC),
    )


class _Txn:
    async def __aenter__(self) -> Any:
        return self

    async def __aexit__(self, *_: object) -> bool:
        return False

    def transaction(self, **_: Any) -> _Txn:
        return self


class _Pool:
    def acquire(self) -> _Txn:
        return _Txn()


class TestListSessionsIdsSql:
    async def test_ids_emits_any_clause(self) -> None:
        conn = _CapturingConn()
        await session_queries.list_sessions(conn, account_id="acc_x", ids=["sess_1", "sess_2"])
        assert conn.sql is not None
        assert "id = ANY(" in conn.sql
        assert ["sess_1", "sess_2"] in conn.args

    async def test_empty_ids_skips_query(self) -> None:
        conn = _CapturingConn()
        rows = await session_queries.list_sessions(conn, account_id="acc_x", ids=[])
        assert rows == []
        assert conn.fetched is False

    async def test_no_ids_omits_any_clause(self) -> None:
        conn = _CapturingConn()
        await session_queries.list_sessions(conn, account_id="acc_x")
        assert conn.sql is not None
        assert "id = ANY(" not in conn.sql


class TestListSessionsLiteSkipsFatHydration:
    async def test_lite_skips_vaults_echoes_triggers_obligations(self) -> None:
        fat: list[str] = []
        listed = [_session("sess_1")]

        async def list_rows(*_: Any, **kwargs: Any) -> list[Session]:
            assert kwargs.get("ids") == ["sess_1"]
            return listed

        async def trap_vaults(*_: Any, **__: Any) -> dict[str, list[str]]:
            fat.append("vaults")
            return {}

        async def trap_echoes(*_: Any, **__: Any) -> dict[str, list[Any]]:
            fat.append("echoes")
            return {}

        async def trap_triggers(*_: Any, **__: Any) -> dict[str, list[Any]]:
            fat.append("triggers")
            return {}

        async def trap_obligations(*_: Any, **__: Any) -> dict[str, list[Any]]:
            fat.append("obligations")
            return {}

        async def awaiting(*_: Any, **__: Any) -> dict[str, list[AwaitingToolCall]]:
            return {
                "sess_1": [
                    AwaitingToolCall(
                        tool_call_id="tc_1",
                        name="ask_user",
                        kind="custom",
                        pending_since=datetime(2026, 8, 23, tzinfo=UTC),
                    )
                ]
            }

        with (
            patch("aios.services.sessions.queries.list_sessions", list_rows),
            patch("aios.services.sessions.queries.batch_get_session_vault_ids", trap_vaults),
            patch("aios.services.sessions._batch_list_all_echoes", trap_echoes),
            patch("aios.services.sessions.queries.batch_list_session_triggers", trap_triggers),
            patch("aios.services.sessions.compute_awaiting", awaiting),
            patch("aios.services.sessions.compute_obligations", trap_obligations),
        ):
            out = await sessions_service.list_sessions(
                _Pool(),
                account_id="acc_1",
                ids=["sess_1"],
                view="lite",
            )

        assert fat == []
        assert len(out) == 1
        assert out[0].id == "sess_1"
        assert out[0].status == "idle"
        assert out[0].last_event_at == datetime(2026, 8, 23, 12, tzinfo=UTC)
        assert out[0].awaiting[0].kind == "custom"
        assert out[0].vault_ids == []
        assert out[0].resources == []
        assert out[0].triggers == []
        assert out[0].obligations == []

    async def test_full_still_hydrates(self) -> None:
        fat: list[str] = []

        async def list_rows(*_: Any, **__: Any) -> list[Session]:
            return [_session("sess_1")]

        async def vaults(*_: Any, **__: Any) -> dict[str, list[str]]:
            fat.append("vaults")
            return {"sess_1": ["vlt_1"]}

        async def echoes(*_: Any, **__: Any) -> dict[str, list[Any]]:
            fat.append("echoes")
            return {"sess_1": []}

        async def triggers(*_: Any, **__: Any) -> dict[str, list[Any]]:
            fat.append("triggers")
            return {"sess_1": []}

        async def obligations(*_: Any, **__: Any) -> dict[str, list[Any]]:
            fat.append("obligations")
            return {"sess_1": []}

        async def awaiting(*_: Any, **__: Any) -> dict[str, list[Any]]:
            return {}

        with (
            patch("aios.services.sessions.queries.list_sessions", list_rows),
            patch("aios.services.sessions.queries.batch_get_session_vault_ids", vaults),
            patch("aios.services.sessions._batch_list_all_echoes", echoes),
            patch("aios.services.sessions.queries.batch_list_session_triggers", triggers),
            patch("aios.services.sessions.compute_awaiting", awaiting),
            patch("aios.services.sessions.compute_obligations", obligations),
        ):
            out = await sessions_service.list_sessions(
                _Pool(),
                account_id="acc_1",
            )

        assert fat == ["vaults", "echoes", "triggers", "obligations"]
        assert out[0].vault_ids == ["vlt_1"]

    async def test_empty_ids_does_not_touch_db(self) -> None:
        listed = AsyncMock()
        with patch("aios.services.sessions.queries.list_sessions", listed):
            out = await sessions_service.list_sessions(
                MagicMock(),
                account_id="acc_1",
                ids=[],
                view="lite",
            )
        assert out == []
        listed.assert_not_called()
