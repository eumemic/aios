from typing import Any

import pytest

from aios.db.queries.triggers import record_workflow_trigger_failure


class _ConnectionSpy:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *args: Any) -> str:
        self.calls.append((query, args))
        return "UPDATE 1"


@pytest.mark.asyncio
async def test_workflow_failure_updates_echo_only_for_latest_fire() -> None:
    conn = _ConnectionSpy()

    await record_workflow_trigger_failure(
        conn,  # type: ignore[arg-type]
        run_id="run_old",
        error_summary="failed late",
    )

    assert len(conn.calls) == 1
    query, args = conn.calls[0]
    assert "RETURNING trigger_id, started_at" in query
    assert "trigger.last_fire_at = failed.started_at" in query
    assert args == ("run_old", "failed late")
