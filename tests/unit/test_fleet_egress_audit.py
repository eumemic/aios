from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.harness import fleet_egress_audit as audit


class _Acquire:
    def __init__(self, conn: MagicMock) -> None:
        self.conn = conn

    async def __aenter__(self) -> MagicMock:
        return self.conn

    async def __aexit__(self, *_args: object) -> None:
        return None


def _pool(conn: MagicMock) -> MagicMock:
    pool = MagicMock()
    pool.acquire.return_value = _Acquire(conn)
    return pool


async def test_audit_reads_24h_authoritative_stream_and_alerts_every_finding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(UTC)
    rows = [
        {
            "id": "healthy",
            "session_id": "s0",
            "account_id": "a0",
            "created_at": now,
            "data": {"event": "egress_provisioned", "hosts_skipped": []},
        },
        {
            "id": "fail",
            "session_id": "s1",
            "account_id": "a1",
            "created_at": now,
            "data": {"event": "egress_provision_failed", "reason": "down"},
        },
        {
            "id": "skip",
            "session_id": "s2",
            "account_id": "a2",
            "created_at": now,
            "data": {"event": "egress_provisioned", "hosts_skipped": [{"host": "x"}]},
        },
    ]
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=rows)
    warning = MagicMock()
    monkeypatch.setattr(audit.log, "warning", warning)

    result = await audit.run_fleet_egress_audit(_pool(conn))

    sql = conn.fetch.await_args.args[0]
    assert "FROM events" in sql
    assert "INTERVAL '24 hours'" in sql
    assert "egress_provision_failed" in sql
    assert "egress_provisioned" in sql
    assert result.events_examined == 3
    assert result.healthy_events_observed == 1
    assert [finding.event_id for finding in result.findings] == ["fail", "skip"]
    assert warning.call_count == 2


async def test_silent_writer_never_reports_health(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=[])
    healthy = MagicMock()
    monkeypatch.setattr(audit.log, "info", healthy)

    with pytest.raises(audit.EgressLifecycleWriterSilentError, match="no healthy"):
        await audit.run_fleet_egress_audit(_pool(conn))

    healthy.assert_not_called()


async def test_read_failure_propagates_and_never_reports_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = MagicMock()
    conn.fetch = AsyncMock(side_effect=RuntimeError("event stream unavailable"))
    healthy = MagicMock()
    monkeypatch.setattr(audit.log, "info", healthy)

    with pytest.raises(RuntimeError, match="event stream unavailable"):
        await audit.run_fleet_egress_audit(_pool(conn))

    healthy.assert_not_called()
