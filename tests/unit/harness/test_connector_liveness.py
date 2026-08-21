from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.harness.connector_liveness import (
    BoundConnectionActivity,
    ConnectorLivenessDetector,
    DockerConnectorHealthReader,
    TransportHealth,
)


class _HealthReader:
    def __init__(self, health: dict[str, TransportHealth]) -> None:
        self.health = health

    async def read(self) -> dict[str, TransportHealth]:
        return self.health


@pytest.mark.asyncio
async def test_unhealthy_replica_is_not_hidden_by_healthy_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unhealthy = MagicMock()
    unhealthy.show = AsyncMock(
        return_value={
            "Names": ["/aios-whatsapp"],
            "State": {"Status": "running", "Health": {"Status": "unhealthy"}},
        }
    )
    stale_healthy = MagicMock()
    stale_healthy.show = AsyncMock(
        return_value={
            "Names": ["/old-whatsapp"],
            "State": {"Status": "running", "Health": {"Status": "healthy"}},
        }
    )
    docker = MagicMock()
    docker.containers.list = AsyncMock(return_value=[unhealthy, stale_healthy])
    docker.close = AsyncMock()
    monkeypatch.setattr("aios.harness.connector_liveness.aiodocker.Docker", lambda: docker)

    health = await DockerConnectorHealthReader().read()

    assert health["whatsapp"].healthy is False
    assert health["whatsapp"].detail == "unhealthy"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("healthy", "age_days", "expected"),
    [
        (False, 9, 1),  # both facts: alarm
        (True, 9, 0),  # quiet but healthy: no noise
        (False, 1, 0),  # unhealthy but recently active: no noise
    ],
)
async def test_alarm_requires_unhealthy_transport_and_stale_session(
    monkeypatch: pytest.MonkeyPatch, healthy: bool, age_days: int, expected: int
) -> None:
    now = datetime(2026, 8, 21, tzinfo=UTC)
    activity = BoundConnectionActivity(
        connection_id="conn_1",
        connector="whatsapp",
        last_activity_at=now - timedelta(days=age_days),
        threshold_seconds=7 * 86400,
    )
    monkeypatch.setattr(
        "aios.harness.connector_liveness.read_bound_connection_activity",
        AsyncMock(return_value=[activity]),
    )
    alarm = MagicMock()
    detector = ConnectorLivenessDetector(
        object(),
        thresholds={"whatsapp": 7 * 86400},
        health_reader=_HealthReader({"whatsapp": TransportHealth(healthy, "running (unhealthy)")}),
        alarm=alarm,
        rate_limit_seconds=3600,
    )

    findings = await detector.check_once(now=now, monotonic_now=10000)

    assert len(findings) == expected
    assert alarm.call_count == expected
    if expected:
        finding = findings[0]["finding"]
        assert "transport unhealthy" in finding
        assert "no bound-session activity" in finding
        assert "9.0d" in finding


@pytest.mark.asyncio
async def test_stopped_or_absent_container_fires_after_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation path: stopping/removing the connector is observed out-of-process."""
    now = datetime(2026, 8, 21, tzinfo=UTC)
    monkeypatch.setattr(
        "aios.harness.connector_liveness.read_bound_connection_activity",
        AsyncMock(
            return_value=[
                BoundConnectionActivity("conn_1", "telegram", now - timedelta(days=4), 3 * 86400)
            ]
        ),
    )
    alarm = MagicMock()
    detector = ConnectorLivenessDetector(
        object(),
        thresholds={"telegram": 3 * 86400},
        health_reader=_HealthReader({}),
        alarm=alarm,
        rate_limit_seconds=3600,
    )

    await detector.check_once(now=now, monotonic_now=10000)

    alarm.assert_called_once()
    assert "container absent" in alarm.call_args.args[1]["finding"]
