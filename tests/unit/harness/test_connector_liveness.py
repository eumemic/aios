from __future__ import annotations

import json
import sqlite3
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


class _SQLitePool:
    """Relational fixture adapter that executes the production reader SQL."""

    def __init__(self) -> None:
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.db.executescript(
            """
            CREATE TABLE connections (
                id TEXT PRIMARY KEY, connector TEXT, metadata TEXT, archived_at TEXT
            );
            CREATE TABLE bindings (
                connection_id TEXT, mode TEXT, session_id TEXT,
                created_at TEXT, archived_at TEXT
            );
            CREATE TABLE chat_sessions (
                connection_id TEXT, session_id TEXT, created_at TEXT
            );
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, created_at TEXT, archived_at TEXT
            );
            CREATE TABLE events (session_id TEXT, created_at TEXT);
            """
        )

    async def fetch(self, query: str) -> list[dict[str, object]]:
        rows = self.db.execute(query).fetchall()
        return [
            {
                **dict(row),
                "metadata": json.loads(row["metadata"]),
                "last_activity_at": datetime.fromisoformat(row["last_activity_at"]),
            }
            for row in rows
        ]


@pytest.mark.asyncio
async def test_detector_uses_current_binding_activity_from_production_reader() -> None:
    """Historical per-chat traffic cannot hide a silent current binding."""
    now = datetime(2026, 8, 29, tzinfo=UTC)
    current_activity = now - timedelta(days=9)
    historical_activity = now - timedelta(hours=1)
    pool = _SQLitePool()
    pool.db.execute(
        "INSERT INTO connections VALUES (?, ?, ?, NULL)",
        ("conn_1", "whatsapp", json.dumps({"liveness_silence_threshold_seconds": 604800})),
    )
    pool.db.executemany(
        "INSERT INTO sessions VALUES (?, ?, NULL)",
        [
            ("session_current", (now - timedelta(days=10)).isoformat()),
            ("session_historical", (now - timedelta(days=30)).isoformat()),
        ],
    )
    pool.db.executemany(
        "INSERT INTO bindings VALUES (?, ?, ?, ?, ?)",
        [
            ("conn_1", "per_chat", None, (now - timedelta(days=30)).isoformat(), now.isoformat()),
            (
                "conn_1",
                "single_session",
                "session_current",
                (now - timedelta(days=10)).isoformat(),
                None,
            ),
        ],
    )
    pool.db.execute(
        "INSERT INTO chat_sessions VALUES (?, ?, ?)",
        ("conn_1", "session_historical", (now - timedelta(days=30)).isoformat()),
    )
    pool.db.executemany(
        "INSERT INTO events VALUES (?, ?)",
        [
            ("session_current", current_activity.isoformat()),
            ("session_historical", historical_activity.isoformat()),
        ],
    )
    alarm = MagicMock()
    detector = ConnectorLivenessDetector(
        pool,
        thresholds={"whatsapp": 86400},
        health_reader=_HealthReader({"whatsapp": TransportHealth(False, "unhealthy")}),
        alarm=alarm,
        rate_limit_seconds=3600,
    )

    findings = await detector.check_once(now=now, monotonic_now=10000)

    assert len(findings) == 1
    assert findings[0]["connection_id"] == "conn_1"
    assert findings[0]["last_activity_at"] == current_activity.isoformat()
    assert findings[0]["silence_threshold_seconds"] == 604800
    alarm.assert_called_once()


@pytest.mark.asyncio
async def test_per_chat_binding_without_session_remains_observable() -> None:
    """A never-contacted per-chat binding uses binding time as its silence baseline."""
    now = datetime(2026, 8, 29, tzinfo=UTC)
    bound_at = now - timedelta(days=9)
    pool = _SQLitePool()
    pool.db.execute(
        "INSERT INTO connections VALUES (?, ?, ?, NULL)",
        ("conn_1", "whatsapp", json.dumps({"liveness_silence_threshold_seconds": 604800})),
    )
    pool.db.execute(
        "INSERT INTO bindings VALUES (?, ?, ?, ?, NULL)",
        ("conn_1", "per_chat", None, bound_at.isoformat()),
    )
    alarm = MagicMock()
    detector = ConnectorLivenessDetector(
        pool,
        thresholds={"whatsapp": 86400},
        health_reader=_HealthReader({"whatsapp": TransportHealth(False, "unhealthy")}),
        alarm=alarm,
        rate_limit_seconds=3600,
    )

    findings = await detector.check_once(now=now, monotonic_now=10000)

    assert len(findings) == 1
    assert findings[0]["connection_id"] == "conn_1"
    assert findings[0]["last_activity_at"] == bound_at.isoformat()
    assert findings[0]["session_silent_seconds"] == 9 * 86400
    alarm.assert_called_once()


@pytest.mark.asyncio
async def test_rebinding_per_chat_excludes_prior_binding_sessions() -> None:
    """A current per-chat binding uses only routing rows from its lifetime."""
    now = datetime(2026, 8, 29, tzinfo=UTC)
    current_activity = now - timedelta(days=9)
    historical_activity = now - timedelta(hours=1)
    current_bound_at = now - timedelta(days=10)
    pool = _SQLitePool()
    pool.db.execute(
        "INSERT INTO connections VALUES (?, ?, ?, NULL)",
        ("conn_1", "whatsapp", json.dumps({"liveness_silence_threshold_seconds": 604800})),
    )
    pool.db.executemany(
        "INSERT INTO sessions VALUES (?, ?, NULL)",
        [
            ("session_current", (now - timedelta(days=10)).isoformat()),
            ("session_historical", (now - timedelta(days=30)).isoformat()),
        ],
    )
    pool.db.executemany(
        "INSERT INTO bindings VALUES (?, ?, ?, ?, ?)",
        [
            (
                "conn_1",
                "per_chat",
                None,
                (now - timedelta(days=30)).isoformat(),
                current_bound_at.isoformat(),
            ),
            ("conn_1", "per_chat", None, current_bound_at.isoformat(), None),
        ],
    )
    pool.db.executemany(
        "INSERT INTO chat_sessions VALUES (?, ?, ?)",
        [
            ("conn_1", "session_historical", (now - timedelta(days=20)).isoformat()),
            ("conn_1", "session_current", current_bound_at.isoformat()),
        ],
    )
    pool.db.executemany(
        "INSERT INTO events VALUES (?, ?)",
        [
            ("session_current", current_activity.isoformat()),
            ("session_historical", historical_activity.isoformat()),
        ],
    )
    alarm = MagicMock()
    detector = ConnectorLivenessDetector(
        pool,
        thresholds={"whatsapp": 86400},
        health_reader=_HealthReader({"whatsapp": TransportHealth(False, "unhealthy")}),
        alarm=alarm,
        rate_limit_seconds=3600,
    )

    findings = await detector.check_once(now=now, monotonic_now=10000)

    assert len(findings) == 1
    assert findings[0]["last_activity_at"] == current_activity.isoformat()
    alarm.assert_called_once()


@pytest.mark.asyncio
async def test_running_container_without_health_result_is_not_healthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = MagicMock()
    container.show = AsyncMock(
        return_value={
            "Names": ["/aios-whatsapp"],
            "State": {"Status": "running"},
        }
    )
    docker = MagicMock()
    docker.containers.list = AsyncMock(return_value=[container])
    docker.close = AsyncMock()
    monkeypatch.setattr("aios.harness.connector_liveness.aiodocker.Docker", lambda: docker)

    health = await DockerConnectorHealthReader().read()

    assert health["whatsapp"].healthy is False
    assert health["whatsapp"].detail == "health status unavailable"


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
async def test_stopped_container_is_observed_by_docker_reader_after_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stopping the connector is observed through the production Docker reader."""
    now = datetime(2026, 8, 21, tzinfo=UTC)
    monkeypatch.setattr(
        "aios.harness.connector_liveness.read_bound_connection_activity",
        AsyncMock(
            return_value=[
                BoundConnectionActivity("conn_1", "telegram", now - timedelta(days=4), 3 * 86400)
            ]
        ),
    )
    stopped = MagicMock()
    stopped.show = AsyncMock(
        return_value={
            "Names": ["/aios-telegram"],
            "State": {"Status": "exited"},
        }
    )
    docker = MagicMock()
    docker.containers.list = AsyncMock(return_value=[stopped])
    docker.close = AsyncMock()
    monkeypatch.setattr("aios.harness.connector_liveness.aiodocker.Docker", lambda: docker)
    alarm = MagicMock()
    detector = ConnectorLivenessDetector(
        object(),
        thresholds={"telegram": 3 * 86400},
        health_reader=DockerConnectorHealthReader(),
        alarm=alarm,
        rate_limit_seconds=3600,
    )

    await detector.check_once(now=now, monotonic_now=10000)

    docker.containers.list.assert_awaited_once_with(all=True)
    stopped.show.assert_awaited_once_with()
    docker.close.assert_awaited_once_with()
    alarm.assert_called_once()
    assert "transport unhealthy (exited)" in alarm.call_args.args[1]["finding"]


@pytest.mark.asyncio
async def test_review_two_silent_connections_stopped_container_alarm() -> None:
    now = datetime(2026, 8, 29, tzinfo=UTC)
    pool = _SQLitePool()
    for connection_id in ("conn_1", "conn_2"):
        session_id = f"session_{connection_id}"
        bound_at = now - timedelta(days=10)
        pool.db.execute(
            "INSERT INTO connections VALUES (?, ?, ?, NULL)",
            (connection_id, "whatsapp", "{}"),
        )
        pool.db.execute(
            "INSERT INTO sessions VALUES (?, ?, NULL)",
            (session_id, bound_at.isoformat()),
        )
        pool.db.execute(
            "INSERT INTO bindings VALUES (?, ?, ?, ?, NULL)",
            (connection_id, "single_session", session_id, bound_at.isoformat()),
        )
        pool.db.execute(
            "INSERT INTO events VALUES (?, ?)",
            (session_id, (now - timedelta(days=9)).isoformat()),
        )
    detector = ConnectorLivenessDetector(
        pool,
        thresholds={"whatsapp": 7 * 86400},
        health_reader=_HealthReader({"whatsapp": TransportHealth(False, "container exited")}),
        alarm=MagicMock(),
        rate_limit_seconds=3600,
    )

    findings = await detector.check_once(now=now, monotonic_now=10000)

    assert {finding["connection_id"] for finding in findings} == {"conn_1", "conn_2"}


@pytest.mark.asyncio
async def test_two_silent_connections_without_container_both_alarm() -> None:
    """Container absence is connector-wide, not an ambiguous sibling failure."""
    now = datetime(2026, 8, 29, tzinfo=UTC)
    pool = _SQLitePool()
    for connection_id in ("conn_1", "conn_2"):
        bound_at = now - timedelta(days=9)
        pool.db.execute(
            "INSERT INTO connections VALUES (?, ?, ?, NULL)",
            (connection_id, "whatsapp", "{}"),
        )
        pool.db.execute(
            "INSERT INTO bindings VALUES (?, ?, ?, ?, NULL)",
            (connection_id, "per_chat", None, bound_at.isoformat()),
        )
    alarm = MagicMock()
    detector = ConnectorLivenessDetector(
        pool,
        thresholds={"whatsapp": 7 * 86400},
        health_reader=_HealthReader({}),
        alarm=alarm,
        rate_limit_seconds=3600,
    )

    findings = await detector.check_once(now=now, monotonic_now=10000)

    assert {finding["connection_id"] for finding in findings} == {"conn_1", "conn_2"}
    assert {finding["transport_detail"] for finding in findings} == {"container absent"}
    assert alarm.call_count == 2


@pytest.mark.asyncio
async def test_review_does_not_attribute_one_connections_failure_to_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 8, 29, tzinfo=UTC)
    monkeypatch.setattr(
        "aios.harness.connector_liveness.read_bound_connection_activity",
        AsyncMock(
            return_value=[
                BoundConnectionActivity(
                    "healthy_silent", "whatsapp", now - timedelta(days=9), 7 * 86400
                ),
                BoundConnectionActivity(
                    "unhealthy_recent", "whatsapp", now - timedelta(hours=1), 7 * 86400
                ),
            ]
        ),
    )
    detector = ConnectorLivenessDetector(
        MagicMock(),
        thresholds={"whatsapp": 7 * 86400},
        health_reader=_HealthReader({"whatsapp": TransportHealth(False, "unhealthy")}),
        alarm=MagicMock(),
        rate_limit_seconds=3600,
    )

    assert await detector.check_once(now=now, monotonic_now=10000) == []


@pytest.mark.asyncio
async def test_connection_correlated_health_alarms_only_unhealthy_silent_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 8, 29, tzinfo=UTC)
    activities = [
        BoundConnectionActivity("silent_unhealthy", "whatsapp", now - timedelta(days=9), 7 * 86400),
        BoundConnectionActivity("silent_healthy", "whatsapp", now - timedelta(days=9), 7 * 86400),
    ]
    monkeypatch.setattr(
        "aios.harness.connector_liveness.read_bound_connection_activity",
        AsyncMock(return_value=activities),
    )
    container = MagicMock()
    container.show = AsyncMock(
        return_value={
            "Names": ["/aios-whatsapp"],
            "State": {
                "Status": "running",
                "Health": {
                    "Status": "unhealthy",
                    "Log": [
                        {
                            "Output": json.dumps(
                                {
                                    "healthy_connection_ids": ["silent_healthy"],
                                    "unhealthy_connection_ids": ["silent_unhealthy"],
                                }
                            )
                        }
                    ],
                },
            },
        }
    )
    docker = MagicMock()
    docker.containers.list = AsyncMock(return_value=[container])
    docker.close = AsyncMock()
    monkeypatch.setattr("aios.harness.connector_liveness.aiodocker.Docker", lambda: docker)
    detector = ConnectorLivenessDetector(
        object(),
        thresholds={"whatsapp": 7 * 86400},
        health_reader=DockerConnectorHealthReader(),
        alarm=MagicMock(),
        rate_limit_seconds=3600,
    )

    findings = await detector.check_once(now=now, monotonic_now=10000)

    assert {finding["connection_id"] for finding in findings} == {"silent_unhealthy"}


@pytest.mark.asyncio
async def test_exited_container_retained_healthy_log_does_not_suppress_alarm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding #1: a stopped runtime's retained probe log must not mark a
    connection transport-healthy. The last successful probe named conn_1 in
    healthy_connection_ids, but the container has since exited."""
    now = datetime(2026, 8, 29, tzinfo=UTC)
    monkeypatch.setattr(
        "aios.harness.connector_liveness.read_bound_connection_activity",
        AsyncMock(
            return_value=[
                BoundConnectionActivity("conn_1", "whatsapp", now - timedelta(days=9), 7 * 86400)
            ]
        ),
    )
    container = MagicMock()
    container.show = AsyncMock(
        return_value={
            "Names": ["/aios-whatsapp"],
            "State": {
                "Status": "exited",
                "Health": {
                    "Status": "unhealthy",
                    "Log": [
                        {
                            "Output": json.dumps(
                                {
                                    "healthy_connection_ids": ["conn_1"],
                                    "unhealthy_connection_ids": [],
                                }
                            )
                        }
                    ],
                },
            },
        }
    )
    docker = MagicMock()
    docker.containers.list = AsyncMock(return_value=[container])
    docker.close = AsyncMock()
    monkeypatch.setattr("aios.harness.connector_liveness.aiodocker.Docker", lambda: docker)

    health = await DockerConnectorHealthReader().read()
    assert health["conn_1"].healthy is False
    assert health["conn_1"].definitive_connector_outage is True

    alarm = MagicMock()
    detector = ConnectorLivenessDetector(
        object(),
        thresholds={"whatsapp": 7 * 86400},
        health_reader=DockerConnectorHealthReader(),
        alarm=alarm,
        rate_limit_seconds=3600,
    )
    findings = await detector.check_once(now=now, monotonic_now=10000)
    assert {finding["connection_id"] for finding in findings} == {"conn_1"}


@pytest.mark.asyncio
async def test_running_container_correlated_healthy_log_still_trusted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Over-correction guard for finding #1: a RUNNING container's retained
    healthy log must still classify conn_ok healthy (do not degrade into
    'ignore the log entirely')."""
    now = datetime(2026, 8, 29, tzinfo=UTC)
    monkeypatch.setattr(
        "aios.harness.connector_liveness.read_bound_connection_activity",
        AsyncMock(
            return_value=[
                BoundConnectionActivity("conn_ok", "whatsapp", now - timedelta(days=9), 7 * 86400)
            ]
        ),
    )
    container = MagicMock()
    container.show = AsyncMock(
        return_value={
            "Names": ["/aios-whatsapp"],
            "State": {
                "Status": "running",
                "Health": {
                    "Status": "unhealthy",
                    "Log": [
                        {
                            "Output": json.dumps(
                                {
                                    "healthy_connection_ids": ["conn_ok"],
                                    "unhealthy_connection_ids": [],
                                }
                            )
                        }
                    ],
                },
            },
        }
    )
    docker = MagicMock()
    docker.containers.list = AsyncMock(return_value=[container])
    docker.close = AsyncMock()
    monkeypatch.setattr("aios.harness.connector_liveness.aiodocker.Docker", lambda: docker)

    health = await DockerConnectorHealthReader().read()
    assert health["conn_ok"].healthy is True

    alarm = MagicMock()
    detector = ConnectorLivenessDetector(
        object(),
        thresholds={"whatsapp": 7 * 86400},
        health_reader=DockerConnectorHealthReader(),
        alarm=alarm,
        rate_limit_seconds=3600,
    )
    findings = await detector.check_once(now=now, monotonic_now=10000)
    assert findings == []
