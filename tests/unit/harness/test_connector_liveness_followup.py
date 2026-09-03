from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from test_connector_liveness import _SQLitePool  # type: ignore[import-not-found]

from aios.harness.connector_liveness import (
    DockerConnectorHealthReader,
    read_bound_connection_activity,
)


@pytest.mark.asyncio
async def test_shared_session_activity_is_correlated_to_connection_channel() -> None:
    now = datetime(2026, 9, 3, tzinfo=UTC)
    pool = _SQLitePool()
    # The fixture's generated account is the same for both, so connectors distinguish channels.
    pool.db.executemany(
        "INSERT INTO connections VALUES (?, ?, ?, NULL)",
        [("wa", "whatsapp", "{}"), ("sl", "slack", "{}")],
    )
    pool.db.execute(
        "INSERT INTO sessions VALUES (?, ?, NULL)",
        ("shared", (now - timedelta(days=10)).isoformat()),
    )
    pool.db.executemany(
        "INSERT INTO bindings VALUES (?, 'single_session', 'shared', ?, NULL)",
        [
            ("wa", (now - timedelta(days=10)).isoformat()),
            ("sl", (now - timedelta(days=10)).isoformat()),
        ],
    )
    pool.db.executemany(
        "INSERT INTO events VALUES ('shared', ?, 'message', 'user', ?)",
        [
            ((now - timedelta(days=9)).isoformat(), "whatsapp/account/chat"),
            ((now - timedelta(minutes=1)).isoformat(), "slack/account/chat"),
        ],
    )

    activities = await read_bound_connection_activity(pool, {"whatsapp": 1, "slack": 1})

    assert {item.connection_id: item.last_activity_at for item in activities}[
        "wa"
    ] == now - timedelta(days=9)


@pytest.mark.asyncio
@pytest.mark.parametrize("include_pre_binding_event", [False, True])
async def test_current_binding_starts_silence_baseline(
    include_pre_binding_event: bool,
) -> None:
    now = datetime(2026, 9, 3, tzinfo=UTC)
    pool = _SQLitePool()
    pool.db.execute(
        "INSERT INTO connections VALUES (?, ?, ?, NULL)",
        ("wa", "whatsapp", "{}"),
    )
    pool.db.execute(
        "INSERT INTO sessions VALUES (?, ?, NULL)",
        ("existing", (now - timedelta(days=30)).isoformat()),
    )
    pool.db.execute(
        "INSERT INTO bindings VALUES (?, 'single_session', 'existing', ?, NULL)",
        ("wa", now.isoformat()),
    )
    if include_pre_binding_event:
        pool.db.execute(
            "INSERT INTO events VALUES (?, ?, 'message', 'user', ?)",
            (
                "existing",
                (now - timedelta(minutes=1)).isoformat(),
                "whatsapp/account/chat",
            ),
        )

    activities = await read_bound_connection_activity(pool, {"whatsapp": 1})

    assert activities[0].last_activity_at == now


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_running_correlated_runtime_wins_over_stopped_replica(
    monkeypatch: pytest.MonkeyPatch, reverse: bool
) -> None:
    output = json.dumps({"healthy_connection_ids": ["conn_1"], "unhealthy_connection_ids": []})

    def container(status: str, health: str) -> MagicMock:
        value = MagicMock()
        value.show = AsyncMock(
            return_value={
                "Names": ["/aios-whatsapp"],
                "State": {
                    "Status": status,
                    "Health": {"Status": health, "Log": [{"Output": output}]},
                },
            }
        )
        return value

    containers = [container("running", "healthy"), container("exited", "unhealthy")]
    if reverse:
        containers.reverse()
    docker = MagicMock()
    docker.containers.list = AsyncMock(return_value=containers)
    docker.close = AsyncMock()
    monkeypatch.setattr("aios.harness.connector_liveness.aiodocker.Docker", lambda: docker)

    health = await DockerConnectorHealthReader().read()

    assert health["conn_1"].healthy is True
