import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.harness.connector_liveness import DockerConnectorHealthReader


def _reader(container):
    docker = MagicMock()
    docker.containers.list = AsyncMock(return_value=[container])
    docker.close = AsyncMock()
    return docker


def _container(status, health, log):
    container = MagicMock()
    container.show = AsyncMock(
        return_value={
            "Names": ["/aios-whatsapp"],
            "State": {
                "Status": status,
                "Health": {"Status": health, "Log": log},
            },
        }
    )
    return container


@pytest.mark.asyncio
async def test_newest_malformed_does_not_resurrect_older_healthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reviewer F1: running/unhealthy container, newest health-log record is
    malformed (unparseable), an OLDER record attributed conn_1 healthy.
    The current probe failure must not resurrect the stale healthy verdict."""
    container = _container(
        "running",
        "unhealthy",
        [
            # OLDER record (index 0) — stale healthy attribution
            {"Output": json.dumps({"healthy_connection_ids": ["conn_1"], "unhealthy_connection_ids": []})},
            # NEWEST record (last) — malformed / current probe failed
            {"Output": "not-json"},
        ],
    )
    monkeypatch.setattr(
        "aios.harness.connector_liveness.aiodocker.Docker", lambda: _reader(container)
    )

    health = await DockerConnectorHealthReader().read()

    assert "conn_1" not in health or health["conn_1"].healthy is False, health.get("conn_1")
    assert health["whatsapp"].healthy is False


@pytest.mark.asyncio
async def test_clean_newest_healthy_record_stays_healthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OVER-CORRECTION GUARD: the fix must suppress only STALE green behind a
    malformed newest record, not green in general. A running/healthy container
    whose newest record cleanly attributes conn_1 healthy must stay healthy."""
    container = _container(
        "running",
        "healthy",
        [
            {"Output": json.dumps({"healthy_connection_ids": [], "unhealthy_connection_ids": []})},
            {"Output": json.dumps({"healthy_connection_ids": ["conn_1"], "unhealthy_connection_ids": []})},
        ],
    )
    monkeypatch.setattr(
        "aios.harness.connector_liveness.aiodocker.Docker", lambda: _reader(container)
    )

    health = await DockerConnectorHealthReader().read()

    assert health["conn_1"].healthy is True
    assert health["whatsapp"].healthy is True


@pytest.mark.asyncio
async def test_newest_malformed_still_attributes_which_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POSITIVE CONTROL: even with a malformed newest record, the reader must
    still attribute WHICH connection was served (conn_1 appears in the result),
    just as unhealthy — attribution is not erased, only the green verdict is."""
    container = _container(
        "running",
        "unhealthy",
        [
            {"Output": json.dumps({"healthy_connection_ids": ["conn_1"], "unhealthy_connection_ids": []})},
            {"Output": "not-json"},
        ],
    )
    monkeypatch.setattr(
        "aios.harness.connector_liveness.aiodocker.Docker", lambda: _reader(container)
    )

    health = await DockerConnectorHealthReader().read()

    assert "conn_1" in health
    assert health["conn_1"].healthy is False
