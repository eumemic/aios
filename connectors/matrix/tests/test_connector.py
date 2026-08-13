"""Pins the §4.4 secrets-less contract for the Matrix connector type.

Two layers:

* the declaration itself (``uses_connection_secrets = False``) — the
  flag the SDK runner branches on, and
* behavior: a multi-connection discovery backfill spawns every worker
  with an empty secrets dict and issues **zero**
  ``GET /v1/connectors/runtime/secrets`` round-trips.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from aios_matrix.connector import MatrixConnector


def test_matrix_declares_secrets_less() -> None:
    assert MatrixConnector.uses_connection_secrets is False
    assert MatrixConnector.connector == "matrix"


async def test_matrix_backfill_makes_zero_secrets_requests() -> None:
    """A multi-connection backfill must issue zero secrets GETs."""
    connector = MatrixConnector(base_url="http://x", token="aios_runtime_x")
    connector._client = MagicMock()

    spawned: list[str] = []

    async def _record(connection_id: str, secrets: dict[str, str]) -> None:
        assert secrets == {}
        spawned.append(connection_id)

    connector.serve_connection = _record  # type: ignore[method-assign]

    with patch("aios_connector_http.runner._get_runtime_secrets") as secrets_get:
        async with asyncio.TaskGroup() as tg:
            for i in range(5):
                await connector._on_connection_added(tg, f"con_{i}", f"_aios_agent_{i}")
            # ``_record`` returns immediately, so every spawned worker
            # drains on its own and the TaskGroup exits without cancels.

    assert sorted(spawned) == [f"con_{i}" for i in range(5)]
    secrets_get.assert_not_called()
