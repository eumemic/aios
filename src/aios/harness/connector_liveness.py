"""Conjunctive connector transport/session-silence detector.

The detector runs in the worker, outside connector containers, so stopping a
connector cannot also stop the observer that reports it.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import aiodocker

from aios.logging import get_logger

log = get_logger("aios.worker.connector_liveness")

# Deliberately per connector type. Operators can replace this map with
# AIOS_CONNECTOR_LIVENESS_THRESHOLDS_SECONDS; there is no global fallback that
# silently assigns a high-volume channel's policy to a rarely-used one.
DEFAULT_THRESHOLDS_SECONDS: dict[str, float] = {
    "echo": 24 * 60 * 60,
    "matrix": 24 * 60 * 60,
    "signal": 7 * 24 * 60 * 60,
    "slack": 24 * 60 * 60,
    "sms": 7 * 24 * 60 * 60,
    "telegram": 3 * 24 * 60 * 60,
    "whatsapp": 7 * 24 * 60 * 60,
}


@dataclass(frozen=True, slots=True)
class BoundConnectionActivity:
    connection_id: str
    connector: str
    last_activity_at: datetime
    threshold_seconds: float


@dataclass(frozen=True, slots=True)
class TransportHealth:
    healthy: bool
    detail: str
    # True only when the observation proves that no runtime for this connector
    # type can receive. Ordinary heartbeat failures may belong to just one of
    # several connections and therefore default to unscoped.
    definitive_connector_outage: bool | None = None

    def __post_init__(self) -> None:
        if self.definitive_connector_outage is not None:
            return
        detail = self.detail.lower()
        definitive = detail in {
            "container absent",
            "container exited",
            "dead",
            "exited",
            "removing",
        }
        object.__setattr__(self, "definitive_connector_outage", definitive)


async def read_bound_connection_activity(
    pool: Any, thresholds: Mapping[str, float]
) -> list[BoundConnectionActivity]:
    """Read every active connection, including bindings awaiting their first session."""
    rows = await pool.fetch(
        """
        WITH bound_sessions AS (
            SELECT c.id AS connection_id, c.connector, c.metadata,
                   s.id AS session_id, s.created_at AS session_created_at,
                   b.created_at AS bound_at
              FROM connections c
              JOIN bindings b ON b.connection_id = c.id
                             AND b.archived_at IS NULL
                             AND b.mode = 'single_session'
              JOIN sessions s ON s.id = b.session_id AND s.archived_at IS NULL
             WHERE c.archived_at IS NULL
            UNION ALL
            SELECT c.id, c.connector, c.metadata,
                   s.id, s.created_at, b.created_at
              FROM connections c
              JOIN bindings b ON b.connection_id = c.id
                             AND b.archived_at IS NULL
                             AND b.mode = 'per_chat'
              LEFT JOIN chat_sessions cs ON cs.connection_id = c.id
                                        AND cs.created_at >= b.created_at
              LEFT JOIN sessions s ON s.id = cs.session_id AND s.archived_at IS NULL
             WHERE c.archived_at IS NULL
        )
        SELECT bs.connection_id, bs.connector, bs.metadata,
               COALESCE(MAX(e.created_at), MAX(bs.session_created_at), MAX(bs.bound_at))
                   AS last_activity_at
          FROM bound_sessions bs
          LEFT JOIN events e ON e.session_id = bs.session_id
                            AND e.kind = 'message'
                            AND e.role = 'user'
                            AND e.orig_channel IS NOT NULL
         GROUP BY bs.connection_id, bs.connector, bs.metadata
         ORDER BY bs.connection_id
        """
    )
    result: list[BoundConnectionActivity] = []
    for row in rows:
        metadata = row["metadata"] or {}
        configured = metadata.get("liveness_silence_threshold_seconds")
        threshold = (
            float(configured) if configured is not None else thresholds.get(row["connector"])
        )
        if threshold is None:
            # Missing policy is unknown, never silently interpreted as healthy.
            log.error(
                "connector.liveness_threshold_missing_alarm",
                alarm_event=True,
                connector=row["connector"],
                connection_id=row["connection_id"],
            )
            continue
        result.append(
            BoundConnectionActivity(
                connection_id=row["connection_id"],
                connector=row["connector"],
                last_activity_at=row["last_activity_at"],
                threshold_seconds=threshold,
            )
        )
    return result


class DockerConnectorHealthReader:
    """Read connector health from Docker, including stopped containers."""

    async def read(self) -> dict[str, TransportHealth]:
        docker = aiodocker.Docker()
        try:
            containers = await docker.containers.list(all=True)
            payloads = [await container.show() for container in containers]
        finally:
            await docker.close()
        result: dict[str, TransportHealth] = {}
        for data in payloads:
            config = data.get("Config") or {}
            labels = data.get("Labels") or config.get("Labels") or {}
            raw_names = data.get("Names") or [data.get("Name")]
            names = [str(name).lstrip("/") for name in raw_names if name]
            connector = labels.get("aios.connector") or labels.get("com.docker.compose.service")
            if connector == "echo-http":
                connector = "echo"
            if connector not in DEFAULT_THRESHOLDS_SECONDS:
                aliases = {"echo": ("echo", "echo-http", "aios-echo-http")}
                connector = next(
                    (
                        candidate
                        for candidate in DEFAULT_THRESHOLDS_SECONDS
                        if any(
                            name in aliases.get(candidate, (candidate, f"aios-{candidate}"))
                            for name in names
                        )
                    ),
                    None,
                )
            if connector is None:
                continue
            state_payload = data.get("State") or {}
            if isinstance(state_payload, dict):
                state = str(state_payload.get("Status") or "unknown")
                health = str((state_payload.get("Health") or {}).get("Status") or "")
                detail = health or ("health status unavailable" if state == "running" else state)
                healthy = state == "running" and health == "healthy"
            else:  # list() payloads used by lightweight Docker-compatible APIs
                state = str(state_payload)
                status = str(data.get("Status") or "").lower()
                healthy = state == "running" and "(healthy)" in status
                detail = status or "health status unavailable"
            if healthy:
                detail = "healthy"

            # SDK probes print connection-correlated state into Docker's health
            # log. Consume the newest valid record so one failed account does
            # not make a healthy sibling look failed (or hide the real failure).
            #
            # The health log is HISTORICAL: it survives process exit and retains
            # the last successful probe output. When the runtime is definitively
            # unavailable (stopped/dead/exiting/absent), that retained "healthy"
            # verdict is no longer true for any connection this runtime served,
            # so we must not let it classify a connection transport-healthy. We
            # still consume the record to attribute WHICH connections the dead
            # runtime was serving — every one of them is now unhealthy with a
            # definitive connector outage — but a correlated "unhealthy" verdict
            # is likewise upgraded to a definitive outage.
            runtime_down = state in {"dead", "exited", "removing"}
            if isinstance(state_payload, dict):
                health_payload = state_payload.get("Health") or {}
                for entry in reversed(health_payload.get("Log") or []):
                    try:
                        correlated = json.loads(entry.get("Output") or "")
                    except (AttributeError, TypeError, ValueError):
                        continue
                    for connection_id in correlated.get("healthy_connection_ids", []):
                        if runtime_down:
                            result[str(connection_id)] = TransportHealth(
                                False, detail, definitive_connector_outage=True
                            )
                        else:
                            result[str(connection_id)] = TransportHealth(True, "healthy")
                    for connection_id in correlated.get("unhealthy_connection_ids", []):
                        result[str(connection_id)] = TransportHealth(
                            False,
                            detail if runtime_down else "transport not serving",
                            definitive_connector_outage=runtime_down,
                        )
                    break

            stopped = state in {"dead", "exited", "removing"}
            observation = TransportHealth(
                healthy=healthy,
                detail=detail,
                definitive_connector_outage=stopped,
            )
            # A connector type is healthy only when every observed container is
            # healthy. It is definitively unavailable only when every observed
            # runtime is stopped; a stopped stale replica beside a running one is
            # still an uncorrelated type-level failure.
            previous = result.get(connector)
            if previous is None:
                result[connector] = observation
            else:
                result[connector] = TransportHealth(
                    healthy=previous.healthy and observation.healthy,
                    detail=(previous.detail if not previous.healthy else observation.detail),
                    definitive_connector_outage=(
                        bool(previous.definitive_connector_outage)
                        and bool(observation.definitive_connector_outage)
                    ),
                )
        return result


class TransportHealthReader(Protocol):
    async def read(self) -> dict[str, TransportHealth]: ...


class ConnectorLivenessDetector:
    def __init__(
        self,
        pool: Any,
        *,
        thresholds: Mapping[str, float],
        health_reader: TransportHealthReader,
        alarm: Callable[[str, dict[str, Any]], None],
        rate_limit_seconds: float,
    ) -> None:
        self.pool = pool
        self.thresholds = thresholds
        self.health_reader = health_reader
        self.alarm = alarm
        self.rate_limit_seconds = rate_limit_seconds
        self._last_alarm: dict[str, float] = {}

    async def check_once(
        self, *, now: datetime | None = None, monotonic_now: float | None = None
    ) -> list[dict[str, Any]]:
        wall_now = now or datetime.now(UTC)
        stamp = time.monotonic() if monotonic_now is None else monotonic_now
        activities, health = await asyncio.gather(
            read_bound_connection_activity(self.pool, self.thresholds), self.health_reader.read()
        )
        alarms: list[dict[str, Any]] = []
        connection_counts: dict[str, int] = {}
        for activity in activities:
            connection_counts[activity.connector] = connection_counts.get(activity.connector, 0) + 1
        for activity in activities:
            connection_transport = health.get(activity.connection_id)
            transport = connection_transport or health.get(
                activity.connector, TransportHealth(False, "container absent")
            )
            silent_seconds = max(0.0, (wall_now - activity.last_activity_at).total_seconds())
            # Connector-wide Docker health cannot identify which connection is
            # unhealthy.  Applying it to every row would manufacture a false
            # conjunction for a healthy-but-quiet sibling.  A type-level signal
            # remains actionable when there is exactly one bound connection;
            # readers may provide a connection-id key for precise multi-account
            # attribution.
            uncorrelated_multi_connection = (
                connection_transport is None
                and connection_counts[activity.connector] > 1
                and not transport.definitive_connector_outage
            )
            # The conjunction is load-bearing: neither an ordinary restart, a
            # healthy quiet channel, nor an uncorrelated sibling failure emits.
            if (
                transport.healthy
                or uncorrelated_multi_connection
                or silent_seconds <= activity.threshold_seconds
            ):
                continue
            if (
                stamp - self._last_alarm.get(activity.connection_id, float("-inf"))
                < self.rate_limit_seconds
            ):
                continue
            days = silent_seconds / 86400
            finding = (
                f"{activity.connector}: transport unhealthy ({transport.detail}), "
                f"no bound-session activity in {days:.1f}d "
                f"(threshold {activity.threshold_seconds / 86400:.1f}d)"
            )
            specimen = {
                "finding": finding,
                "connector": activity.connector,
                "connection_id": activity.connection_id,
                "transport_unhealthy": True,
                "transport_detail": transport.detail,
                "session_silent_seconds": silent_seconds,
                "silence_threshold_seconds": activity.threshold_seconds,
                "last_activity_at": activity.last_activity_at.isoformat(),
            }
            self.alarm("connector_liveness", specimen)
            self._last_alarm[activity.connection_id] = stamp
            alarms.append(specimen)
        return alarms


async def run_connector_liveness_detector(
    pool: Any,
    *,
    thresholds: Mapping[str, float],
    interval_seconds: float,
    rate_limit_seconds: float,
) -> None:
    def emit(kind: str, specimen: dict[str, Any]) -> None:
        log.error(f"worker.{kind}_alarm", alarm_event=True, **specimen)

    detector = ConnectorLivenessDetector(
        pool,
        thresholds=thresholds,
        health_reader=DockerConnectorHealthReader(),
        alarm=emit,
        rate_limit_seconds=rate_limit_seconds,
    )
    while True:
        try:
            await detector.check_once()
        except asyncio.CancelledError:
            raise
        except Exception:
            # Monitoring failure is itself loud but does not kill the worker.
            log.exception("connector.liveness_check_failed", alarm_event=True)
        await asyncio.sleep(interval_seconds)
