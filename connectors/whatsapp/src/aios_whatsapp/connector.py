"""WhatsApp connector built on the aios-connector-http SDK.

One container, one connector type (``"whatsapp"``), N connections —
each bound to one paired WhatsApp phone.  Unlike Signal's shared
multi-account daemon, **each connection owns its own ``whatsapp-daemon``
subprocess**: whatsmeow's ``Client`` is per-device, so a daemon per
phone keeps lifecycles isolated (a crash of one phone's daemon doesn't
take siblings down — :meth:`HttpConnector._isolated_serve_connection`
contains the failure).

Lifecycle:

* :meth:`setup` is a no-op — no container-wide resource to bring up.
* :meth:`serve_connection` per connection: read ``secrets["phone"]``,
  spawn a :class:`WhatsappDaemon` for that phone, wait for readiness,
  and drain the daemon's notification stream until the connection is
  removed or the daemon crashes.
* :meth:`teardown` is a no-op — each connection's daemon is cleaned up
  inside its own ``serve_connection`` ``finally``.

This module deliberately ships **no** ``@tool`` methods or platform
event handling — that lands in subsequent PRs once the daemon
integrates whatsmeow.  For now ``serve_connection`` just demonstrates
that the daemon spawns, answers ``version``, and surfaces its
notification stream so subsequent slices can plug parse/emit/dispatch
logic into a stable boundary.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import structlog
from aios_connector_http import HttpConnector

from .config import Settings
from .daemon import WhatsappDaemon

log = structlog.get_logger(__name__)


@dataclass
class _WhatsappConnectionState:
    """Per-connection state: identity + handle to its daemon.

    Subsequent PRs will extend this with the bot's own JID, push_name,
    contact roster cache, and group roster cache — all populated at
    ``serve_connection`` time after the daemon completes its WhatsApp
    handshake.
    """

    phone: str
    daemon: WhatsappDaemon


class WhatsappConnector(HttpConnector):
    connector = "whatsapp"
    state: dict[str, _WhatsappConnectionState]

    def __init__(self, cfg: Settings) -> None:
        super().__init__()
        self._cfg = cfg

    async def serve_connection(self, connection_id: str, secrets: dict[str, str]) -> None:
        """Bring up one phone's daemon and drain its notification stream.

        ``secrets["phone"]`` identifies the phone this connection owns.
        Missing → raise; the runner's
        :meth:`HttpConnector._isolated_serve_connection` logs the
        failure under ``connector.connection.serve_failed`` and keeps
        the container serving its other connections.

        The daemon's ``store_dir`` is ``<data_dir>/<phone>/`` — one
        whatsmeow sqlstore per phone, naturally sharded by directory.
        """
        phone = secrets.get("phone")
        if not phone:
            raise RuntimeError(
                f"whatsapp connection {connection_id!r} requires a 'phone' entry in its secrets"
            )

        store_dir = self._cfg.data_dir / phone
        async with WhatsappDaemon(
            daemon_bin=self._cfg.daemon_bin,
            host=self._cfg.daemon_host,
            port=self._cfg.daemon_port,
            store_dir=store_dir,
        ) as daemon:
            state = _WhatsappConnectionState(phone=phone, daemon=daemon)
            self.state[connection_id] = state
            log.info(
                "whatsapp.connection.ready",
                connection_id=connection_id,
                phone=phone,
                port=self._cfg.daemon_port,
            )
            try:
                await self._drain_notifications(connection_id, state)
            finally:
                self.state.pop(connection_id, None)

    async def _drain_notifications(
        self, connection_id: str, state: _WhatsappConnectionState
    ) -> None:
        """Consume the daemon's notification stream.

        v0: log everything that arrives and discard.  Future PRs replace
        this with method-dispatch (``message`` → parse → emit_inbound,
        ``reaction`` → reaction-specific path, ``pairCode`` → pairing
        management handler, etc.).  Notifications arriving at this stage
        are unexpected (the daemon has no whatsmeow integration yet),
        so logging them at WARNING surfaces wiring bugs early.
        """
        async for method, params in state.daemon.listener.notifications():
            log.warning(
                "whatsapp.notification.unhandled",
                connection_id=connection_id,
                method=method,
                params=params,
            )

    async def teardown(self) -> None:
        # Each connection's daemon is owned by its own ``serve_connection``
        # task and cleaned up via that task's ``async with`` exit.  Nothing
        # for the connector itself to release.
        await asyncio.sleep(0)
