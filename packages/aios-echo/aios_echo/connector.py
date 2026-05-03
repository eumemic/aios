"""Echo connector class.

Subclasses :class:`aios_connector.Connector` to demonstrate the SDK's
shape: account snapshot, two model-facing tools, and one tool that
synthesizes an inbound message so tests can exercise
spool→append→ack without a real platform daemon.
"""

from __future__ import annotations

from typing import Any

from aios_connector import Connector, focal_required, make_account, tool


class EchoConnector(Connector):
    name = "echo"
    version = "0.1.0"
    instructions = "Echo connector. Use `ping` to verify wiring; `echo` to round-trip text."

    async def discover_accounts(self) -> list[dict[str, Any]]:
        return [make_account(id="echo-1", display_name="Echo Account One")]

    @tool()
    async def ping(self) -> dict[str, Any]:
        """Return ``{"status": "pong"}`` — useful for verifying the connector is alive."""
        return {"status": "pong"}

    @tool()
    @focal_required
    async def echo(self, text: str, *, focal: str) -> dict[str, Any]:
        """Echo ``text`` back to the caller, tagged with the focal chat id.

        Demonstrates :func:`focal_required` — the connector author writes the
        method as if ``focal`` is a normal kwarg, and the SDK lifts it from
        ``_meta.aios.focal_channel_path`` at dispatch time.
        """
        return {"text": text, "focal": focal}

    @tool()
    async def trigger_inbound(
        self,
        account: str,
        chat_id: str,
        sender_name: str,
        content: str,
    ) -> dict[str, Any]:
        """Synthesize an inbound message — used by e2e tests, not by the model in prod.

        Returns the assigned ``event_id`` so tests can correlate the
        worker-side append against the spool entry.
        """
        event_id = await self.emit_inbound(
            account=account,
            chat_id=chat_id,
            sender={"display_name": sender_name},
            content=content,
        )
        return {"event_id": event_id}
