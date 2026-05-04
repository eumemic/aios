"""Echo connector class.

Subclasses :class:`aios_connector.Connector` to demonstrate the SDK's
shape: multi-account snapshot, model-facing tools that route by
account, and one tool that synthesizes an inbound message so tests can
exercise spool→append→ack without a real platform daemon.
"""

from __future__ import annotations

from typing import Any

from aios_connector import Connector, focal_required, make_account, tool


class EchoConnector(Connector):
    name = "echo"
    version = "0.1.0"
    instructions = (
        "Echo connector. Use `ping` to verify wiring; `echo` to round-trip text "
        "(routes by account); `trigger_inbound` to synthesize an inbound message."
    )

    async def discover_accounts(self) -> list[dict[str, Any]]:
        # Two accounts so the multi-account focal-injection path is exercised
        # by default — single-account topology is the degenerate case (still
        # supported via SingleAccount-style tool sigs without the account kwarg).
        return [
            make_account(id="echo-1", display_name="Echo Account One"),
            make_account(id="echo-2", display_name="Echo Account Two"),
        ]

    @tool()
    async def ping(self) -> dict[str, Any]:
        """Return ``{"status": "pong"}`` — useful for verifying the connector is alive."""
        return {"status": "pong"}

    @tool()
    @focal_required
    async def echo(self, text: str, *, account: str, chat_id: str) -> dict[str, Any]:
        """Echo ``text`` back to the caller, tagged with the focal chat id and account.

        Demonstrates the multi-account focal-injection path: the SDK lifts
        ``account`` and ``chat_id`` from ``_meta.aios.focal_channel_path`` at
        dispatch time and the connector routes by account internally.
        """
        return {"text": text, "account": account, "chat_id": chat_id}

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
