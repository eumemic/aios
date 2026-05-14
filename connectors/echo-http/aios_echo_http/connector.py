"""Reference connector — three tools, no real platform.

Mirrors :class:`aios_echo.EchoConnector` (the legacy MCP-stdio echo)
so existing parity tests can carry over.  The model calls these tools
through the standard ``requires_action`` flow; the connector executes
them as plain Python and POSTs the result back.

Echo has no per-platform event source — no real inbound feed to
subscribe to — so :meth:`HttpConnector.serve_connection` is left as
the default no-op block.  Each ``trigger_inbound`` tool call is the
test-driven inbound trigger.
"""

from __future__ import annotations

from typing import Any

from aios_connector_http import HttpConnector, tool


class EchoConnector(HttpConnector):
    """Three tools: ping, echo, trigger_inbound."""

    connector = "echo"

    @tool()
    async def ping(self) -> dict[str, Any]:
        """Return ``{"status": "pong"}`` — verifies the connector is alive."""
        return {"status": "pong"}

    @tool()
    async def echo(self, *, text: str) -> dict[str, Any]:
        """Echo ``text`` back; verifies tool dispatch + result POST."""
        return {"text": text}

    @tool()
    async def trigger_inbound(
        self,
        *,
        connection_id: str,
        chat_id: str,
        sender_name: str,
        content: str,
    ) -> dict[str, Any]:
        """Synthesize an inbound message — used by integration tests, not
        by the model in production (production inbounds arrive via a
        real platform feed in :meth:`serve_connection`)."""
        # raise_on_4xx so the test tool surfaces validation errors as
        # exceptions instead of silently dropping them — this method
        # exists for integration tests, which want loud failures.
        result = await self.emit_inbound(
            connection_id=connection_id,
            chat_id=chat_id,
            sender={"display_name": sender_name},
            content=content,
            raise_on_4xx=True,
        )
        assert result is not None  # ``raise_on_4xx=True`` guarantees this
        return {"event_id": result.get("appended_event_id")}


def main() -> None:
    """Module-level entry point: ``python -m aios_echo_http``."""
    import asyncio

    asyncio.run(EchoConnector().run())
