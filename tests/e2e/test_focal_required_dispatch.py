"""E2E: outbound focal-required MCP dispatch through the aios-connector SDK.

Closes the test gap that allowed the SDK ``ctx.meta`` bug into PR #211.
That bug was on the SDK's *receive* path: the harness sent
``_meta.aios.focal_channel_path`` correctly over JSON-RPC, but
``_focal_from_request_meta`` read ``ctx.request.meta`` (which is None
on a ``CallToolRequest``) instead of ``ctx.meta`` (where the lowlevel
server stashes the parsed meta).  Every focal-required tool call
failed with ``"aios should inject _meta.aios.focal_channel_path"``
even when aios *had* injected it.

PR #211's e2e tests covered the inbound path
(``test_connector_inbound.py``) and the supervisor's bare dispatch
mechanics (``test_connector_supervisor.py`` against a hand-written
fixture that bypasses the SDK).  Neither exercised an outbound
focal-required tool dispatched through a *real* aios-connector
subprocess — the SDK code path that was broken.

This test spawns ``python -m aios_echo`` as a real subprocess (using
the SDK's framing end-to-end), sets a focal channel suffix on the
``call_tool`` ``_meta``, and asserts the tool result reflects the
account/chat_id the SDK extracted.  Without the ``ctx.meta`` fix it
returns a tool error; with it, the round-trip succeeds.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from aios_connector import ConnectorSpec

from aios.config import ConnectorInstance, Settings
from aios.harness.connector_supervisor import ConnectorSubprocessRegistry


def _aios_echo_specs() -> list[tuple[ConnectorInstance, ConnectorSpec]]:
    """Spawn the echo connector via the SDK runner — the supervisor's
    real launch path.

    ``aios_echo`` uses the public :class:`aios_connector.Connector` base
    class and is loaded by ``python -m aios_connector echo`` via the
    ``aios.connectors`` entry point — exercising the SDK's
    ``_focal_from_request_meta`` in a real subprocess.
    """
    return [
        (
            ConnectorInstance(connector="echo", instance="echo"),
            ConnectorSpec(
                name="echo",
                command=sys.executable,
                args=["-m", "aios_connector", "echo"],
            ),
        )
    ]


async def _wait_for(predicate: Any, *, max_wait_s: float = 10.0) -> None:
    deadline = asyncio.get_event_loop().time() + max_wait_s
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"predicate {predicate!r} did not become true within {max_wait_s}s")


class TestFocalRequiredDispatch:
    """``mcp__echo__echo`` is ``@focal_required``; it MUST receive the
    SDK-extracted ``account`` and ``chat_id`` derived from
    ``_meta.aios.focal_channel_path`` rather than failing with
    ``"aios should inject _meta.aios.focal_channel_path"``.
    """

    async def test_focal_meta_round_trips_to_focal_required_tool(self) -> None:
        registry = ConnectorSubprocessRegistry(_aios_echo_specs(), settings=Settings())
        await registry.start()
        try:
            session = await asyncio.wait_for(registry.get_session("echo", "echo"), timeout=15.0)

            # Wait until the SDK has finished its post-init handshake and the
            # accounts snapshot is in.  ``aios_echo``'s ``discover_accounts``
            # advertises ``echo-1`` and ``echo-2``.
            state = registry.state("echo", "echo")
            assert state is not None

            await _wait_for(
                lambda: any(a.get("id") == "echo-1" for a in state.accounts),
                max_wait_s=15.0,
            )

            # Dispatch through the same code path the harness uses:
            # ``ClientSession.call_tool(name, arguments, meta=...)``.  The
            # account+chat_id segments mirror what
            # ``aios.harness.tool_dispatch`` builds from the session's
            # focal_channel for outbound calls.
            result = await asyncio.wait_for(
                session.call_tool(
                    "echo",
                    {"text": "hello"},
                    meta={"aios.focal_channel_path": "echo-1/chat-42"},
                ),
                timeout=10.0,
            )

            # Without the SDK fix, isError=True with content like:
            # "tool 'echo' requires a focal channel — aios should inject ..."
            assert not result.isError, (
                f"focal-required tool should not error when _meta is set; got: "
                f"{[c.text for c in result.content if hasattr(c, 'text')]}"
            )

            # The SDK-extracted account/chat_id must round-trip through the
            # tool's return value.
            assert result.content
            text_blocks = [c.text for c in result.content if hasattr(c, "text")]
            payload = " ".join(text_blocks)
            assert "echo-1" in payload, f"account not in result payload: {payload!r}"
            assert "chat-42" in payload, f"chat_id not in result payload: {payload!r}"
        finally:
            await registry.shutdown()

    async def test_missing_focal_meta_returns_tool_error(self) -> None:
        """Negative case: with no ``_meta``, the SDK raises a tool-level
        error message — proves the focal-meta plumbing is the load-bearing
        path (i.e. the success above isn't passing for some unrelated
        reason).
        """
        registry = ConnectorSubprocessRegistry(_aios_echo_specs(), settings=Settings())
        await registry.start()
        try:
            session = await asyncio.wait_for(registry.get_session("echo", "echo"), timeout=15.0)
            state = registry.state("echo", "echo")
            assert state is not None
            await _wait_for(lambda: bool(state.accounts), max_wait_s=15.0)

            result = await asyncio.wait_for(
                session.call_tool("echo", {"text": "hello"}),  # no meta
                timeout=10.0,
            )

            # The SDK's @focal_required guard surfaces as a tool-level
            # error rather than a transport exception.
            text_blocks = [c.text for c in result.content if hasattr(c, "text")]
            payload = " ".join(text_blocks)
            assert "focal_channel_path" in payload or "focal channel" in payload, (
                f"expected focal-channel error, got: {payload!r}"
            )
        finally:
            await registry.shutdown()
