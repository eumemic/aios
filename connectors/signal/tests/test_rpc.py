"""Tests for rpc.py — RpcClient (fresh conn per call) + RpcListener (persistent)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from aios_signal.errors import ListenerClosedError, RpcError, RpcTimeoutError
from aios_signal.rpc import RpcClient, RpcListener
from tests.conftest import _start_server


async def test_rpc_client_opens_fresh_connection_per_call() -> None:
    connection_count = 0

    async def handler(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        nonlocal connection_count
        connection_count += 1
        line = await r.readline()
        req = json.loads(line)
        response = {"jsonrpc": "2.0", "id": req["id"], "result": {"ok": True}}
        w.write(json.dumps(response).encode() + b"\n")
        await w.drain()
        w.close()

    server, port = await _start_server(handler)
    async with server:
        client = RpcClient("127.0.0.1", port)
        r1, r2, r3 = await asyncio.gather(
            client.call("method_a"),
            client.call("method_b"),
            client.call("method_c"),
        )
    assert r1 == r2 == r3 == {"ok": True}
    assert connection_count == 3  # fresh conn per call


async def test_rpc_client_propagates_error_field() -> None:
    async def handler(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        line = await r.readline()
        req = json.loads(line)
        response = {
            "jsonrpc": "2.0",
            "id": req["id"],
            "error": {"code": -32601, "message": "method not found"},
        }
        w.write(json.dumps(response).encode() + b"\n")
        await w.drain()
        w.close()

    server, port = await _start_server(handler)
    async with server:
        client = RpcClient("127.0.0.1", port)
        with pytest.raises(RpcError, match="method not found"):
            await client.call("bogus")


async def test_rpc_client_timeout() -> None:
    async def handler(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        await r.readline()
        # Sleep longer than the client's timeout but short enough that
        # `async with server` teardown doesn't stall the test.
        await asyncio.sleep(1.0)

    server, port = await _start_server(handler)
    async with server:
        client = RpcClient("127.0.0.1", port, timeout=0.1)
        with pytest.raises(RpcTimeoutError):
            await client.call("slow")


async def test_rpc_listener_yields_account_envelope_pairs() -> None:
    """In multi-account daemon mode every receive notification carries
    ``params.account`` — the listener exposes it alongside the envelope
    so the connector can route inbound to the right per-phone bot UUID.
    """

    async def handler(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        for i in range(3):
            notification = {
                "jsonrpc": "2.0",
                "method": "receive",
                "params": {
                    "account": f"+155500000{i:02d}",
                    "envelope": {"timestamp": i, "sourceUuid": f"u{i}"},
                },
            }
            w.write(json.dumps(notification).encode() + b"\n")
            await w.drain()
        w.close()

    server, port = await _start_server(handler)
    async with server:
        listener = RpcListener("127.0.0.1", port)
        await listener.connect()
        pairs: list[tuple[str, dict[str, Any]]] = []
        with pytest.raises(ListenerClosedError):
            async for pair in _take_then_wait(listener.messages()):
                pairs.append(pair)
        assert len(pairs) == 3
        assert pairs[0] == ("+15550000000", {"timestamp": 0, "sourceUuid": "u0"})
        assert pairs[2] == ("+15550000002", {"timestamp": 2, "sourceUuid": "u2"})
        await listener.aclose()


async def _take_then_wait(
    it: AsyncIterator[tuple[str, dict[str, Any]]],
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    async for pair in it:
        yield pair


async def test_rpc_listener_drops_notifications_without_account() -> None:
    """Multi-account contract violation: a receive without ``account`` is
    a connector-side bug (signal-cli stamps it in multi-account mode).
    Drop with a warning rather than guessing.
    """

    async def handler(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        # Stray RPC response, non-receive notification, receive WITHOUT
        # account (must be dropped), and finally a real receive WITH account.
        for msg in [
            {"jsonrpc": "2.0", "id": 99, "result": {"ignored": True}},
            {"jsonrpc": "2.0", "method": "other", "params": {}},
            {
                "jsonrpc": "2.0",
                "method": "receive",
                "params": {"envelope": {"timestamp": 1}},  # no account
            },
            {
                "jsonrpc": "2.0",
                "method": "receive",
                "params": {"account": "+15550001111", "envelope": {"timestamp": 2}},
            },
        ]:
            w.write(json.dumps(msg).encode() + b"\n")
            await w.drain()
        w.close()

    server, port = await _start_server(handler)
    async with server:
        listener = RpcListener("127.0.0.1", port)
        await listener.connect()
        pairs: list[tuple[str, dict[str, Any]]] = []
        with pytest.raises(ListenerClosedError):
            async for pair in listener.messages():
                pairs.append(pair)
        # Only the well-formed receive survives.
        assert pairs == [("+15550001111", {"timestamp": 2})]
        await listener.aclose()


async def _unused_port() -> int:
    # Bind to an ephemeral port, then release it. Nothing's listening.
    server = await asyncio.start_server(lambda r, w: None, host="127.0.0.1", port=0)
    port: int = server.sockets[0].getsockname()[1]
    server.close()
    await server.wait_closed()
    return port


async def test_rpc_client_fails_cleanly_when_server_closed() -> None:
    port = await _unused_port()
    client = RpcClient("127.0.0.1", port, timeout=2.0)
    with pytest.raises(RpcError):
        await client.call("ping")


async def test_rpc_listener_connect_failure() -> None:
    port = await _unused_port()
    listener = RpcListener("127.0.0.1", port)
    with pytest.raises(ListenerClosedError):
        await listener.connect()
