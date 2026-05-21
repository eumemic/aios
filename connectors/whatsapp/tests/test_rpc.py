"""Tests for rpc.py — RpcClient (fresh conn per call) + RpcListener (persistent)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from typing import Any

import pytest

from aios_whatsapp.errors import ListenerClosedError, RpcError, RpcTimeoutError
from aios_whatsapp.rpc import RpcClient, RpcListener

Handler = Callable[[asyncio.StreamReader, asyncio.StreamWriter], Coroutine[Any, Any, None]]


async def _start_server(handler: Handler) -> tuple[asyncio.Server, int]:
    server = await asyncio.start_server(handler, host="127.0.0.1", port=0)
    port = server.sockets[0].getsockname()[1]
    return server, port


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
        with pytest.raises(RpcError, match="method not found") as ei:
            await client.call("bogus")
        assert ei.value.code == -32601


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


async def test_rpc_listener_yields_method_and_params() -> None:
    """The listener generalises Signal's hard-coded ``"receive"``: it
    yields ``(method, params)`` for every JSON-RPC notification frame so
    the connector can dispatch on method itself.
    """

    async def handler(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        for method, params in [
            ("message", {"id": "m1", "text": "hello"}),
            ("reaction", {"target_id": "m1", "emoji": "👍"}),
            ("connectionState", {"state": "connected"}),
        ]:
            notification = {"jsonrpc": "2.0", "method": method, "params": params}
            w.write(json.dumps(notification).encode() + b"\n")
            await w.drain()
        w.close()

    server, port = await _start_server(handler)
    async with server:
        listener = RpcListener("127.0.0.1", port)
        await listener.connect()
        seen: list[tuple[str, dict[str, Any]]] = []
        with pytest.raises(ListenerClosedError):
            async for pair in listener.notifications():
                seen.append(pair)
        assert seen == [
            ("message", {"id": "m1", "text": "hello"}),
            ("reaction", {"target_id": "m1", "emoji": "👍"}),
            ("connectionState", {"state": "connected"}),
        ]
        await listener.aclose()


async def test_rpc_listener_ignores_responses_and_malformed_frames() -> None:
    """Stray RPC responses (frames with ``id``), frames with missing /
    non-string method, and frames with non-dict params are dropped; the
    listener keeps reading.
    """

    async def handler(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        for msg in [
            {"jsonrpc": "2.0", "id": 99, "result": {"ignored": True}},  # response
            {"jsonrpc": "2.0", "method": 7, "params": {}},  # non-string method
            {"jsonrpc": "2.0", "method": "bad", "params": "not-a-dict"},  # bad params
            {"jsonrpc": "2.0", "method": "good", "params": {"k": "v"}},  # OK
        ]:
            w.write(json.dumps(msg).encode() + b"\n")
            await w.drain()
        w.close()

    server, port = await _start_server(handler)
    async with server:
        listener = RpcListener("127.0.0.1", port)
        await listener.connect()
        seen: list[tuple[str, dict[str, Any]]] = []
        with pytest.raises(ListenerClosedError):
            async for pair in listener.notifications():
                seen.append(pair)
        assert seen == [("good", {"k": "v"})]
        await listener.aclose()


async def test_rpc_client_fails_cleanly_when_server_closed(unused_port: int) -> None:
    client = RpcClient("127.0.0.1", unused_port, timeout=2.0)
    with pytest.raises(RpcError):
        await client.call("ping")


async def test_rpc_listener_connect_failure(unused_port: int) -> None:
    listener = RpcListener("127.0.0.1", unused_port)
    with pytest.raises(ListenerClosedError):
        await listener.connect()
