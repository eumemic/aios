"""A stand-in for the real Go ``whatsapp-daemon`` used in unit tests.

Speaks the same wire protocol as the production daemon (line-delimited
JSON-RPC 2.0 over TCP) so the Python side's spawn → readiness-probe →
RPC-call → notification-stream code path can be exercised end-to-end
without a Go toolchain in CI.

Only implements the methods PR-1 tests need (``version``, ``echo``,
``crash``).  Subsequent PRs may extend this — or replace it with a
``go build`` fixture once the Go daemon grows whatsmeow integration.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any


async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            line = await reader.readline()
            if not line:
                return
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                return
            method = req.get("method")
            params = req.get("params") or {}
            req_id = req.get("id")

            if method == "version":
                resp: dict[str, Any] = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"name": "whatsapp-daemon-fake", "version": "test"},
                }
            elif method == "echo":
                resp = {"jsonrpc": "2.0", "id": req_id, "result": params}
            elif method == "crash":
                # Used by daemon-crash tests: terminate the process so the
                # Python side's `crashed()` future fires.
                writer.close()
                # Defer the exit so the response (if any) flushes first.
                asyncio.get_running_loop().call_later(0.05, sys.exit, 1)
                return
            elif method == "notify":
                # `notify` triggers a daemon-initiated notification on the
                # SAME socket so the listener fixture can observe it.
                notif = {
                    "jsonrpc": "2.0",
                    "method": params.get("method", "message"),
                    "params": params.get("params") or {},
                }
                writer.write(json.dumps(notif).encode() + b"\n")
                await writer.drain()
                continue
            else:
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"method not found: {method}"},
                }

            writer.write(json.dumps(resp).encode() + b"\n")
            await writer.drain()
    finally:
        writer.close()


async def _serve(host: str, port: int) -> None:
    server = await asyncio.start_server(_handle, host=host, port=port)
    async with server:
        await server.serve_forever()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-listen", required=True)
    p.add_argument("-store-dir", required=True)
    p.add_argument("-log-level", default="info")
    # ``-version`` is the only flag mode the real daemon supports as an
    # early exit; mirror it so fixtures that probe ``--version`` succeed
    # without spinning up the server.
    p.add_argument("-version", action="store_true")
    args = p.parse_args()

    if args.version:
        print("whatsapp-daemon-fake test")
        return

    host_str, port_str = args.listen.rsplit(":", 1)
    asyncio.run(_serve(host_str, int(port_str)))


if __name__ == "__main__":
    main()
