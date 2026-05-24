"""Stand-in for ``whatsapp-daemon`` used in unit tests.

Speaks the same wire protocol (line-delimited JSON-RPC 2.0 over TCP) as
the production Go daemon.  Implements: ``version``, ``subscribe``,
``crash``, ``notify``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys


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
                resp: dict[str, object] = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"name": "whatsapp-daemon-fake", "version": "test"},
                }
            elif method == "subscribe":
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"status": "subscribed"},
                }
            elif method == "crash":
                writer.close()
                # Defer the exit so any in-flight write flushes first.
                asyncio.get_running_loop().call_later(0.05, sys.exit, 1)
                return
            elif method == "notify":
                # Emit a daemon-initiated notification on the same socket
                # so listener fixtures can observe it.
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
    p.add_argument("-version", action="store_true")
    args = p.parse_args()

    if args.version:
        print("whatsapp-daemon-fake test")
        return

    host_str, port_str = args.listen.rsplit(":", 1)
    asyncio.run(_serve(host_str, int(port_str)))


if __name__ == "__main__":
    main()
