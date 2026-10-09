"""A server-side FATAL mid-transaction must not leak a pool holder (#2540).

Prod fingerprint (worker 612d6a30, 2026-10-09): Postgres killed an
idle-in-transaction pooled connection; the next statement on it raised
``InternalClientError: cannot switch to state 15; another operation (2) is in
progress``; afterwards the pool showed ``in_use 1`` with ``idle_size == size``
— a holder still marked checked-out whose connection is closed. Each such
event permanently removes one of ``db_pool_max_size`` slots until restart.

Mechanism (asyncpg 0.31.0): the FATAL ErrorResponse arrives while the protocol
is idle, so the protocol enters ERROR_CONSUME. The next query fails
``_set_state`` → ``_coreproto_error()`` → ``abort()``, which sets
``closing=True``; ``connection_lost`` then takes the "closing" branch and never
runs ``Connection._cleanup()``, and ``PoolConnectionHolder.release()`` returns
early on ``is_closed()``. Fixed in asyncpg 0.32.0 (MagicStack/asyncpg#1324:
``release()`` terminates the closed connection, which re-queues the holder).

Why the TCP relay: on a loopback/unix socket the server's FIN lands in the same
read as the FATAL, so asyncpg sees ``connection_lost`` first and cleans up
correctly — the bug needs the next query to be sent *before* the FIN is
observed, as happens across a real network hop. The relay forwards bytes
unchanged and only delays propagating the server's close to the client.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pytest

from aios.db.pool import create_pool, normalize_dsn

pytestmark = [pytest.mark.integration]

_MAX_SIZE = 2
_CLOSE_LAG_S = 1.0
_BOUND_S = 5.0


async def _pipe(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, close_lag: float
) -> None:
    with contextlib.suppress(Exception):
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    if close_lag:
        await asyncio.sleep(close_lag)
    writer.close()


@contextlib.asynccontextmanager
async def _lagging_close_relay(db_url: str) -> AsyncIterator[str]:
    """Yield a DSN that routes through a relay delaying server-close by ``_CLOSE_LAG_S``."""
    parts = urlsplit(normalize_dsn(db_url))
    up_host, up_port = parts.hostname or "localhost", parts.port or 5432
    tasks: set[asyncio.Task[Any]] = set()

    async def handle(cr: asyncio.StreamReader, cw: asyncio.StreamWriter) -> None:
        ur, uw = await asyncio.open_connection(up_host, up_port)
        await asyncio.gather(_pipe(cr, uw, 0), _pipe(ur, cw, _CLOSE_LAG_S))

    def on_connect(cr: asyncio.StreamReader, cw: asyncio.StreamWriter) -> None:
        t = asyncio.ensure_future(handle(cr, cw))
        tasks.add(t)
        t.add_done_callback(tasks.discard)

    server = await asyncio.start_server(on_connect, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    userinfo = parts.netloc.rpartition("@")[0]
    netloc = f"{userinfo}@127.0.0.1:{port}" if userinfo else f"127.0.0.1:{port}"
    try:
        yield urlunsplit(parts._replace(netloc=netloc))
    finally:
        server.close()
        for t in list(tasks):
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _in_use(pool: Any) -> int:
    return sum(h._in_use is not None for h in pool._holders)  # asyncpg internals, audited


@pytest.mark.asyncio
async def test_server_fatal_mid_transaction_returns_holder_to_pool(db_url: str) -> None:
    async with _lagging_close_relay(db_url) as relay_url:
        pool = await create_pool(relay_url, min_size=_MAX_SIZE, max_size=_MAX_SIZE)
        try:
            with pytest.raises(Exception):  # noqa: B017 -- any error; the leak is the subject
                async with pool.acquire() as conn, conn.transaction():
                    await conn.execute("SET LOCAL idle_in_transaction_session_timeout = '200ms'")
                    await asyncio.sleep(0.6)  # server FATALs the idle-in-txn backend
                    await conn.execute("SELECT 1")
            # Let the lagged FIN arrive so any close-driven cleanup has run.
            await asyncio.sleep(_CLOSE_LAG_S + 0.5)

            assert _in_use(pool) == 0, (
                f"holder leaked: in_use={_in_use(pool)} size={pool.get_size()} "
                f"idle={pool.get_idle_size()}"
            )
            async with asyncio.timeout(_BOUND_S):
                conns = [await pool.acquire() for _ in range(_MAX_SIZE)]
            try:
                for c in conns:
                    assert await c.fetchval("SELECT 1") == 1
            finally:
                for c in conns:
                    await pool.release(c)
        finally:
            # A leaked holder makes Pool.close() wait forever on it; terminate
            # so a RED run fails on the assertion above, not on teardown.
            try:
                async with asyncio.timeout(_BOUND_S):
                    await pool.close()
            except TimeoutError:
                pool.terminate()
