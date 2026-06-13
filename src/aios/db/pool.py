"""asyncpg connection pool helpers.

The API and worker processes each construct their own pool via
:func:`create_pool` at startup and stash it on their own state
(``app.state.pool`` / ``runtime.pool``).  Tests do the same.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg

from aios.config import get_settings
from aios.logging import get_logger

_KNOWN_POOL_COUNT = 2  # API pool + worker pool (production call sites)
log = get_logger("aios.db.pool")

LISTENER_TCP_KEEPALIVE_SETTINGS = {
    "tcp_keepalives_idle": "60",
    "tcp_keepalives_interval": "10",
    "tcp_keepalives_count": "5",
}

_POOL_TCP_KEEPALIVE_SETTINGS = LISTENER_TCP_KEEPALIVE_SETTINGS


def normalize_dsn(db_url: str) -> str:
    """Strip SQLAlchemy/alembic driver prefixes; asyncpg wants bare ``postgresql://``."""
    for prefix in ("postgresql+asyncpg://", "postgresql+psycopg://"):
        if db_url.startswith(prefix):
            return "postgresql://" + db_url[len(prefix) :]
    return db_url


def listener_application_name(instance_id: str | None = None) -> str:
    """Postgres ``application_name`` tag for dedicated SSE/notify listener conns.

    Single source of truth shared by the listener connect path
    (:func:`aios.db.listen._connect_listener`) and the e2e leak test, which
    filters ``pg_stat_activity`` by this exact label so its backend count is
    scoped to THIS aios instance's listeners — robust to concurrent backends
    from other xdist workers / the app pool. ``instance_id`` defaults to
    ``get_settings().instance_id``; passed explicitly only by tests.
    Truncated to 63 characters; ``instance_id`` is ASCII (Settings pattern
    ``^[a-z_][a-z0-9_]*$``), so the result stays within Postgres's 63-byte
    ``application_name`` limit (``NAMEDATALEN - 1``).
    """
    iid = instance_id if instance_id is not None else get_settings().instance_id
    return f"aios-listener:{iid}"[:63]


def _jsonb_encoder(value: Any) -> str:
    """Encode a Python value for a JSONB bind parameter.

    A bare ``str`` is assumed to be **already-serialized JSON text** and passes
    through untouched; anything else is ``json.dumps``'d. This is the load-bearing
    detail that makes the codec land additively (Stage 1 of #1062): the existing
    ~57 write sites still pre-serialize with ``json.dumps(...)`` + ``$N::jsonb``,
    and asyncpg's codec encoder runs *on top of* that bound value — a plain
    ``json.dumps`` encoder would double-encode every current write (store
    ``"{\\"a\\": 1}"`` instead of ``{"a": 1}``). Passthrough-on-``str`` keeps every
    pre-serialized write single-encoded while already accepting the bare
    dicts/lists the Stage 2 sweep will switch writes to.

    The passthrough is unambiguous because no JSONB column is ever written a bare
    top-level JSON *string* value — every write is an object or array — so a
    ``str`` at this boundary is always serialized JSON, never a scalar payload.
    When Stage 2 retires the pre-serialization, this collapses to plain
    ``json.dumps``.
    """
    return value if isinstance(value, str) else json.dumps(value)


async def register_jsonb_codec(conn: asyncpg.Connection[Any]) -> None:
    """Make JSONB cross the seam as native Python on ``conn``.

    Used as the :func:`create_pool` ``init`` callback so every pooled connection
    carries the codec. Also exported so any code path that runs the
    ``aios.db.queries`` functions on a connection it opened directly (rather than
    acquiring from the pool) can opt the connection in — the query layer now
    relies on this codec for jsonb (``parse_jsonb`` is a pure passthrough), so a
    bare ``asyncpg.connect()`` running those functions MUST register it first or
    reads come back as raw JSON strings.

    Without a codec, asyncpg returns JSONB columns as raw JSON *strings* on read
    and demands pre-serialized strings on write, forcing every call site to
    hand-roll ``json.loads``/``json.dumps``. Registering the codec once enforces
    the impedance boundary at the connection instead of at ~140 vigilant call
    sites: reads return parsed Python (``json.loads`` decoder) and writes accept
    bare dicts/lists (:func:`_jsonb_encoder`).
    """
    await conn.set_type_codec(
        "jsonb",
        encoder=_jsonb_encoder,
        decoder=json.loads,
        schema="pg_catalog",
    )


# Back-compat alias: the original private name used as the pool ``init``.
_register_jsonb_codec = register_jsonb_codec


async def create_pool(db_url: str, *, min_size: int = 1, max_size: int = 8) -> asyncpg.Pool[Any]:
    """Create a new asyncpg pool against ``db_url``."""
    # asyncpg exposes no client-side keepalive kwarg, so the statement/idle
    # timeouts and TCP keepalive are applied as Postgres USERSET GUCs on every
    # pooled connection — bounding runaway scans, idle-in-txn leaks, and dead
    # connections behind a silently-dropped TCP link.
    pool = await asyncpg.create_pool(
        dsn=normalize_dsn(db_url),
        min_size=min_size,
        max_size=max_size,
        init=register_jsonb_codec,
        server_settings={
            "statement_timeout": "30000",
            "idle_in_transaction_session_timeout": "60000",
            **_POOL_TCP_KEEPALIVE_SETTINGS,
        },
    )
    if pool is None:
        raise RuntimeError(f"asyncpg.create_pool returned None for {db_url}")
    async with pool.acquire() as conn:
        pg_max_connections = int(await conn.fetchval("SHOW max_connections"))
    total_pool_capacity = max_size * _KNOWN_POOL_COUNT
    if total_pool_capacity >= pg_max_connections:
        log.warning(
            "db.pool.unsafe_max_size",
            max_size=max_size,
            known_pool_count=_KNOWN_POOL_COUNT,
            total_pool_capacity=total_pool_capacity,
            pg_max_connections=pg_max_connections,
        )
    return pool
