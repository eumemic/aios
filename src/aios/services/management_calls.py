"""Operator→connector RPC plane (#348).

The api process is the public face of management operations that today
require SSHing into a connector container (signal-cli's ``register``,
``verify``, ``updateProfile``).  ``submit_call`` is the single primitive
that operator-facing routes call: it inserts a pending row, NOTIFYs the
runtime container's SSE channel, and blocks on a per-call result channel
until the connector POSTs the answer back.

The wakeup primitive is :func:`aios.db.listen.listen_for_connector_result`
— a dedicated LISTEN on ``connector_result_<call_id>`` that yields a
one-shot queue.  The api never holds an in-process ``asyncio.Future``
keyed by ``call_id``; the channel-per-call pattern means multi-process
deployments work without coordination.

Sibling of :mod:`aios.services.sessions` but per-connector-type, not
per-session.  The runtime side is :class:`HttpConnector._management_call_loop`
in ``packages/aios-connector-http``.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg

from aios.db import listen, queries
from aios.errors import ManagementCallTimeoutError
from aios.ids import make_id

# Slack on top of the request-side timeout: the row outlives the request
# briefly so a connector POSTing right at the deadline doesn't UPDATE a
# row the operator has already given up on.  The connector still won't
# get a successful operator wake, but the audit row reflects the truth.
_EXPIRY_SLACK_S: float = 5.0


async def submit_call(
    db_url: str,
    pool: asyncpg.Pool[asyncpg.Record],
    *,
    connector: str,
    method: str,
    params: dict[str, Any],
    timeout_s: float,
) -> tuple[Any, bool]:
    """Submit a management call and block until the connector resolves it.

    Returns ``(result, is_error)``.  ``result`` is whatever the connector
    POSTed (signal-cli wrappers post ``{"status": "sms_sent", ...}``;
    captcha-required posts ``{"status": "captcha_required", ...}`` with
    ``is_error=true``).  Raises :class:`ManagementCallTimeoutError` if the
    runtime container doesn't POST within ``timeout_s``.

    Ordering is load-bearing: LISTEN before INSERT.  If the NOTIFY
    fired before the LISTEN was wired up, we'd miss the wake and time
    out despite the connector having delivered.  Identical invariant
    to the SSE handlers in :mod:`aios.api.sse`.
    """
    call_id = make_id("mgmt")
    expires_at = datetime.now(UTC) + timedelta(seconds=timeout_s + _EXPIRY_SLACK_S)
    channel = f"connector_management_calls_{connector}"

    async with listen.listen_for_connector_result(db_url, call_id) as queue:
        async with pool.acquire() as conn:
            await queries.insert_management_call(
                conn,
                call_id=call_id,
                connector=connector,
                method=method,
                params=params,
                expires_at=expires_at,
            )
            # pg_notify in function form (the literal NOTIFY case-folds
            # unquoted identifiers, and mixed-case channel names break
            # asyncpg's quoted-LISTEN; see queries.append_event for the
            # full rationale).
            await conn.execute("SELECT pg_notify($1, $2)", channel, call_id)

        try:
            payload = await asyncio.wait_for(queue.get(), timeout=timeout_s)
        except TimeoutError as exc:
            raise ManagementCallTimeoutError(
                f"connector {connector!r} did not resolve {method!r} within {timeout_s}s",
                detail={"call_id": call_id, "connector": connector, "method": method},
            ) from exc

    envelope = json.loads(payload)
    return envelope["result"], bool(envelope["is_error"])
