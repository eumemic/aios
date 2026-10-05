"""Rebuild a request a session sent (#2471).

A request span (``model_request_start`` for a native model call, the
``model_workflow_park`` record for a workflow-as-model launch) carries the
``request`` record :func:`aios.harness.request_capture.capture_request` wrote.
:func:`rebuild_request` recomposes that request from the record, its blobs and
the event log, using today's renderer: ``build_messages`` and
``finalize_messages``, the same two functions the step renders with.

The result is a kind:

* :class:`Rebuilt` with ``fidelity``:
  * ``"exact"``: the rebuild hashes to the captured ``payload_sha``. It is the
    request the session composed, which is not byte for byte the provider wire
    body: the send path then adds cache breakpoints and provider kwargs
    (``completion.call_litellm``), may strip media to fit a body limit, and a
    workflow-as-model run receives it as its input instead.
  * ``"inexact"``: same target model, different bytes. The renderer has changed
    since (compare ``render_version``), or an image the request inlined has
    changed on disk.
  * ``"rerendered"``: rendered for a different target model, so there's no hash
    to compare.
* :class:`Missing`: data the rebuild needs is gone (a blob, or an attachment
  file the request inlined).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import asyncpg

from aios.db import queries
from aios.db.queries import requests as request_queries
from aios.errors import NotFoundError
from aios.harness.context import build_messages, finalize_messages
from aios.harness.request_capture import sha256_hex
from aios.harness.window import WindowOmission

REQUEST_SPAN_EVENTS = frozenset({"model_request_start", "model_workflow_park"})


@dataclass(frozen=True, slots=True)
class Rebuilt:
    request: dict[str, Any]
    fidelity: Literal["exact", "inexact", "rerendered"]
    record: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Missing:
    what: str
    record: dict[str, Any]


def _decode(body: bytes) -> Any:
    return json.loads(body.decode("utf-8", "surrogatepass"))


async def rebuild_request(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    session_id: str,
    request_event_id: str,
    target_model: str | None = None,
) -> Rebuilt | Missing:
    """Rebuild the request the span ``request_event_id`` opened.

    ``target_model`` renders for another capability model, applying its vision
    and thinking gates to the same slate. Raises :class:`NotFoundError` when the
    session has no such span, or the span records no request (it predates
    capture, or it is an ``auto_review`` checker call).
    """
    async with pool.acquire() as conn:
        span = await queries.get_event(conn, session_id, request_event_id, account_id=account_id)
        record = span.data.get("request") if span.kind == "span" else None
        if (
            span.data.get("event") not in REQUEST_SPAN_EVENTS
            or not isinstance(record, dict)
            or "payload_sha" not in record
        ):
            raise NotFoundError(
                f"event {request_event_id} is not a captured request",
                detail={"id": request_event_id},
            )
        shas = [record["system_sha"], record["tools_sha"], record["params_sha"]]
        blobs = await request_queries.get_blobs(conn, account_id=account_id, shas=shas)
        if any(sha not in blobs for sha in shas):
            return Missing(what="blob", record=record)
        slate = record["slate"]
        # A ``None`` last seq is an empty slate: the step read no events.
        events = (
            []
            if slate["through_seq"] is None
            else await queries.read_windowed_context_events(
                conn,
                session_id,
                account_id=account_id,
                after_seq=slate["after_seq"],
                through_seq=slate["through_seq"],
            )
        )
        reminder_rows = await request_queries.get_events_by_seq(
            conn, session_id, account_id=account_id, seqs=list(record["reminder_seqs"])
        )
    # The bind source the send resolved ``/workspace`` attachments against. A rebuild
    # outside the worker (no worker filesystem) can't read them: it reports those
    # attachments Missing, or inexact.
    workspace_path = record.get("workspace_path")

    model = target_model or record["capability_model"]
    omission = record["omission"]
    ctx = await asyncio.to_thread(
        build_messages,
        events,
        system_prompt=_decode(blobs[record["system_sha"]]),
        model=model,
        session_id=session_id,
        workspace_path=None if workspace_path is None else Path(workspace_path),
        in_flight_tool_call_ids=frozenset(record["inflight_tool_call_ids"]),
        tz_name=record["tz"],
        omission=(
            None
            if omission is None
            else WindowOmission(
                began_at=datetime.fromisoformat(omission["began_at"]),
                omitted_messages=omission["omitted_messages"],
            )
        ),
    )
    messages = finalize_messages(
        ctx.messages,
        reminder_contents=tuple(row.data["content"] for row in reminder_rows),
        model=model,
    )
    tools = _decode(blobs[record["tools_sha"]])
    request = {
        "messages": messages,
        "tools": tools or None,
        "params": _decode(blobs[record["params_sha"]]),
    }
    if model != record["capability_model"]:
        return Rebuilt(request=request, fidelity="rerendered", record=record)
    if sha256_hex(request) == record["payload_sha"]:
        return Rebuilt(request=request, fidelity="exact", record=record)
    if ctx.unavailable_attachments:
        return Missing(what="attachment", record=record)
    return Rebuilt(request=request, fidelity="inexact", record=record)
