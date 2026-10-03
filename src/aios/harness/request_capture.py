"""The request a session composes, as bytes, and what a send records (#2471).

One encoding serves every request hash and every request blob, so a hash
computed when a request is sent and one computed when it is rebuilt agree
exactly when the requests do.

:func:`encode` is order-preserving (tool order and key order are part of the
request), keeps non-ASCII characters as UTF-8, and is total over everything a
request can carry: NUL characters and lone surrogates from tool output or MCP
schemas, and NaN or infinite floats. That's why it can't reuse
``aios.workflows.determinism.canonical_json``, which sorts keys and rejects
NUL, lone surrogates and NaN because its values must fit in jsonb.

:data:`RENDER_VERSION` names the renderer that turns the event log into a
request (``build_messages`` and ``finalize_messages``). Bump it with any change
that alters rendered bytes. ``tests/unit/test_render_golden.py`` renders a fixed
log to a pinned hash, so a rendering change can't land without a deliberate
bump. Such a change also busts every session's prompt cache once, so it
deserves the attention.

:func:`capture_request` records a composed request: the parts the event log
doesn't hold go into per-account blobs (:func:`store_capture`), and the rest
onto the ``request`` record its span carries. ``aios.services.requests``
rebuilds the request from them.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncpg

    from aios.harness.completion import LlmRequest
    from aios.harness.window import WindowOmission
    from aios.models.agents import StepBinding

RENDER_VERSION = 1


def encode(value: Any) -> bytes:
    """The bytes a request value is hashed and stored as."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8", "surrogatepass"
    )


def sha256_hex(value: Any) -> str:
    """The content address of ``value``: sha256 over :func:`encode`."""
    return hashlib.sha256(encode(value)).hexdigest()


def payload(request: LlmRequest) -> dict[str, Any]:
    """The part of a request ``payload_sha`` covers: what the session composed."""
    return {"messages": request.messages, "tools": request.tools, "params": request.params}


@dataclass(frozen=True, slots=True)
class RequestCapture:
    """What one send records.

    ``record`` is the ``request`` object the span carries. Everything a rebuild
    needs that the event log doesn't hold is either on it or in ``blobs``, which are
    keyed by sha256 and stored per account.
    """

    record: dict[str, Any]
    blobs: dict[str, bytes]


def capture_request(
    request: LlmRequest,
    *,
    system_prompt: str,
    model: str,
    capability_model: str,
    binding: StepBinding,
    after_seq: int | None,
    through_seq: int | None,
    omission: WindowOmission | None,
    reminder_seqs: tuple[int, ...],
    in_flight_tool_call_ids: frozenset[str],
    tz_name: str,
) -> RequestCapture:
    """Record a composed request, before anything on the send path touches it.

    Pure CPU (it encodes the whole payload, images included), so callers run it
    off the event loop. ``params`` is the agent's ``litellm_extra`` as composed,
    before provider auth resolves: a key resolved from an ancestor account's
    model provider never reaches a blob.
    """
    from importlib.metadata import version

    bodies = {
        "system": encode(system_prompt),
        "tools": encode(request.tools or []),
        "params": encode(request.params),
    }
    shas = {part: hashlib.sha256(body).hexdigest() for part, body in bodies.items()}
    record: dict[str, Any] = {
        "render_version": RENDER_VERSION,
        "litellm_version": version("litellm"),
        "model": model,
        "capability_model": capability_model,
        "binding": binding.model_dump(mode="json"),
        "system_sha": shas["system"],
        "tools_sha": shas["tools"],
        "params_sha": shas["params"],
        "slate": {"after_seq": after_seq, "through_seq": through_seq},
        "omission": (
            None
            if omission is None
            else {
                "began_at": omission.began_at.isoformat(),
                "omitted_messages": omission.omitted_messages,
            }
        ),
        "reminder_seqs": list(reminder_seqs),
        "inflight_tool_call_ids": sorted(in_flight_tool_call_ids),
        "tz": tz_name,
        "payload_sha": sha256_hex(payload(request)),
    }
    return RequestCapture(record=record, blobs={shas[part]: body for part, body in bodies.items()})


# (account_id, sha256) pairs this worker has already stored. Blobs are never
# deleted while their account lives, so a hit means the INSERT can be skipped;
# the steady state, where the system prompt, tools and params don't change from
# step to step, writes nothing.
_STORED: OrderedDict[tuple[str, str], None] = OrderedDict()
_STORED_MAX = 4096


async def store_capture(
    pool: asyncpg.Pool[Any], capture: RequestCapture, *, account_id: str
) -> None:
    """Store the capture's blobs the account doesn't have yet. Must complete before
    the span that references them is written."""
    from aios.db.queries import requests as request_queries

    missing = {sha: body for sha, body in capture.blobs.items() if (account_id, sha) not in _STORED}
    if missing:
        async with pool.acquire() as conn:
            await request_queries.insert_blobs(conn, account_id=account_id, blobs=missing)
    for sha in capture.blobs:
        _STORED[(account_id, sha)] = None
        _STORED.move_to_end((account_id, sha))
    while len(_STORED) > _STORED_MAX:
        _STORED.popitem(last=False)


def forget_stored_blobs() -> None:
    """Drop this worker's record of stored blobs (for tests that reset the database)."""
    _STORED.clear()
