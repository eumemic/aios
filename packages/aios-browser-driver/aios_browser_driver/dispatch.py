"""Request dispatch: parse → validate → route → envelope.

Playwright-free by construction — it depends only on the :class:`Host`
protocol, so it unit-tests without a browser. Every path returns a valid
:class:`BrowserResponse` document (``ok``/``boot``/``epoch`` always present); a
handler that raises becomes an ``ok: false`` envelope, never a crash, so the
exit-0-iff-a-document contract holds.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Protocol, get_args

from aios_browser_driver.browser_protocol import (
    BrowserError,
    BrowserOp,
    BrowserRequest,
    BrowserResponse,
)

# The driver self-reports before the exec wrapper's SIGKILL: keep the handler's
# own deadline this far below the request's ``timeout_ms``.
_DEADLINE_MARGIN_S = 2.0
_ERROR_MESSAGE_MAX = 200

_OPS: frozenset[str] = frozenset(get_args(BrowserOp))


class Host(Protocol):
    """What dispatch needs from the browser host. PR2's ``BrowserHost`` and
    PR1's skeleton host both satisfy it structurally."""

    boot: str
    epoch: int

    async def handle(self, request: BrowserRequest, *, deadline: float) -> BrowserResponse: ...


def _short(exc: BaseException) -> str:
    text = str(exc)
    return text.splitlines()[0][:_ERROR_MESSAGE_MAX] if text else type(exc).__name__


def _error(host: Host, code: str, message: str) -> BrowserResponse:
    return BrowserResponse(
        ok=False, boot=host.boot, epoch=host.epoch, error=BrowserError(code=code, message=message)
    )


async def dispatch(raw: str, host: Host) -> str:
    """Turn one request line into one response line."""
    return (await _dispatch(raw, host)).model_dump_json()


async def _dispatch(raw: str, host: Host) -> BrowserResponse:
    try:
        doc = json.loads(raw)
    except ValueError:
        return _error(host, "invalid_request", "request was not valid JSON")
    if not isinstance(doc, dict) or "op" not in doc:
        return _error(host, "invalid_request", "request must be an object carrying an 'op'")
    op = doc["op"]
    # Membership on a non-hashable op (list/dict) would raise, not envelope —
    # guard the type so every request line still yields one response document.
    if not isinstance(op, str):
        return _error(host, "invalid_request", "op must be a string")
    if op not in _OPS:
        return _error(host, "unknown_op", f"unknown op: {op!r}")
    try:
        request = BrowserRequest.model_validate(doc)
    except ValueError as exc:  # pydantic ValidationError is a ValueError
        return _error(host, "invalid_request", _short(exc))

    delay = max(0.0, request.timeout_ms / 1000.0 - _DEADLINE_MARGIN_S)
    deadline = time.monotonic() + delay
    try:
        async with asyncio.timeout(delay):
            return await host.handle(request, deadline=deadline)
    except TimeoutError:
        return _error(host, "action_timeout", f"{request.op} exceeded {request.timeout_ms}ms")
    except NotImplementedError:
        return _error(host, "unknown_op", f"{request.op} is not implemented by this driver build")
    except Exception as exc:  # a handler fault must degrade to an envelope, never crash
        return _error(host, "internal", _short(exc))
