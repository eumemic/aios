"""The driver's action-failure currency and the shared envelope helpers.

Guards, ref resolution, and action handlers raise :class:`ActionError` with a
wire error code from ``browser_protocol.ERROR_CODES``; the host's ``handle``
boundary is the currency's total sink — it turns any :class:`ActionError`
into an ``ok: false`` envelope (an action response also attaches a fresh
snapshot so the model can self-correct). ``guardrail`` marks the credential
guardrail's refusals for logging.
"""

from __future__ import annotations

from typing import Any

from aios_browser_driver.browser_protocol import BrowserError, BrowserResponse

_ERROR_MESSAGE_MAX = 200


class ActionError(Exception):
    def __init__(self, code: str, message: str, *, guardrail: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.guardrail = guardrail


def first_line(exc: BaseException) -> str:
    """The exception's first line, truncated — the model-visible error text.

    One home for the wire error-message policy (dispatch, host, and actions all
    format exceptions this way)."""
    text = str(exc)
    return text.splitlines()[0][:_ERROR_MESSAGE_MAX] if text else type(exc).__name__


def error_response(boot: str, epoch: int, code: str, message: str) -> BrowserResponse:
    """A bare ``ok: false`` envelope — no page observation. Shared by dispatch
    (protocol-shape errors) and the host (control-op failures)."""
    return BrowserResponse(
        ok=False, boot=boot, epoch=epoch, error=BrowserError(code=code, message=message)
    )


# ── request-argument helpers (raise the currency; the boundary envelopes it) ─


def require_str(args: dict[str, Any], key: str, *, allow_empty: bool = False) -> str:
    value = args.get(key)
    if not isinstance(value, str) or (not value and not allow_empty):
        suffix = "a string" if allow_empty else "a non-empty string"
        raise ActionError("invalid_request", f"{key!r} must be {suffix}")
    return value


def require_number(args: dict[str, Any], key: str) -> float:
    value = args.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ActionError("invalid_request", f"{key!r} must be a number")
    return float(value)


def require_int(args: dict[str, Any], key: str, *, lo: int, hi: int) -> int:
    """An integer in ``[lo, hi]``, accepting an integral float (``2.0``).

    Draft 2020-12 ``"integer"`` validation — which the worker's tool schema
    gate applies — accepts ``2.0``; rejecting it here would make the two
    validation layers of one pipeline disagree."""
    value = args.get(key)
    if isinstance(value, bool):
        raise ActionError("invalid_request", f"{key!r} must be an integer from {lo} to {hi}")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int) or not lo <= value <= hi:
        raise ActionError("invalid_request", f"{key!r} must be an integer from {lo} to {hi}")
    return value
