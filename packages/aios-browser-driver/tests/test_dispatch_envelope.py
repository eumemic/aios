"""Dispatch must turn any request line into exactly one valid response
document — ok/boot/epoch always present, handler faults degraded to typed
error codes, never a crash."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

import pytest
from aios_browser_driver.browser_protocol import BrowserRequest, BrowserResponse
from aios_browser_driver.dispatch import dispatch

Handler = Callable[[BrowserRequest, float], Awaitable[BrowserResponse]]


class _Host:
    def __init__(self, handler: Handler | None = None) -> None:
        self.boot = "01BOOTULIDBOOTULIDBOOTUL"
        self.epoch = 3
        self._handler = handler

    async def handle(self, request: BrowserRequest, *, deadline: float) -> BrowserResponse:
        if self._handler is not None:
            return await self._handler(request, deadline)
        return BrowserResponse(ok=True, boot=self.boot, epoch=self.epoch, url="https://ok.test")


async def _run(raw: str, host: _Host | None = None) -> BrowserResponse:
    host = host or _Host()
    return BrowserResponse.model_validate_json(await dispatch(raw, host))


async def test_valid_op_is_routed_to_the_host() -> None:
    resp = await _run(json.dumps({"op": "status"}))
    assert resp.ok and resp.url == "https://ok.test"
    assert resp.boot == "01BOOTULIDBOOTULIDBOOTUL" and resp.epoch == 3


async def test_malformed_json_is_invalid_request() -> None:
    resp = await _run("{ not json")
    assert not resp.ok and resp.error is not None and resp.error.code == "invalid_request"


@pytest.mark.parametrize("raw", ['["op", "status"]', "42", '{"noop": 1}'])
async def test_non_object_or_missing_op_is_invalid_request(raw: str) -> None:
    resp = await _run(raw)
    assert resp.error is not None and resp.error.code == "invalid_request"


async def test_op_outside_the_vocabulary_is_unknown_op() -> None:
    resp = await _run(json.dumps({"op": "teleport"}))
    assert resp.error is not None and resp.error.code == "unknown_op"


@pytest.mark.parametrize("op", [["click"], {"op": "x"}, 42, None, True])
async def test_non_string_op_is_invalid_request_never_a_crash(op: object) -> None:
    # A list/dict op would make the `op not in _OPS` membership test raise
    # (unhashable) — dispatch must still yield one envelope, never crash.
    resp = await _run(json.dumps({"op": op}))
    assert resp.error is not None and resp.error.code == "invalid_request"


async def test_bad_arg_shape_is_invalid_request() -> None:
    resp = await _run(json.dumps({"op": "click", "args": ["not", "a", "dict"]}))
    assert resp.error is not None and resp.error.code == "invalid_request"


async def test_handler_notimplemented_is_unknown_op() -> None:
    async def _raise(_r: BrowserRequest, _d: float) -> BrowserResponse:
        raise NotImplementedError("click")

    resp = await _run(json.dumps({"op": "click"}), _Host(_raise))
    assert resp.error is not None and resp.error.code == "unknown_op"


async def test_handler_exception_is_internal_not_a_crash() -> None:
    async def _boom(_r: BrowserRequest, _d: float) -> BrowserResponse:
        raise RuntimeError("chromium exploded\nsecond line dropped")

    resp = await _run(json.dumps({"op": "snapshot"}), _Host(_boom))
    assert resp.error is not None and resp.error.code == "internal"
    assert resp.error.message == "chromium exploded"  # one line, no traceback


async def test_handler_overrun_is_action_timeout() -> None:
    async def _slow(_r: BrowserRequest, _d: float) -> BrowserResponse:
        await asyncio.sleep(10)
        raise AssertionError("should have been cancelled")

    # timeout_ms 2050 → dispatch delay = 0.05s (2.0s margin), so this is fast.
    resp = await _run(json.dumps({"op": "navigate", "timeout_ms": 2050}), _Host(_slow))
    assert resp.error is not None and resp.error.code == "action_timeout"


async def test_every_response_carries_boot_and_epoch() -> None:
    for raw in ("{bad", json.dumps({"op": "teleport"}), json.dumps({"op": "status"})):
        resp = await _run(raw)
        assert resp.boot == "01BOOTULIDBOOTULIDBOOTUL" and isinstance(resp.epoch, int)
