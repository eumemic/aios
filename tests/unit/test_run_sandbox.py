"""run_sandbox: the worker-side ``sandbox()`` executor for a workflow run (#988).

Pure in-memory — a fake registry stands in for the real provision/exec path, and
the signal-persist + wake are patched so the task's behaviour (envelope shape,
timeout/output resolution, _INFLIGHT cleanup) is the surface under test.
"""

from __future__ import annotations

import shlex
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aios.config import get_settings
from aios.sandbox.backends.base import CommandResult, SandboxBackendError, SandboxHandle
from aios.workflows import run_sandbox
from aios.workflows.wf_script_host import sandbox


def _run() -> Any:
    """A WfRun stand-in carrying only the id the executor reads."""
    from types import SimpleNamespace

    return SimpleNamespace(id="wfr_1", account_id="acc_t")


def _handle() -> SandboxHandle:
    from pathlib import Path

    return SandboxHandle(owner_id="wfr_1", sandbox_id="cid", workspace_path=Path("/tmp/w"))


class _FakeRegistry:
    """Mimics the two registry verbs ``_execute`` uses. Records the exec kwargs and
    can be steered to raise on provision or exec."""

    def __init__(
        self,
        *,
        result: CommandResult | None = None,
        provision_raises: bool = False,
        exec_raises: bool = False,
        provision_exc: BaseException | None = None,
    ) -> None:
        self.result = result or CommandResult(
            exit_code=0, stdout="hi\n", stderr="", timed_out=False, truncated=False
        )
        self.provision_raises = provision_raises
        self.exec_raises = exec_raises
        # A non-SandboxBackendError to raise from provision (run-not-found ValueError,
        # asyncpg error, …) — the path the broad backstop must still turn into a signal.
        self.provision_exc = provision_exc
        self.exec_calls: list[dict[str, Any]] = []
        self.provision_count = 0

    async def get_or_provision_run(self, run_id: str, *, pool: Any = None) -> SandboxHandle:
        self.provision_count += 1
        if self.provision_exc is not None:
            raise self.provision_exc
        if self.provision_raises:
            raise SandboxBackendError("provision boom")
        return _handle()

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        timeout_seconds: int,
        max_output_bytes: int,
        cwd: str = "/workspace",
    ) -> CommandResult:
        self.exec_calls.append(
            {
                "command": command,
                "timeout_seconds": timeout_seconds,
                "max_output_bytes": max_output_bytes,
                "cwd": cwd,
            }
        )
        if self.exec_raises:
            raise SandboxBackendError("exec boom")
        return self.result


@pytest.fixture(autouse=True)
def _clear_inflight() -> Iterator[None]:
    run_sandbox._INFLIGHT.clear()
    yield
    run_sandbox._INFLIGHT.clear()


class _Acquire:
    async def __aenter__(self) -> Any:
        return object()

    async def __aexit__(self, *a: Any) -> bool:
        return False


class _Pool:
    """A fake asyncpg pool whose ``acquire()`` yields a throwaway conn — the
    insert_run_signal patch ignores the conn, so any object suffices."""

    def acquire(self) -> _Acquire:
        return _Acquire()


def _patches(registry: _FakeRegistry, captured: dict[str, Any]) -> Any:
    async def _capture_signal(
        conn: Any, *, run_id: str, call_key: str, kind: str, result: Any
    ) -> None:
        captured["kind"] = kind
        captured["result"] = result
        captured["call_key"] = call_key

    return (
        patch("aios.workflows.run_sandbox.runtime.require_sandbox_registry", return_value=registry),
        patch("aios.workflows.run_sandbox.runtime.require_pool", return_value=_Pool()),
        patch("aios.workflows.run_sandbox.wf_queries.insert_run_signal", new=_capture_signal),
        patch("aios.workflows.run_sandbox.defer_run_wake", new=AsyncMock()),
    )


def test_sandbox_verb_builds_capability() -> None:
    cap = sandbox("echo hi")
    assert cap._capability_id == "sandbox"
    assert cap._spec == {"command": "echo hi", "timeout_s": None}


async def test_sandbox_task_writes_ok_signal_on_success() -> None:
    registry = _FakeRegistry()
    captured: dict[str, Any] = {}
    p1, p2, p3, p4 = _patches(registry, captured)
    with p1, p2, p3, p4 as wake:
        await run_sandbox._run_sandbox_task(
            _Pool(), _run(), call_key="k0", command="echo hi", timeout_s=None
        )
    assert captured["kind"] == "sandbox_result"
    assert captured["result"] == {
        "ok": {
            "exit_code": 0,
            "stdout": "hi\n",
            "stderr": "",
            "timed_out": False,
            "truncated": False,
        }
    }
    wake.assert_awaited_once()
    # Tool-always-appends-result invariant: the in-flight entry is popped.
    assert ("wfr_1", "k0") not in run_sandbox._INFLIGHT


async def test_sandbox_task_writes_error_signal_on_infra_failure() -> None:
    registry = _FakeRegistry(exec_raises=True)
    captured: dict[str, Any] = {}
    p1, p2, p3, p4 = _patches(registry, captured)
    with p1, p2, p3, p4:
        await run_sandbox._run_sandbox_task(
            _Pool(), _run(), call_key="k0", command="boom", timeout_s=None
        )
    assert captured["kind"] == "sandbox_result"
    assert captured["result"]["error"]["capability"] == "sandbox"
    assert captured["result"]["error"]["code"] == "exec_failed"
    assert "exec boom" in captured["result"]["error"]["message"]


async def test_sandbox_task_provision_failure_is_provision_failed() -> None:
    registry = _FakeRegistry(provision_raises=True)
    captured: dict[str, Any] = {}
    p1, p2, p3, p4 = _patches(registry, captured)
    with p1, p2, p3, p4:
        await run_sandbox._run_sandbox_task(
            _Pool(), _run(), call_key="k0", command="x", timeout_s=None
        )
    assert captured["result"]["error"]["code"] == "provision_failed"
    assert registry.exec_calls == []  # never reached exec


async def test_sandbox_task_non_backend_exception_still_signals() -> None:
    """A non-SandboxBackendError escaping provisioning (e.g. a ValueError from
    build_spec_from_run for a run-not-found / reserved-image-prefix) must NOT skip
    the signal — every path writes exactly one error signal (code provision_failed)
    and pops _INFLIGHT, upholding the always-signals invariant. Without this the
    stale-sandbox sweep horizon would park the run forever."""
    registry = _FakeRegistry(provision_exc=ValueError("run wfr_1 not found"))
    captured: dict[str, Any] = {}
    p1, p2, p3, p4 = _patches(registry, captured)
    with p1, p2, p3, p4 as wake:
        await run_sandbox._run_sandbox_task(
            _Pool(), _run(), call_key="k0", command="x", timeout_s=None
        )
    assert captured["kind"] == "sandbox_result"
    assert captured["result"]["error"]["capability"] == "sandbox"
    assert captured["result"]["error"]["code"] == "provision_failed"
    assert "run wfr_1 not found" in captured["result"]["error"]["message"]
    assert registry.exec_calls == []  # never reached exec
    wake.assert_awaited_once()
    assert ("wfr_1", "k0") not in run_sandbox._INFLIGHT


async def test_sandbox_task_default_timeout_and_max_output() -> None:
    settings = get_settings()
    registry = _FakeRegistry()
    captured: dict[str, Any] = {}
    p1, p2, p3, p4 = _patches(registry, captured)
    with p1, p2, p3, p4:
        # timeout_s=None → the worker-global bash default.
        await run_sandbox._run_sandbox_task(
            _Pool(), _run(), call_key="k0", command="x", timeout_s=None
        )
    call = registry.exec_calls[0]
    assert call["timeout_seconds"] == settings.bash_default_timeout_seconds
    assert call["max_output_bytes"] == settings.bash_max_output_bytes
    assert call["cwd"] == "/workspace"

    # A float timeout_s is truncated to int seconds (2.7 → 2).
    registry2 = _FakeRegistry()
    captured2: dict[str, Any] = {}
    q1, q2, q3, q4 = _patches(registry2, captured2)
    with q1, q2, q3, q4:
        await run_sandbox._run_sandbox_task(
            _Pool(), _run(), call_key="k1", command="x", timeout_s=2.7
        )
    assert registry2.exec_calls[0]["timeout_seconds"] == 2


async def test_sandbox_task_prepends_idempotency_preamble() -> None:
    """The command handed to ``backend.exec`` is prefixed with a shlex-quoted
    ``export AIOS_RUN_ID=… AIOS_CALL_KEY=…`` line, followed by the verbatim author
    command (#988 amendment). The run.id / call_key are exposed so an author can pass
    ``$AIOS_CALL_KEY`` to an external service as an idempotency key. The preamble is
    applied ONLY to the execed string — the journaled command is the author's."""
    registry = _FakeRegistry()
    captured: dict[str, Any] = {}
    # A call_key carrying the structural punctuation the deterministic keyer emits
    # (``/ . : #``) — shlex.quote must keep it shell-safe.
    call_key = "main/0.1/sha:deadbeef#2"
    p1, p2, p3, p4 = _patches(registry, captured)
    with p1, p2, p3, p4:
        await run_sandbox._run_sandbox_task(
            _Pool(),
            _run(),
            call_key=call_key,
            command="curl -X POST https://api/charge",
            timeout_s=None,
        )
    execed = registry.exec_calls[0]["command"]
    expected_preamble = (
        f"export AIOS_RUN_ID={shlex.quote('wfr_1')} AIOS_CALL_KEY={shlex.quote(call_key)}\n"
    )
    assert execed == expected_preamble + "curl -X POST https://api/charge"
    # The original author command is unchanged inside the preamble-prefixed string.
    assert execed.endswith("curl -X POST https://api/charge")
