"""run_sandbox: the worker-side ``sandbox()`` executor for a workflow run (#988).

Pure in-memory — a fake registry stands in for the real provision/exec path, and
the signal-persist + wake are patched so the task's behaviour (envelope shape,
timeout/output resolution, _INFLIGHT cleanup) is the surface under test.
"""

from __future__ import annotations

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
    ) -> None:
        self.result = result or CommandResult(
            exit_code=0, stdout="hi\n", stderr="", timed_out=False, truncated=False
        )
        self.provision_raises = provision_raises
        self.exec_raises = exec_raises
        self.exec_calls: list[dict[str, Any]] = []
        self.provision_count = 0

    async def get_or_provision_run(self, run_id: str, *, pool: Any = None) -> SandboxHandle:
        self.provision_count += 1
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
