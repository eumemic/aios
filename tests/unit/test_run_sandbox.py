"""run_sandbox: the worker-side ``tool('bash', …)`` executor for a workflow run (#988).

Pure in-memory — a fake registry stands in for the real provision/exec path, and
the signal-persist + wake are patched, so the task's observable behaviour (the
bare-dict ``tool_result`` envelope, the recoverable ``{"error": …}`` values, the
hashed idempotency preamble, the timeout resolution, the shared-``_INFLIGHT``
cleanup) is the surface under test.

Bash now rides the existing ``tool`` capability, so the success result is the BARE
bash dict (``{exit_code, stdout, stderr, timed_out, truncated}``) and a failure is
a FLAT ``{"error": str}`` value — no ``{"ok"}/{"error"}`` envelope, no
``SandboxError``. The harvest folds ``sig.result`` straight into the call_result.
"""

from __future__ import annotations

import hashlib
import re
import shlex
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aios.config import get_settings
from aios.models.agents import ToolSpec
from aios.models.workflows import WORKFLOW_SCRIPT_CONTRACT
from aios.sandbox.backends.base import CommandResult, SandboxBackendError, SandboxHandle
from aios.workflows import run_sandbox, run_tools
from aios.workflows.idempotency_key import idempotency_key


def _run() -> Any:
    """A WfRun stand-in carrying only what the executor reads: id, account, and a
    declared+enabled ``bash`` tool so the surface gate admits the call."""
    return SimpleNamespace(
        id="wfr_1", account_id="acc_t", tools=[ToolSpec(type="bash")], http_servers=[]
    )


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
    # run_sandbox shares run_tools' _INFLIGHT — clear it both sides of every test.
    run_tools._INFLIGHT.clear()
    yield
    run_tools._INFLIGHT.clear()


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


async def _drive(
    registry: _FakeRegistry,
    captured: dict[str, Any],
    *,
    call_key: str = "k0",
    tool_input: Any,
) -> AsyncMock:
    """Run one _run_sandbox_task under the patches, returning the wake mock."""
    p1, p2, p3, p4 = _patches(registry, captured)
    with p1, p2, p3, p4 as wake:
        await run_sandbox._run_sandbox_task(
            _Pool(), _run(), call_key=call_key, tool_name="bash", tool_input=tool_input
        )
    assert isinstance(wake, AsyncMock)
    return wake


async def test_success_writes_bare_bash_dict_tool_result() -> None:
    registry = _FakeRegistry()
    captured: dict[str, Any] = {}
    wake = await _drive(registry, captured, tool_input={"command": "echo hi"})
    assert captured["kind"] == "tool_result"
    # BARE bash dict — no {"ok": …} envelope.
    assert captured["result"] == {
        "exit_code": 0,
        "stdout": "hi\n",
        "stderr": "",
        "timed_out": False,
        "truncated": False,
    }
    wake.assert_awaited_once()
    # Tool-always-appends-result invariant: the shared in-flight entry is popped.
    assert ("wfr_1", "k0") not in run_tools._INFLIGHT


async def test_provision_failure_is_recoverable_error_value() -> None:
    registry = _FakeRegistry(provision_raises=True)
    captured: dict[str, Any] = {}
    wake = await _drive(registry, captured, tool_input={"command": "x"})
    assert captured["kind"] == "tool_result"
    assert "error" in captured["result"]
    assert "provisioning failed" in captured["result"]["error"]
    assert "provision boom" in captured["result"]["error"]
    assert registry.exec_calls == []  # never reached exec
    wake.assert_awaited_once()


async def test_exec_failure_is_recoverable_error_value() -> None:
    registry = _FakeRegistry(exec_raises=True)
    captured: dict[str, Any] = {}
    await _drive(registry, captured, tool_input={"command": "boom"})
    assert captured["kind"] == "tool_result"
    assert "error" in captured["result"]
    assert "exec failed" in captured["result"]["error"]
    assert "exec boom" in captured["result"]["error"]


async def test_non_backend_exception_still_signals() -> None:
    """A non-SandboxBackendError escaping provisioning (e.g. a ValueError from
    build_spec_from_run for a run-not-found / reserved-image-prefix) must NOT skip
    the signal — every path writes exactly one tool_result and pops the shared
    _INFLIGHT, upholding the always-signals invariant."""
    registry = _FakeRegistry(provision_exc=ValueError("run wfr_1 not found"))
    captured: dict[str, Any] = {}
    wake = await _drive(registry, captured, tool_input={"command": "x"})
    assert captured["kind"] == "tool_result"
    assert "error" in captured["result"]
    assert "run wfr_1 not found" in captured["result"]["error"]
    assert registry.exec_calls == []
    wake.assert_awaited_once()
    assert ("wfr_1", "k0") not in run_tools._INFLIGHT


async def test_default_timeout_and_max_output() -> None:
    settings = get_settings()
    registry = _FakeRegistry()
    captured: dict[str, Any] = {}
    # timeout_seconds absent → the worker-global bash default.
    await _drive(registry, captured, tool_input={"command": "x"})
    call = registry.exec_calls[0]
    assert call["timeout_seconds"] == settings.bash_default_timeout_seconds
    assert call["max_output_bytes"] == settings.bash_max_output_bytes
    assert call["cwd"] == "/workspace"

    # A float timeout_seconds is truncated to int seconds (2.7 → 2).
    registry2 = _FakeRegistry()
    captured2: dict[str, Any] = {}
    await _drive(
        registry2, captured2, call_key="k1", tool_input={"command": "x", "timeout_seconds": 2.7}
    )
    assert registry2.exec_calls[0]["timeout_seconds"] == 2


async def test_subsecond_timeout_floors_to_one() -> None:
    """A positive sub-second ``timeout_seconds`` must NEVER resolve to 0 (#988 issue 1).

    The in-container ``timeout`` is GNU coreutils, which treats DURATION 0 as "no
    limit" — so a bare ``int(0.5) == 0`` would run the command UNBOUNDED. The
    ``max(1, int(timeout_seconds))`` floor guarantees a positive request is never
    disabled. This is the mutation guard: reverting to a bare ``int(timeout_seconds)``
    makes this fail (resolved seconds would be 0)."""
    for sub_second in (0.5, 0.001, 0.999):
        registry = _FakeRegistry()
        captured: dict[str, Any] = {}
        await _drive(registry, captured, tool_input={"command": "x", "timeout_seconds": sub_second})
        assert registry.exec_calls[0]["timeout_seconds"] == 1, sub_second


async def test_timeout_clamped_to_bash_ceiling() -> None:
    ceiling = get_settings().bash_default_timeout_seconds
    registry = _FakeRegistry()
    captured: dict[str, Any] = {}
    await _drive(
        registry, captured, tool_input={"command": "x", "timeout_seconds": float(ceiling + 500)}
    )
    assert registry.exec_calls[0]["timeout_seconds"] == ceiling


@pytest.mark.parametrize("bad", [0, -1, 0.0, "30", True, float("nan"), float("inf")])
async def test_invalid_timeout_is_recoverable_error_value(bad: Any) -> None:
    """A malformed ``timeout_seconds`` (non-positive / non-numeric / bool / NaN / inf)
    is a recoverable ``{"error": …}`` VALUE — bash is a tool, so tool() never raises;
    the run continues. exec is never reached."""
    registry = _FakeRegistry()
    captured: dict[str, Any] = {}
    wake = await _drive(registry, captured, tool_input={"command": "x", "timeout_seconds": bad})
    assert captured["kind"] == "tool_result"
    assert "error" in captured["result"]
    assert "timeout_seconds" in captured["result"]["error"]
    assert registry.exec_calls == []  # never reached exec
    wake.assert_awaited_once()


def test_workflow_script_contract_documents_timeout_seconds() -> None:
    assert "timeout_seconds" in WORKFLOW_SCRIPT_CONTRACT
    assert re.search(r"\btimeout_s\b", WORKFLOW_SCRIPT_CONTRACT) is None


async def test_missing_command_is_recoverable_error_value() -> None:
    registry = _FakeRegistry()
    captured: dict[str, Any] = {}
    await _drive(registry, captured, tool_input={})
    assert "error" in captured["result"]
    assert "command" in captured["result"]["error"]
    assert registry.exec_calls == []


async def test_prepends_hashed_idempotency_preamble() -> None:
    """The command handed to ``backend.exec`` is prefixed with a shlex-quoted
    ``export AIOS_RUN_ID=… AIOS_IDEMPOTENCY_KEY=…`` line — the idempotency key is
    the sha256 of ``run_id\\0call_key`` (NOT the raw call_key) — followed by the
    verbatim author command. The journaled command is the author's; only the
    execed string carries the preamble."""
    registry = _FakeRegistry()
    captured: dict[str, Any] = {}
    # A call_key carrying the structural punctuation the deterministic keyer emits.
    call_key = "sha:deadbeef#2"
    await _drive(
        registry,
        captured,
        call_key=call_key,
        tool_input={"command": "curl -X POST https://api/charge"},
    )
    execed = registry.exec_calls[0]["command"]
    # The sandbox path uses the shared per-call derivation — pin cross-path byte-identity
    # against the worker http_request delivery (a drift in either is a silent dedup bug).
    idem = idempotency_key("wfr_1", call_key)
    assert idem == hashlib.sha256(f"wfr_1\0{call_key}".encode()).hexdigest()  # wire-format pin
    expected_preamble = (
        f"export AIOS_RUN_ID={shlex.quote('wfr_1')} AIOS_IDEMPOTENCY_KEY={shlex.quote(idem)}\n"
    )
    assert execed == expected_preamble + "curl -X POST https://api/charge"
    # Distinct call_key → distinct idempotency key.
    registry2 = _FakeRegistry()
    captured2: dict[str, Any] = {}
    await _drive(registry2, captured2, call_key="sha:other#0", tool_input={"command": "echo x"})
    other_idem = idempotency_key("wfr_1", "sha:other#0")
    assert other_idem in registry2.exec_calls[0]["command"]
    assert idem not in registry2.exec_calls[0]["command"]


async def test_gate_not_callable_value() -> None:
    """A non-run-callable tool (read) at the executor returns the exact
    not-callable gate string as a value — exec never runs."""
    registry = _FakeRegistry()
    captured: dict[str, Any] = {}
    p1, p2, p3, p4 = _patches(registry, captured)
    with p1, p2, p3, p4:
        await run_sandbox._run_sandbox_task(
            _Pool(), _run(), call_key="k0", tool_name="read", tool_input={"command": "x"}
        )
    assert captured["result"] == {"error": "tool 'read' is not callable from a workflow run"}
    assert registry.exec_calls == []
    assert registry.provision_count == 0


async def test_gate_not_declared_value() -> None:
    """bash is run-callable but, when the run's frozen tools don't declare it, the
    executor returns the exact not-declared gate string as a value."""
    registry = _FakeRegistry()
    captured: dict[str, Any] = {}
    run: Any = SimpleNamespace(id="wfr_1", account_id="acc_t", tools=[], http_servers=[])
    p1, p2, p3, p4 = _patches(registry, captured)
    with p1, p2, p3, p4:
        await run_sandbox._run_sandbox_task(
            _Pool(), run, call_key="k0", tool_name="bash", tool_input={"command": "x"}
        )
    assert captured["result"] == {"error": "tool 'bash' is not in the workflow's declared tools"}
    assert registry.provision_count == 0


def test_launch_is_inflight_guarded() -> None:
    """launch_sandbox_task registers in the SHARED run_tools._INFLIGHT and is a
    no-op when a live task already holds the key (class-agnostic has_inflight)."""
    import asyncio

    async def _go() -> None:
        registry = _FakeRegistry()
        captured: dict[str, Any] = {}
        p1, p2, p3, p4 = _patches(registry, captured)
        with p1, p2, p3, p4:
            run_sandbox.launch_sandbox_task(
                _Pool(), _run(), call_key="k0", tool_name="bash", tool_input={"command": "x"}
            )
            assert ("wfr_1", "k0") in run_tools._INFLIGHT
            task = run_tools._INFLIGHT[("wfr_1", "k0")]
            # A second launch while live is a no-op — same task object.
            run_sandbox.launch_sandbox_task(
                _Pool(), _run(), call_key="k0", tool_name="bash", tool_input={"command": "x"}
            )
            assert run_tools._INFLIGHT[("wfr_1", "k0")] is task
            await task

    asyncio.run(_go())
