"""Test doubles for the sandbox-backend surface.

Use ``FakeBackend`` when a test needs to drive the real
:class:`SandboxRegistry` without hitting Docker. It records every
backend call so tests can assert on the verb sequence, and returns
canned :class:`SandboxHandle` and :class:`CommandResult` values.

Tests that don't exercise registry internals (most tool-handler
tests) keep their lightweight ``_StubRegistry`` pattern; they just
need to expose a ``get_or_provision`` and an ``exec`` method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aios.sandbox.backends.base import (
    CommandResult,
    ManagedSandboxRef,
    SandboxBackend,
    SandboxHandle,
    SandboxSpec,
)


def make_handle(
    *,
    session_id: str = "sess_01TEST",
    sandbox_id: str = "abc123def456abc123def456",
    workspace_path: Path | None = None,
    mount_snapshot: frozenset[tuple[str, ...]] = frozenset(),
) -> SandboxHandle:
    """Construct a :class:`SandboxHandle` with sensible defaults for tests."""
    return SandboxHandle(
        session_id=session_id,
        sandbox_id=sandbox_id,
        workspace_path=workspace_path or Path("/tmp/aios-test-workspace"),
        mount_snapshot=mount_snapshot,
    )


@dataclass
class FakeBackend:
    """In-memory :class:`SandboxBackend` for unit tests.

    Records every method call on ``self.calls`` (a list of ``(verb,
    kwargs)`` tuples) so tests can assert verb order. Returns a
    canned :class:`SandboxHandle` from ``create``; ``exec`` returns
    a configurable :class:`CommandResult` (defaults to a happy
    exit-zero result) — set ``self.next_result`` to override.

    The ``destroy``, ``force_remove``, and ``list_managed`` methods
    are no-ops by default; tests that exercise the orphan reaper
    can populate ``self.managed`` with the refs to return.
    """

    name: str = "fake"
    next_handle_id: str = "fake_sandbox_id"
    next_result: CommandResult | None = None
    managed: list[ManagedSandboxRef] = field(default_factory=list)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        self.calls.append(("create", {"session_id": spec.session_id}))
        return SandboxHandle(
            session_id=spec.session_id,
            sandbox_id=self.next_handle_id,
            workspace_path=spec.workspace.host_path,
            mount_snapshot=spec.mount_snapshot,
        )

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        timeout_seconds: int,
        max_output_bytes: int,
        cwd: str = "/workspace",
    ) -> CommandResult:
        self.calls.append(
            (
                "exec",
                {
                    "session_id": handle.session_id,
                    "sandbox_id": handle.sandbox_id,
                    "command": command,
                    "timeout_seconds": timeout_seconds,
                    "max_output_bytes": max_output_bytes,
                    "cwd": cwd,
                },
            )
        )
        if self.next_result is not None:
            return self.next_result
        return CommandResult(
            exit_code=0,
            stdout="",
            stderr="",
            timed_out=False,
            truncated=False,
        )

    async def destroy(self, handle: SandboxHandle) -> None:
        self.calls.append(
            ("destroy", {"session_id": handle.session_id, "sandbox_id": handle.sandbox_id})
        )

    async def list_managed(self, *, instance_id: str) -> list[ManagedSandboxRef]:
        self.calls.append(("list_managed", {"instance_id": instance_id}))
        return list(self.managed)

    async def force_remove(self, sandbox_id: str) -> None:
        self.calls.append(("force_remove", {"sandbox_id": sandbox_id}))


# Static check that FakeBackend satisfies the Protocol.
def _assert_protocol(backend: SandboxBackend) -> None:
    pass


_assert_protocol(FakeBackend())
