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
    spec_version: int = 0,
) -> SandboxHandle:
    """Construct a :class:`SandboxHandle` with sensible defaults for tests."""
    return SandboxHandle(
        session_id=session_id,
        sandbox_id=sandbox_id,
        workspace_path=workspace_path or Path("/tmp/aios-test-workspace"),
        mount_snapshot=mount_snapshot,
        spec_version=spec_version,
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
    # Sandbox ids the backend should report as dead via ``is_alive``.
    # Default-empty so existing tests behave as before; tests covering
    # the stale-handle path (#691) populate this.
    dead_sandbox_ids: set[str] = field(default_factory=set)

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        self.calls.append(("create", {"session_id": spec.session_id}))
        return SandboxHandle(
            session_id=spec.session_id,
            sandbox_id=self.next_handle_id,
            workspace_path=spec.workspace.host_path,
            mount_snapshot=spec.mount_snapshot,
            spec_version=spec.spec_version,
        )

    async def is_alive(self, handle: SandboxHandle) -> bool:
        self.calls.append(
            ("is_alive", {"session_id": handle.session_id, "sandbox_id": handle.sandbox_id})
        )
        return handle.sandbox_id not in self.dead_sandbox_ids

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


def patch_build_spec_deps(
    *,
    env_config: Any = None,
    docker_image: str = "ghcr.io/eumemic/aios-sandbox:latest",
    sandbox_disk_bytes: int | None = None,
    env_var_credentials: Any = None,
    github_clones: Any = None,
    tool_broker: Any = None,
) -> tuple[Any, ...]:
    """Context-manager bundle stubbing every external dependency of
    ``build_spec_from_session`` so it runs to the ``_assemble_plan`` call
    with synthetic settings.

    The three keyword overrides let a test install its OWN mock for a
    materializer or the tool broker — each target is patched exactly
    once, so the mock a test asserts on is unambiguously the installed
    one (no nested re-patching).
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    settings = MagicMock()
    settings.docker_image = docker_image
    settings.sandbox_disk_bytes = sandbox_disk_bytes
    settings.instance_id = "inst_TEST"
    settings.sandbox_cpu_quota = None
    settings.sandbox_memory_bytes = None
    settings.sandbox_pids_limit = None
    settings.sandbox_seccomp_profile = "/app/docker/seccomp-sandbox.json"
    settings.tool_broker_socket_path = None

    if tool_broker is None:
        tool_broker = MagicMock()
        tool_broker.port = 54321
        tool_broker.register_session = MagicMock()
        tool_broker.unregister_session = MagicMock()

    return (
        patch("aios.sandbox.spec.get_settings", return_value=settings),
        patch(
            "aios.sandbox.spec.sessions_service.load_session_account_id",
            AsyncMock(return_value="acct_x"),
        ),
        patch(
            "aios.sandbox.spec._load_session_provisioning",
            # (workspace_path, env, spec_version) since #713.
            AsyncMock(return_value=("/tmp/w", {}, 0)),
        ),
        # ``build_spec_from_session`` imports these function-locally from
        # ``aios.sandbox.volumes`` (deferred import to avoid a cycle), so
        # patch them at the source module, not on ``aios.sandbox.spec``.
        patch("aios.sandbox.volumes.validate_workspace_path", MagicMock()),
        patch(
            "aios.sandbox.volumes.ensure_workspace_path",
            MagicMock(return_value=Path("/tmp/w")),
        ),
        patch(
            "aios.sandbox.spec._load_environment_config",
            AsyncMock(return_value=env_config),
        ),
        patch(
            "aios.sandbox.spec._materialize_memory_mounts",
            AsyncMock(return_value=[]),
        ),
        patch(
            "aios.sandbox.spec._materialize_env_var_credentials",
            env_var_credentials or AsyncMock(return_value=()),
        ),
        patch(
            "aios.sandbox.spec._materialize_github_clones",
            github_clones or AsyncMock(return_value=([], None)),
        ),
        patch("aios.sandbox.spec.runtime.require_pool", MagicMock()),
        patch(
            "aios.sandbox.spec.runtime.require_tool_broker",
            MagicMock(return_value=tool_broker),
        ),
        patch(
            "aios.sandbox.volumes.ensure_session_attachments_dir",
            return_value=Path("/tmp/a"),
        ),
        patch(
            "aios.sandbox.volumes.ensure_session_uploads_dir",
            return_value=Path("/tmp/u"),
        ),
    )
