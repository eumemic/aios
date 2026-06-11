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
    ManagedImage,
    ManagedSandboxRef,
    SandboxBackend,
    SandboxBackendError,
    SandboxHandle,
    SandboxSpec,
    SnapshotOutcome,
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
    # ── durable-session-sandbox surface (drives registry lifecycle tests
    #    without Docker) ──────────────────────────────────────────────────
    # The outcome ``snapshot`` returns (default: a committed layer). Set to
    # exercise skip/flatten paths; set ``snapshot_raises`` to simulate an
    # infra failure (corpse retained, no rm).
    next_snapshot_outcome: SnapshotOutcome | None = None
    snapshot_raises: bool = False
    managed_images: list[ManagedImage] = field(default_factory=list)
    # In-memory image table the store seam reads through (image ref → labels);
    # ``None`` value models "absent" for verified-negative ``image_labels``.
    image_labels_by_ref: dict[str, dict[str, str] | None] = field(default_factory=dict)
    image_sizes_by_ref: dict[str, int] = field(default_factory=dict)
    # Refs ``remove_image`` should refuse (return False) rather than remove.
    refuse_remove_refs: set[str] = field(default_factory=set)
    removed_image_refs: list[str] = field(default_factory=list)
    # Results ``run_netns_sidecar`` returns, popped in order (apply, verify);
    # default exit-0 when drained. Set to exercise lockdown apply/verify paths.
    sidecar_results: list[CommandResult] = field(default_factory=list)

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        self.calls.append(("create", {"session_id": spec.session_id}))
        return SandboxHandle(
            session_id=spec.session_id,
            sandbox_id=self.next_handle_id,
            workspace_path=spec.workspace.host_path,
            mount_snapshot=spec.mount_snapshot,
            spec_version=spec.spec_version,
            snapshot_image=spec.snapshot_image,
            disk_limit_bytes=spec.snapshot_budget_bytes,
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

    async def list_managed(
        self, *, instance_id: str, session_id: str | None = None
    ) -> list[ManagedSandboxRef]:
        self.calls.append(("list_managed", {"instance_id": instance_id, "session_id": session_id}))
        if session_id is None:
            return list(self.managed)
        return [ref for ref in self.managed if ref.session_id == session_id]

    async def force_remove(self, sandbox_id: str) -> None:
        self.calls.append(("force_remove", {"sandbox_id": sandbox_id}))

    async def stop(self, sandbox_id: str) -> None:
        self.calls.append(("stop", {"sandbox_id": sandbox_id}))

    async def snapshot(
        self,
        sandbox_id: str,
        tag: str,
        *,
        empty_floor_bytes: int,
        flatten_if_unique_bytes_over: int | None,
    ) -> SnapshotOutcome:
        self.calls.append(
            (
                "snapshot",
                {
                    "sandbox_id": sandbox_id,
                    "tag": tag,
                    "empty_floor_bytes": empty_floor_bytes,
                    "flatten_if_unique_bytes_over": flatten_if_unique_bytes_over,
                },
            )
        )
        if self.snapshot_raises:
            raise SandboxBackendError("fake snapshot failure")
        if self.next_snapshot_outcome is not None:
            return self.next_snapshot_outcome
        return SnapshotOutcome(
            kind="committed", image_id=f"sha256:{tag}", unique_bytes=1024, depth=1
        )

    async def list_managed_images(self, *, instance_id: str) -> list[ManagedImage]:
        self.calls.append(("list_managed_images", {"instance_id": instance_id}))
        return list(self.managed_images)

    async def remove_image(self, ref: str) -> bool:
        self.calls.append(("remove_image", {"ref": ref}))
        if ref in self.refuse_remove_refs:
            return False
        self.removed_image_refs.append(ref)
        self.image_labels_by_ref.pop(ref, None)
        self.image_sizes_by_ref.pop(ref, None)
        return True

    async def image_size(self, image: str) -> int:
        self.calls.append(("image_size", {"image": image}))
        if image not in self.image_sizes_by_ref:
            raise SandboxBackendError(f"fake image not found: {image}")
        return self.image_sizes_by_ref[image]

    async def image_labels(self, image: str) -> dict[str, str] | None:
        self.calls.append(("image_labels", {"image": image}))
        # Absent key OR explicit None value ⇒ verified-not-found.
        return self.image_labels_by_ref.get(image)

    async def run_netns_sidecar(
        self,
        target_sandbox_id: str,
        *,
        image: str,
        script: str,
        timeout_seconds: int,
        max_output_bytes: int,
    ) -> CommandResult:
        self.calls.append(
            (
                "run_netns_sidecar",
                {
                    "target_sandbox_id": target_sandbox_id,
                    "image": image,
                    "script": script,
                    "timeout_seconds": timeout_seconds,
                },
            )
        )
        # Each call pops the next queued result (apply, then verify); default
        # exit-0 once the queue is drained.
        if self.sidecar_results:
            return self.sidecar_results.pop(0)
        return CommandResult(exit_code=0, stdout="", stderr="", timed_out=False, truncated=False)


async def purge_all_sandboxes(registry: Any) -> None:
    """Hard test-teardown for a real-Docker registry (durable session sandboxes).

    ``stop_all`` only STOPS containers (their filesystems are meant to survive
    for the next worker's GC tick) and never touches snapshot images, so a test
    that truncates the sessions table would leak stopped corpses + snapshot
    images across runs. This removes both for the test instance so each test
    starts from a clean daemon. Refused image removals (a child still
    references the layer) are ignored — the leaf's removal cascades.
    """
    from aios.config import get_settings

    await registry.stop_all()
    backend = registry._backend
    instance_id = get_settings().instance_id
    for ref in await backend.list_managed(instance_id=instance_id):
        await backend.force_remove(ref.sandbox_id)
    for img in await backend.list_managed_images(instance_id=instance_id):
        removal_ref = img.repo_tags[0] if img.repo_tags else img.image_id
        await backend.remove_image(removal_ref)


class _FakeConn:
    """Minimal asyncpg-conn stand-in for the snapshot-pointer queries the
    registry runs (``execute``/``fetchrow``/``fetchval``). Reads return benign
    defaults; ``fetchval`` returns a stub account id for ``load_session_account_id``."""

    async def execute(self, *args: Any, **kwargs: Any) -> str:
        return "OK"

    async def fetchrow(self, *args: Any, **kwargs: Any) -> Any:
        return None

    async def fetchval(self, *args: Any, **kwargs: Any) -> Any:
        return "acct_test"

    async def fetch(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


class _FakeAcquire:
    async def __aenter__(self) -> _FakeConn:
        return _FakeConn()

    async def __aexit__(self, *args: Any) -> bool:
        return False


class FakePool:
    """A fake asyncpg pool whose ``acquire()`` yields a no-op :class:`_FakeConn`.

    Lets registry snapshot/pointer code that calls ``runtime.require_pool()``
    run under unit tests without a real database (the pointer writes are
    no-op ``execute``s). Set ``runtime.pool`` to an instance and restore it.
    """

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire()


# Static check that FakeBackend satisfies the Protocol.
def _assert_protocol(backend: SandboxBackend) -> None:
    pass


_assert_protocol(FakeBackend())


def patch_build_spec_deps(
    *,
    env_config: Any = None,
    session_env: dict[str, str] | None = None,
    docker_image: str = "ghcr.io/eumemic/aios-sandbox:latest",
    sandbox_snapshot_budget_bytes: int | None = None,
    env_var_credentials: Any = None,
    github_clones: Any = None,
    tool_broker: Any = None,
) -> tuple[Any, ...]:
    """Context-manager bundle stubbing every external dependency of
    ``build_spec_from_session`` so it runs to the ``_assemble_plan`` call
    with synthetic settings.

    The keyword overrides let a test install its OWN mock for a
    materializer, the tool broker, or the per-session env — each target
    is patched exactly once, so the mock a test asserts on is
    unambiguously the installed one (no nested re-patching).
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    settings = MagicMock()
    settings.docker_image = docker_image
    settings.sandbox_snapshot_budget_bytes = sandbox_snapshot_budget_bytes
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
            # (workspace_path, session_env, spec_version, snapshot_ref).
            AsyncMock(return_value=("/tmp/w", session_env or {}, 0, None)),
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
