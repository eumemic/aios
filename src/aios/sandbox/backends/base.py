"""The abstract sandbox-backend surface.

Everything in :mod:`aios.sandbox.registry` and :mod:`aios.sandbox.spec`
talks to backends through the :class:`SandboxBackend` ``Protocol`` — the
registry doesn't know whether it has Docker, a host subprocess, a
microVM, or a remote executor. The five methods on the Protocol are
the only verbs the rest of the system needs:

- ``create(spec)`` — provision a new sandbox; return a handle to it.
- ``exec(handle, command, ...)`` — run a shell command inside it.
- ``destroy(handle)`` — tear it down.
- ``list_managed(instance_id=...)`` — for orphan reaping at worker startup.
- ``force_remove(sandbox_id)`` — for orphan reaping when we only have an id.

The data types here are deliberately backend-agnostic. ``SandboxSpec``
expresses *what* the sandbox should be (workspace, mounts, env, network
policy) in semantic terms; each backend translates to its own primitives.
A Docker backend turns ``Limited`` into ``--cap-add NET_ADMIN`` plus an
iptables script; a host-subprocess backend would either implement that
via host firewall rules or warn-and-noop.

``SandboxHandle`` is a frozen dataclass — no methods, no behavior. All
command execution flows through ``backend.exec(handle, ...)`` so the handle
stays trivially serializable and can't accumulate hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Mount:
    """A bind mount from a host path into a sandbox path.

    ``host_path`` must be absolute when handed to a Docker backend (the
    daemon rejects relative paths). Other backends may interpret host
    paths as plain filesystem references.
    """

    host_path: Path
    sandbox_path: str
    read_only: bool = False


class NetworkPolicy:
    """Tagged-union base for network policies. See subclasses."""


@dataclass(frozen=True, slots=True)
class Unrestricted(NetworkPolicy):
    """No network restrictions; sandbox can reach anything the host can."""


@dataclass(frozen=True, slots=True)
class Limited(NetworkPolicy):
    """Allow outbound only to ``allowed_hosts`` (resolved at apply time).

    The Docker backend interprets this as ``--cap-add NET_ADMIN`` on
    create, with the actual iptables script applied separately via
    :func:`aios.sandbox.setup.apply_network_lockdown` after create
    returns. The script-application path is shared logic, not a backend
    concern, so backends that can't enforce it surface failures the same
    way as backends that can.
    """

    allowed_hosts: frozenset[str]


@dataclass(frozen=True, slots=True)
class SandboxSpec:
    """Everything a backend needs to provision a sandbox for a session.

    Built by :func:`aios.sandbox.spec.build_spec_from_session` from the
    session's DB row, attached resources, environment config, and
    workspace directory. Pure data — no live handles to host-side
    services, no DB pool. Anything that needs to outlive the spec (the
    GitProxy, materialized memory-store snapshots) is owned elsewhere.

    ``mount_snapshot`` is a derived value from the same echoes that
    populated ``extra_mounts``; the backend stamps it onto the handle
    it returns so the registry's drift detector can compare it against
    the current echo set on each step.

    ``spec_version`` is the ``sessions.spec_version`` snapshot at build
    time (issue #713). The backend copies it onto the handle so the
    registry can re-read the current version on a warm hit and recycle
    the sandbox when a session-scoped resource changed between steps.
    """

    session_id: str
    instance_id: str
    workspace: Mount
    extra_mounts: tuple[Mount, ...]
    environment: dict[str, str]
    labels: dict[str, str]
    network_policy: NetworkPolicy
    host_gateway_alias: str | None
    image: str
    mount_snapshot: frozenset[tuple[str, ...]] = frozenset()
    # ``sessions.spec_version`` snapshot at build time (issue #713). Bumped
    # by Postgres triggers on the resource tables that feed this spec
    # (``session_memory_stores`` / ``session_github_repositories``); the
    # registry's warm-hit probe compares it against the current value.
    spec_version: int = 0
    # Per-sandbox resource caps (multi-tenancy hardening — #367 PR 9).
    # ``None`` leaves the host's default in place. The spec builder
    # populates these from the AIOS_SANDBOX_* settings; the backend
    # injects the corresponding ``docker run`` flags.
    cpu_quota: float | None = None
    memory_bytes: int | None = None
    pids_limit: int | None = None
    # Writable-layer disk cap (issue #725). ``None`` leaves the host's
    # default (unbounded) in place. The spec builder resolves this from
    # the environment's ``disk_bytes`` override, falling back to the
    # global ``settings.sandbox_disk_bytes``; the backend injects
    # ``--storage-opt size=`` when set.
    disk_bytes: int | None = None
    # Authored seccomp profile (#807). Always set by the spec builder to the
    # configured path (a host filesystem path the docker CLI reads), or the
    # literal "unconfined" for emergency rollback. The backend ALWAYS emits
    # --security-opt seccomp=<value>; this field is never None so the default
    # profile is never silently shipped. The default below is a fallback for
    # bare test construction only; production always sets it from settings.
    seccomp_profile: str = "unconfined"


@dataclass(frozen=True, slots=True)
class SandboxHandle:
    """Opaque handle to a running sandbox. Backend-agnostic.

    ``sandbox_id`` is a backend-internal identifier — a Docker container
    id today, a fabricated uuid for a host-subprocess backend tomorrow.
    The registry treats it as opaque; only the backend that issued it
    knows how to interpret it.

    ``mount_snapshot`` lets the registry detect when a session's mounts
    have changed since the sandbox was provisioned (e.g. a memory store
    was attached or detached) — see
    :meth:`SandboxRegistry.release_if_mounts_changed`.

    ``spec_version`` is the provision-time snapshot of
    ``sessions.spec_version`` (issue #713). The registry's staleness
    probe re-reads the current version on a warm hit and recycles the
    cached sandbox when it has drifted past this snapshot — catching the
    API-process and direct-SQL mutation gaps the write-path eviction
    can't see.
    """

    session_id: str
    sandbox_id: str
    workspace_path: Path
    mount_snapshot: frozenset[tuple[str, ...]] = frozenset()
    spec_version: int = 0


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Result of a single command run inside a sandbox."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    truncated: bool


@dataclass(frozen=True, slots=True)
class ManagedSandboxRef:
    """A reference to a sandbox the orphan reaper wants to inspect.

    The reaper compares each ``session_id`` against the worker's set of
    active session ids; sandboxes whose session is no longer active are
    removed via ``force_remove``. ``session_id`` is ``None`` when the
    backend can't recover it (e.g. a Docker container missing the
    ``aios.session_id`` label, which shouldn't happen but the reaper is
    defensive).
    """

    sandbox_id: str
    session_id: str | None


class SandboxBackendError(Exception):
    """Raised when the underlying execution layer fails to do its job
    (Docker daemon unreachable, image missing, host-process spawn
    failure, etc.).

    Distinct from a command that runs and returns nonzero — that's a
    successful :class:`CommandResult` with a nonzero ``exit_code``. Only
    the *infrastructure* failing raises this.
    """


@runtime_checkable
class SandboxBackend(Protocol):
    """The five-verb surface every backend implements."""

    name: str

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Provision a sandbox per ``spec`` and return its handle.

        Raises :class:`SandboxBackendError` if the backend cannot
        produce a sandbox (e.g. Docker daemon down, image missing).
        On success the sandbox is alive and ready to receive exec
        calls — backends are responsible for any post-create blocking
        until that is true.
        """
        ...

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        timeout_seconds: int,
        max_output_bytes: int,
        cwd: str = "/workspace",
    ) -> CommandResult:
        """Run ``command`` (interpreted by ``bash -c``) inside ``handle``.

        Output beyond ``max_output_bytes`` is truncated; the result's
        ``truncated`` flag indicates whether truncation happened. The
        command is killed if it runs longer than ``timeout_seconds``;
        ``timed_out`` reports that. A nonzero exit is *not* an error —
        it's a successful :class:`CommandResult` whose ``exit_code`` the
        caller interprets.

        Raises :class:`SandboxBackendError` if the underlying execution
        infrastructure fails (sandbox gone, daemon hiccup, host process
        spawn failure).
        """
        ...

    async def destroy(self, handle: SandboxHandle) -> None:
        """Tear down ``handle``'s sandbox.

        Idempotent: a sandbox that's already gone is treated as a no-op
        (logged, not raised). The host-side workspace directory is NOT
        deleted — workspaces persist across sandbox lifetimes.
        """
        ...

    async def list_managed(self, *, instance_id: str) -> list[ManagedSandboxRef]:
        """List sandboxes belonging to this aios instance.

        Used by the worker's orphan reaper at startup. ``instance_id``
        scopes the list to this deployment so a reaper can't kill
        sandboxes belonging to a concurrent worker on the same machine.

        Raises :class:`SandboxBackendError` if the backend cannot
        enumerate (e.g. Docker daemon unreachable).
        """
        ...

    async def force_remove(self, sandbox_id: str) -> None:
        """Force-remove a sandbox by id.

        Used by the orphan reaper when only the id is known. Logs but
        does not raise on failure — a sandbox already gone is fine.
        """
        ...

    async def is_alive(self, handle: SandboxHandle) -> bool:
        """Return True iff ``handle``'s sandbox is still running.

        Used by the registry's warm path (issue #691) to detect
        sandboxes that the underlying execution layer removed without
        the registry's knowledge. For Docker that is the ``--rm`` path
        after any entrypoint exit (OOM kill, internal crash, signal) —
        the container is gone but the registry's cached handle survives
        until the next command-run fails with "No such container".

        Implementations should be cheap (single round-trip) and total:
        a transient probe failure (daemon hiccup, timeout) should
        return False so the caller re-provisions rather than risk
        returning a possibly-dead handle.
        """
        ...


# ── Standard labels every backend SHOULD set on managed sandboxes ───────────
#
# The orphan reaper uses ``MANAGED_LABEL_KEY=MANAGED_LABEL_VALUE`` plus
# ``INSTANCE_LABEL_KEY=<this instance id>`` to find the worker's own
# sandboxes (and only those — never a sibling worker's). ``SESSION_LABEL_KEY``
# carries the aios session id so the reaper can compare against the worker's
# active session set.
#
# Backends that don't natively support labels (e.g. a host-subprocess
# backend) may carry these in their own bookkeeping instead — the
# convention is shared, the implementation is per-backend.
MANAGED_LABEL_KEY = "aios.managed"
MANAGED_LABEL_VALUE = "true"
INSTANCE_LABEL_KEY = "aios.instance_id"
SESSION_LABEL_KEY = "aios.session_id"
