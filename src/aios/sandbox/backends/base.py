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
from typing import Literal, Protocol, runtime_checkable


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

    The sandbox itself is granted NO ``NET_ADMIN`` (durable session
    sandboxes, §5.8). The iptables lockdown is applied + read-back verified
    separately via :func:`aios.sandbox.setup.apply_network_lockdown` after
    create returns, from an ephemeral operator-image sidecar that joins the
    sandbox's netns — so a tenant can neither poison the binaries that apply
    the lockdown nor flush it from inside. The application path is shared
    logic, not a backend concern, so backends that can't enforce it surface
    failures the same way as backends that can.
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
    # Resume source (durable session sandboxes): the locally-resolved
    # snapshot tag a resuming session runs from, set by the registry's
    # provision path after it resolves ``sessions.snapshot_ref`` through
    # the store (verified-negative, base-drift checked). ``None`` ⇒ cold
    # start from ``image``. The backend runs from ``snapshot_image or
    # image``; everything else (env, mounts, lockdown) is re-derived fresh
    # so resume is structurally a cold start that happens to run on a
    # persisted rootfs.
    snapshot_image: str | None = None
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
    # Run the container under ``docker --init`` (issue #1421). When True the
    # backend injects ``--init`` so Docker runs the bundled ``docker-init``
    # (tini) as PID 1 and execs the keepalive CMD (``tail -f /dev/null``) as
    # its child. tini reaps orphaned zombies automatically, so PID-cgroup
    # slots free as soon as a process dies — even after an OOM-kill orphans a
    # process tree — preventing the zombie pile-up that exhausts
    # ``--pids-limit`` and wedges the sandbox with ``fork() → EAGAIN``.
    # Defaults to True: every session sandbox should run with a reaping PID 1.
    init: bool = True
    # Per-session snapshot budget in unique bytes (durable session
    # sandboxes, §5.7). Drives the release-time flatten trigger: when a
    # session's unique snapshot bytes would exceed this, the snapshot verb
    # flattens (collapse + whiteout) instead of committing another layer.
    # The backend stamps it onto the handle as ``disk_limit_bytes`` so the
    # release path (which only holds the handle) can pass it back. ``None``
    # ⇒ unbounded (never flatten on budget; the depth backstop still
    # applies). Replaces the dead ``--storage-opt size=`` writable-layer
    # cap, which never worked on prod ext4.
    snapshot_budget_bytes: int | None = None
    # Authored seccomp profile (#807). Always set by the spec builder to the
    # configured path (a host filesystem path the docker CLI reads), or the
    # literal "unconfined" for emergency rollback. The backend ALWAYS emits
    # --security-opt seccomp=<value>; this field is never None so the default
    # profile is never silently shipped. The default below is a fallback for
    # bare test construction only; production always sets it from settings.
    seccomp_profile: str = "unconfined"
    # Optional backend-specific container runtime (#1014). ``None`` leaves Docker's
    # default runtime in place (local/CI no-op); DockerBackend translates a value
    # such as ``runsc`` into ``docker run --runtime runsc``.
    runtime: str | None = None


@dataclass(frozen=True, slots=True)
class SandboxHandle:
    """Opaque handle to a running sandbox. Backend-agnostic.

    ``owner_id`` is the opaque label of whatever the sandbox belongs to — a
    session ULID (``sess_…``) for a session sandbox, a workflow-run ULID
    (``wfr_…``) for a run sandbox (#988). The registry keys its handle map on
    it; the backend copies it from ``SandboxSpec.session_id`` (which stays the
    opaque owner-label field) at create time. Logs/error messages stamp it so a
    failure is attributable regardless of which owner kind it is.

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

    owner_id: str
    sandbox_id: str
    workspace_path: Path
    mount_snapshot: frozenset[tuple[str, ...]] = frozenset()
    spec_version: int = 0
    # The locally-resolved snapshot tag this container was created from
    # (durable session sandboxes), or ``None`` for a cold start from base.
    # Stamped from ``spec.snapshot_image`` so the provision span can record
    # whether a resume ran from a snapshot or cold-started.
    snapshot_image: str | None = None
    # Per-session snapshot budget in unique bytes (§5.7), copied from
    # ``spec.snapshot_budget_bytes``. The release path passes it back to
    # ``backend.snapshot`` as the flatten trigger. ``None`` ⇒ unbounded.
    disk_limit_bytes: int | None = None


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
    """A reference to a managed sandbox container the GC/salvage paths inspect.

    The GC corpse pass compares each ``session_id`` against the worker's
    set of active session ids; a corpse whose session is dormant/deleted
    is salvaged then removed. ``session_id`` is ``None`` when the backend
    can't recover it (a container missing the ``aios.session_id`` label,
    which shouldn't happen but the sweep is defensive).

    ``running`` distinguishes a live container from a stopped corpse —
    the salvage path only ``stop``s the former, and a still-running
    container for a session this worker isn't stepping is a crash
    survivor that the GC tick stops + salvages. Populated from ``docker
    ps --all`` (the listing now includes stopped containers — under
    persistence a corpse is salvageable, not just removable).
    """

    sandbox_id: str
    session_id: str | None
    running: bool = True


@dataclass(frozen=True, slots=True)
class ManagedImage:
    """A managed snapshot image (or untagged residue) the GC image pass classifies.

    Enumerated by :meth:`SandboxBackend.list_managed_images` via ``docker
    images -a`` so untagged crash/flatten residue is visible (it is
    invisible without ``-a`` — verified). ``parent_id`` is the image's
    ``.Parent`` (``None`` when absent); the GC excludes any image that is
    the parent of another listed image so it never removes the untagged
    interior of a live chain. On the containerd image store ``.Parent``
    can be empty even for genuine chains — the structural skip then
    no-ops and the ``remove_image`` refusal (a child still references the
    layer) is the safety net, which is correct, just noisier.

    ``repo_tags`` is empty for untagged residue. ``labels`` carries the
    ``aios.*`` metadata (``session_id``, ``base_image``, ``flattened``)
    the classifier keys on without any DB lookup.
    """

    image_id: str
    repo_tags: tuple[str, ...]
    parent_id: str | None
    size_bytes: int
    labels: dict[str, str]


# Outcome of a single snapshot verb (``SandboxBackend.snapshot``).
SnapshotKind = Literal["committed", "flattened", "skipped_empty", "skipped_stale"]


@dataclass(frozen=True, slots=True)
class SnapshotOutcome:
    """What a :meth:`SandboxBackend.snapshot` call did to the canonical tag.

    - ``committed`` — a new layer was committed onto the chain.
    - ``flattened`` — the chain was collapsed to a single standalone layer
      (``export | import``), stripping baked config (the definitive secret
      scrub) and re-rooting off the base.
    - ``skipped_empty`` — the writable layer was at/below the empty floor
      (identity short-circuit), so no new image; the prior tag, if any,
      stays canonical. ``image_id`` is ``None`` only when no prior tag
      existed (a chat/read-only session that never wrote).
    - ``skipped_stale`` — the corpse's parent is a *child* of the existing
      tag (content-equal crash residue under the salvage-before-provision
      invariant); the corpse is discarded without a commit and the
      existing newer tag stays canonical.

    ``image_id`` is the winning canonical image id after the op (``None``
    iff no snapshot exists). ``unique_bytes`` is the accounting figure
    written to ``sessions.snapshot_bytes`` (tag size minus base size, or
    full size for a flattened/standalone image). ``depth`` is the chain
    depth of the canonical image after the op (0 when ``image_id`` is
    ``None``) — surfaced for the provision span and the flatten tests.
    """

    kind: SnapshotKind
    image_id: str | None
    unique_bytes: int
    depth: int


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

    async def list_managed(
        self, *, instance_id: str, session_id: str | None = None
    ) -> list[ManagedSandboxRef]:
        """List managed sandbox containers belonging to this aios instance.

        Includes **stopped** containers (``docker ps --all``): under
        durable persistence a crashed corpse is salvageable, not merely
        removable, so the GC corpse pass and the provision-preamble
        salvage must see stopped containers too. Each ref's ``running``
        flag distinguishes a live container from a corpse.

        ``instance_id`` scopes the list to this deployment so a sweep
        can't touch a concurrent worker's containers. ``session_id``, when
        given, narrows to one session's containers (the salvage preamble
        targets a single session).

        Raises :class:`SandboxBackendError` if the backend cannot
        enumerate (e.g. Docker daemon unreachable).
        """
        ...

    async def snapshot(
        self,
        sandbox_id: str,
        tag: str,
        *,
        empty_floor_bytes: int,
        flatten_if_unique_bytes_over: int | None,
    ) -> SnapshotOutcome:
        """Commit (or flatten) a container's writable rootfs to ``tag``.

        The planned-teardown verb of durable session sandboxes (§5.2):
        ``stop -t 5`` → inspect → **lineage gate** (proceed iff ``tag``
        absent or ``tag.Id == corpse.Image``; else ``skipped_stale``) →
        ``SizeRw <= empty_floor_bytes`` short-circuit (``skipped_empty``)
        → commit-or-flatten. The container is **not** removed — the caller
        removes it only after this returns successfully.

        Flatten (``export | import``, collapsing the chain to one
        standalone layer and stripping baked config — the definitive
        secret scrub) fires when the chain depth would cross the backend's
        soft ceiling, or when ``flatten_if_unique_bytes_over`` is not
        ``None`` and the projected unique bytes exceed it. Commit scrubs
        exactly the ``aios.env_keys`` label's named vars from the image
        config (``ENV K=``); ``base_image``/``env_keys`` are read from the
        container's labels so a salvaged crash corpse needs no spec.

        Raises :class:`SandboxBackendError` on infra failure (daemon down,
        commit/export timeout) — the caller then retains the corpse and
        converges via salvage; it never removes a container whose snapshot
        verb failed.
        """
        ...

    async def stop(self, sandbox_id: str) -> None:
        """``docker stop -t 5`` a container. Idempotent on a stopped corpse.

        Logs but does not raise on a missing container (already gone).
        Raises :class:`SandboxBackendError` on a hung/unreachable daemon.
        """
        ...

    async def list_managed_images(self, *, instance_id: str) -> list[ManagedImage]:
        """Enumerate this instance's managed snapshot images via ``docker images -a``.

        The ``-a`` is load-bearing: untagged crash/flatten residue is
        invisible without it. Each :class:`ManagedImage` carries the
        ``.Parent`` (for the GC's structural skip of live-chain interiors),
        size, repo tags, and ``aios.*`` labels.

        Raises :class:`SandboxBackendError` if the backend cannot enumerate.
        """
        ...

    async def remove_image(self, ref: str) -> bool:
        """``docker rmi`` (no ``--force``) ``ref``. Returns success.

        ``True`` when the image was removed or was already gone
        (idempotent goal-state). ``False`` when the daemon **refused** —
        a container or child image still references it; the GC treats a
        refusal as "retained this tick, retry next tick" rather than
        forcing. Removing a tag cascade-deletes its untagged parent chain
        down to the first still-referenced image.

        Raises :class:`SandboxBackendError` only on infra failure
        (daemon unreachable / timeout).
        """
        ...

    async def image_size(self, image: str) -> int:
        """Return ``image``'s ``.Size`` in bytes.

        Raises :class:`SandboxBackendError` if the image is absent or the
        daemon is unreachable (callers that tolerate absence catch it).
        """
        ...

    async def image_labels(self, image: str) -> dict[str, str] | None:
        """Return ``image``'s ``.Config.Labels``, or ``None`` if absent.

        Verified-negative: a confirmed not-found returns ``None``; an
        indeterminate probe (daemon hiccup, timeout) raises
        :class:`SandboxBackendError` so the caller fails loud rather than
        treating a hiccup as "no labels". Used by the resume path's
        base-image drift check.
        """
        ...

    async def run_netns_sidecar(
        self,
        target_sandbox_id: str,
        *,
        image: str,
        script: str,
        timeout_seconds: int,
        max_output_bytes: int,
        runtime: str | None = None,
    ) -> CommandResult:
        """Run ``script`` in an ephemeral sidecar joined to a sandbox's netns.

        The network-lockdown apply/verify path (§5.8). The sidecar runs the
        **operator-trusted** ``image`` (never the tenant ``env_config.image``)
        with ``NET_ADMIN``, joins ``target_sandbox_id``'s network namespace so
        its iptables edits apply to the sandbox's traffic, and is removed on
        exit. The sandbox itself holds no ``NET_ADMIN``, so root-in-sandbox
        can neither flush its own lockdown nor poison the binaries that apply
        it. ``runtime`` (#1014) selects the backend-specific container runtime
        for the sidecar (e.g. ``runsc``); ``None`` leaves the backend default.
        Callers pass it explicitly — pinned to the target sandbox's spec —
        because the backend layer never reads ambient config. Returns the
        script's :class:`CommandResult`; raises
        :class:`SandboxBackendError` on infra failure / timeout.
        """
        ...

    async def force_remove(self, sandbox_id: str) -> None:
        """Force-remove a sandbox by id.

        Used by the orphan reaper when only the id is known. Logs but
        does not raise on failure — a sandbox already gone is fine.
        """
        ...

    async def prewarm_run(self, image: str) -> str:
        """``docker run --detach`` ``image`` as a transient prewarm container.

        Operator prewarm bake (#1348). A PLAIN run — no managed/instance/
        session labels, no spec — so the committed prewarm image never enters
        the image GC's reapable set. Returns the new container's id. Raises
        :class:`SandboxBackendError` on launch failure.
        """
        ...

    async def prewarm_commit(self, sandbox_id: str, tag: str, *, labels: dict[str, str]) -> None:
        """``docker commit`` ``sandbox_id`` to ``tag``, stamping ``labels``.

        Operator prewarm bake (#1348). Stamps exactly the labels given
        (``BASE_IMAGE_LABEL_KEY`` + ``PREWARM_LABEL_KEY``) and DELIBERATELY
        none of ``MANAGED_LABEL_KEY``/``INSTANCE_LABEL_KEY``/
        ``SESSION_LABEL_KEY`` — that is what keeps the prewarm image out of the
        GC's reapable set. Raises :class:`SandboxBackendError` on failure.
        """
        ...

    async def prewarm_remove(self, sandbox_id: str) -> None:
        """``docker rm -f`` the transient prewarm container (#1348).

        Idempotent / best-effort: a container already gone is a no-op.
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

# ── Durable-session-sandbox labels (§5.1) ───────────────────────────────────
#
# All stamped at ``docker run`` and inherited into committed images
# automatically (verified). They make a container/image self-describing so
# the commit-time scrub and the GC/accounting work off labels alone — no DB
# lookup, including for a crash corpse salvaged after its spec is gone.
#
# ``ENV_KEYS_LABEL_KEY`` carries the comma-separated NAMES (never values) of
# every run-injected env var; the snapshot commit scrubs exactly that set
# from the image config. ``BASE_IMAGE_LABEL_KEY`` records which base ref the
# chain is rooted on (drives resume base-drift detection and unique-bytes
# accounting). ``FLATTENED_LABEL_KEY`` marks standalone export|import images
# that share no layers with the base, so accounting charges them full size.
ENV_KEYS_LABEL_KEY = "aios.env_keys"
BASE_IMAGE_LABEL_KEY = "aios.base_image"
FLATTENED_LABEL_KEY = "aios.flattened"
FLATTENED_LABEL_VALUE = "true"

# ── Operator prewarm label (#1348) ──────────────────────────────────────────
#
# A prewarm image is an operator-baked image that has ``install_egress_ca`` /
# ``install_packages`` already applied against the base, so the cold-start hot
# path can skip those (idempotent) setup execs. It is distinguished from a
# tenant snapshot PURELY by labels (variation-as-KIND-via-labels, consistent
# with ``MANAGED_LABEL_KEY``/``SESSION_LABEL_KEY``/``FLATTENED_LABEL_KEY``) — no
# new image-class flag. ``PREWARM_LABEL_KEY``'s VALUE is the base ref this image
# was baked against; the cold-start skip gate only fires when that value equals
# the CURRENT base ref (``spec.image``), so a base change disables the skip —
# the same equality discipline as the #916 base-drift gate. A prewarm image
# deliberately carries ``PREWARM_LABEL_KEY`` + ``BASE_IMAGE_LABEL_KEY`` but NOT
# ``MANAGED_LABEL_KEY``/``INSTANCE_LABEL_KEY``/``SESSION_LABEL_KEY``, so it never
# enters the image GC's reapable set (it is treated like the base image — an
# external operator-managed dependency).
PREWARM_LABEL_KEY = "aios.prewarmed"  # value = the base ref this image was baked against
