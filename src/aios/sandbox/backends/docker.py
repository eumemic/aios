"""Docker implementation of :class:`SandboxBackend`.

Shells out to the ``docker`` CLI via :func:`asyncio.create_subprocess_exec`
(the Python analog of JavaScript's ``execFile`` — passes argv directly to
the OS, does NOT invoke a host shell, no injection risk on the host
side). aiodocker is available as a dependency but its ergonomics for
simple one-shots are worse than the CLI; the CLI also gives us clean
stdout/stderr separation and timeout semantics out of the box.

Security note: the command string the agent supplies is passed as a
single argv element to ``bash -c`` inside the sandbox. The ``docker``
argv we build on the host contains ONLY trusted values
(container_id we created, the literal strings we synthesize, the
agent-supplied command as a single argv element). No host shell is
invoked on the outside; injection on the host side is impossible
because ``create_subprocess_exec`` does not interpret shell
metacharacters. Injection on the inside is by design — the agent IS
expected to run arbitrary shell inside the sandbox.
"""

from __future__ import annotations

import json

from aios.logging import get_logger
from aios.sandbox._subprocess import (
    run_docker_cli,
    run_docker_pipeline,
    run_subprocess_with_timeout,
)
from aios.sandbox.backends.base import (
    BASE_IMAGE_LABEL_KEY,
    ENV_KEYS_LABEL_KEY,
    FLATTENED_LABEL_KEY,
    FLATTENED_LABEL_VALUE,
    INSTANCE_LABEL_KEY,
    MANAGED_LABEL_KEY,
    MANAGED_LABEL_VALUE,
    SESSION_LABEL_KEY,
    CommandResult,
    ManagedImage,
    ManagedSandboxRef,
    SandboxBackendError,
    SandboxHandle,
    SandboxSpec,
    SnapshotOutcome,
)
from aios.sandbox.network import SANDBOX_NETWORK_NAME

log = get_logger("aios.sandbox.backends.docker")

# In-container timeout (#844). The agent command is wrapped in GNU
# coreutils ``timeout -k <kill-after> -s KILL <deadline>`` INSIDE the
# container so the SIGKILL actually reaches the daemon-spawned workload —
# host-side ``proc.kill()`` only ever hit the ``docker`` CLI, leaving the
# in-container process running (moby#9098).
_CONTAINER_TIMEOUT_KILL_AFTER_S = 5
_HOST_BACKSTOP_MARGIN_S = 5
# GNU ``timeout -s KILL`` exits 137 (128+9) when it force-kills the workload
# on timeout — verified in the sandbox image and locally. We map exactly this
# code to ``timed_out``: our invocation never yields the 124 (TERM-path) code
# from a timeout, and a command that exits 124 on its own is not a timeout.
_CONTAINER_TIMEOUT_EXIT_CODE = 137

# Chain-depth backstop for the snapshot verb. The overlay2 graphdriver
# hard-fails ``docker commit`` at ~125 stacked layers; the production
# **containerd image store** has no such wall (a chain committed and ran
# cleanly through 250 layers in the verification battery). So flatten is
# driven primarily by the per-session unique-bytes budget (§5.6); this
# depth ceiling is only a *soft* performance guard, kept generously below
# the kernel overlayfs lower-layer max so a budget-less session can't grow
# an unbounded chain. NOT a hard-wall dodge on the prod store.
_FLATTEN_DEPTH_CEILING = 200

# Constant floor + per-byte budget for the commit/flatten/export timeout,
# derived from the writable-layer ``SizeRw`` already read in the snapshot
# sequence. A fixed timeout would turn a genuinely huge writable layer into
# a permanent-brick retry loop (commit times out -> corpse retained ->
# salvage times out -> forever); a size-derived bound fires only on a
# genuinely hung daemon, never merely on size. ~10x a conservative
# 50 MB/s measured commit throughput => 20 ns/byte.
_SNAPSHOT_TIMEOUT_FLOOR_S = 60.0
_SNAPSHOT_TIMEOUT_NS_PER_BYTE = 20e-9


def _decode_and_truncate(raw: bytes, max_bytes: int) -> tuple[str, bool]:
    """Decode ``raw`` as UTF-8 (with replacement) and truncate to ``max_bytes``.

    Returns the decoded string and a flag indicating whether truncation
    happened. Truncation is byte-based (not char-based) to avoid surprises
    on very large outputs.
    """
    if len(raw) <= max_bytes:
        return raw.decode("utf-8", errors="replace"), False
    truncated = raw[:max_bytes].decode("utf-8", errors="replace")
    return truncated + "\n\n[output truncated]", True


class DockerBackend:
    """Sandbox backend backed by the host's Docker daemon."""

    name = "docker"

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Run ``docker run`` per ``spec`` and return a handle to the started container."""
        argv: list[str] = [
            "docker",
            "run",
            "--detach",
            # NB: NO ``--rm`` (durable session sandboxes). An unplanned death
            # (crash, OOM, daemon restart) must leave a stopped corpse that
            # the next provision / GC tick salvages (commits) — container
            # death stops losing data. Planned teardown snapshots then
            # removes explicitly.
            # Workspace bind mount.
            "--volume",
            _format_volume(
                spec.workspace.host_path, spec.workspace.sandbox_path, spec.workspace.read_only
            ),
        ]

        # Extra mounts (attachments, memory stores, github working trees).
        for mount in spec.extra_mounts:
            argv.extend(
                ["--volume", _format_volume(mount.host_path, mount.sandbox_path, mount.read_only)]
            )

        # Labels (the registry/spec builder is responsible for setting the
        # managed/instance/session labels; backend just passes through).
        for key, value in spec.labels.items():
            argv.extend(["--label", f"{key}={value}"])

        argv.extend(["--network", SANDBOX_NETWORK_NAME])

        # NB: the sandbox is NOT granted ``--cap-add NET_ADMIN`` (durable
        # session sandboxes, §5.8). The Limited-policy iptables lockdown is
        # applied from an ephemeral operator-image sidecar that joins this
        # container's netns (see ``aios.sandbox.setup.apply_network_lockdown``),
        # so root-in-sandbox can no longer modify netfilter at all — closing
        # both the persisted-poisoned-iptables bypass and the pre-existing
        # ``iptables -F your own lockdown`` hole.

        # WS0 phase-1 hardening: two flags safe to land before the exec-user
        # seam is in place.
        #
        # --security-opt no-new-privileges: blocks privilege *gain* across
        # execve (setuid bits, file capabilities).  Root setup execs don't
        # escalate — apt's _apt drop uses seteuid downward via CAP_SETUID;
        # the Limited iptables lockdown already runs as root holding NET_ADMIN.
        # No setuid binaries ship in the sandbox image today.  Any future
        # setuid binary added to the image will silently not escalate under
        # NNP — acceptable and intended.
        #
        # --ipc private: drops SysV-shm / POSIX-mq sharing with the host and
        # sibling containers.  Nothing uses cross-container IPC; this is the
        # modern Docker default — an explicit assertion rather than a behaviour
        # change.
        #
        # --cap-drop=ALL, --read-only, and a seccomp profile are deferred:
        # cap-drop interacts with apt/dpkg; --read-only conflicts with
        # root-agent workspace writes.  Track those in separate follow-up
        # issues.
        argv.extend(["--security-opt", "no-new-privileges"])
        argv.extend(["--ipc", "private"])

        if spec.host_gateway_alias is not None:
            argv.extend(["--add-host", f"{spec.host_gateway_alias}:host-gateway"])

        # Per-sandbox resource caps (#367 PR 9). All three are best-effort:
        # when unset (None) we leave Docker's host-default behavior in
        # place. The kernel OOM-kills the container when memory is
        # breached; CPU is throttled, not killed; pids-limit denies fork
        # past the cap with a clear errno.
        if spec.cpu_quota is not None:
            # ``f"{x:g}"`` flips to scientific notation around 1e-4
            # (e.g. ``"5e-05"``), which Docker rejects on ``--cpus``.
            # The settings floor is 0.01 so the printable range stays
            # plain-decimal, but format explicitly to avoid future drift.
            argv.extend(["--cpus", f"{spec.cpu_quota:.4f}".rstrip("0").rstrip(".")])
        if spec.memory_bytes is not None:
            argv.extend(["--memory", str(spec.memory_bytes)])
            # Pin swap to the same value so the sandbox can't lean on
            # swap to exceed the resident cap.
            argv.extend(["--memory-swap", str(spec.memory_bytes)])
        if spec.pids_limit is not None:
            argv.extend(["--pids-limit", str(spec.pids_limit)])
        # NB: NO ``--storage-opt size=`` (durable session sandboxes, §5.7).
        # The writable-layer quota required overlay2-on-xfs+pquota and was
        # rejected at ``docker run`` on prod's ext4 daemon — it never
        # worked. Disk pressure is now bounded by the snapshot pool budget
        # (GC eviction) instead; the live writable layer between commits
        # stays unbounded on ext4, same as today's status quo.

        # Authored seccomp deny-list (#807). ALWAYS emitted — never conditional —
        # so a misconfiguration can't silently fall back to Docker's default profile.
        # The value is a host path the docker CLI reads, or the literal "unconfined"
        # (emergency rollback via AIOS_SANDBOX_SECCOMP_PROFILE only).
        argv.extend(["--security-opt", f"seccomp={spec.seccomp_profile}"])

        # Keep stdin open so the container doesn't exit on empty stdin.
        argv.append("--interactive")

        # Environment.
        for key, value in spec.environment.items():
            argv.extend(["--env", f"{key}={value}"])

        # Resume source: run from the resolved snapshot tag when set
        # (durable session sandboxes), else the cold-start base image. The
        # registry's provision path does the verified resolution and sets
        # ``spec.snapshot_image`` to a locally-runnable tag; here we run it
        # blindly. A snapshot tag is single-component (``_is_registry_image``
        # False), so ``--pull always`` is correctly skipped for it — the
        # snapshot is local by construction and a registry lookup would fail.
        run_image = spec.snapshot_image or spec.image
        if _is_registry_image(run_image):
            argv.extend(["--pull", "always"])
        argv.append(run_image)

        rc, stdout_bytes, stderr_bytes = await run_docker_cli(argv)
        if rc != 0:
            raise SandboxBackendError(
                f"docker run failed (exit {rc}): "
                f"{stderr_bytes.decode('utf-8', errors='replace').strip()}"
            )

        container_id = stdout_bytes.decode("utf-8").strip()
        if not container_id:
            raise SandboxBackendError("docker run returned an empty container id")

        return SandboxHandle(
            session_id=spec.session_id,
            sandbox_id=container_id,
            workspace_path=spec.workspace.host_path,
            mount_snapshot=spec.mount_snapshot,
            spec_version=spec.spec_version,
            snapshot_image=spec.snapshot_image,
            disk_limit_bytes=spec.snapshot_budget_bytes,
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
        """Run ``bash -c <command>`` inside the container via ``docker exec``."""
        # Wrap the agent command in an in-container ``timeout`` so the SIGKILL on
        # deadline lands on the workload itself, not just the host ``docker exec``
        # client (#844 / moby#9098). The primary signal is already SIGKILL (-s KILL),
        # which is unblockable; ``-k 5`` is a redundant kill-after backstop kept to
        # match the agreed invocation and only matters if the primary signal is ever
        # softened.
        argv = [
            "docker",
            "exec",
            "--workdir",
            cwd,
            handle.sandbox_id,
            "timeout",
            "-k",
            str(_CONTAINER_TIMEOUT_KILL_AFTER_S),
            "-s",
            "KILL",
            str(timeout_seconds),
            "bash",
            "-c",
            command,
        ]
        # Host-side ``wait_for`` is now only a backstop for the rare case the
        # in-container ``timeout`` never returns (binary missing, docker exec
        # wedged). It MUST exceed the container deadline + its kill-after so it
        # never pre-empts the honest in-container path.
        host_timeout_s = float(
            timeout_seconds + _CONTAINER_TIMEOUT_KILL_AFTER_S + _HOST_BACKSTOP_MARGIN_S
        )
        rc, stdout_bytes, stderr_bytes, timed_out = await run_subprocess_with_timeout(
            argv, timeout_s=host_timeout_s
        )
        # When the in-container ``timeout`` fires it returns 137 well before the
        # host backstop, so the host-derived ``timed_out`` is False on a real
        # timeout. Map the timeout exit code so we stop telling the model the
        # command finished when it was actually killed (#844).
        timed_out = timed_out or rc == _CONTAINER_TIMEOUT_EXIT_CODE
        stdout_str, out_truncated = _decode_and_truncate(stdout_bytes, max_output_bytes)
        stderr_str, err_truncated = _decode_and_truncate(stderr_bytes, max_output_bytes)
        return CommandResult(
            exit_code=rc,
            stdout=stdout_str,
            stderr=stderr_str,
            timed_out=timed_out,
            truncated=out_truncated or err_truncated,
        )

    async def destroy(self, handle: SandboxHandle) -> None:
        """``docker rm --force`` the container. No-op if already gone."""
        argv = ["docker", "rm", "--force", handle.sandbox_id]
        try:
            rc, _, stderr_bytes = await run_docker_cli(argv)
        except SandboxBackendError as err:
            log.warning(
                "sandbox.destroy_launch_failed",
                session_id=handle.session_id,
                container_id=handle.sandbox_id[:12],
                error=str(err),
            )
            return
        if rc != 0:
            log.warning(
                "sandbox.destroy_nonzero",
                session_id=handle.session_id,
                container_id=handle.sandbox_id[:12],
                exit_code=rc,
                stderr=stderr_bytes.decode("utf-8", errors="replace").strip(),
            )
            return
        log.info(
            "sandbox.destroyed",
            session_id=handle.session_id,
            container_id=handle.sandbox_id[:12],
        )

    async def list_managed(
        self, *, instance_id: str, session_id: str | None = None
    ) -> list[ManagedSandboxRef]:
        """List managed containers (running AND stopped) for ``instance_id``.

        ``--all`` so stopped corpses are visible to the salvage/GC paths.
        ``session_id`` narrows to one session's containers (the salvage
        preamble). Each ref carries ``running`` from ``.State.Running``.
        """
        ps_argv = [
            "docker",
            "ps",
            "--all",
            "--quiet",
            "--filter",
            f"label={MANAGED_LABEL_KEY}={MANAGED_LABEL_VALUE}",
            "--filter",
            f"label={INSTANCE_LABEL_KEY}={instance_id}",
        ]
        if session_id is not None:
            ps_argv.extend(["--filter", f"label={SESSION_LABEL_KEY}={session_id}"])
        rc, stdout_bytes, stderr_bytes = await run_docker_cli(ps_argv)
        if rc != 0:
            raise SandboxBackendError(
                f"docker ps failed (exit {rc}): "
                f"{stderr_bytes.decode('utf-8', errors='replace').strip()}"
            )
        container_ids = [
            line.strip() for line in stdout_bytes.decode("utf-8").splitlines() if line.strip()
        ]
        if not container_ids:
            return []

        fmt = '{{.Id}}\t{{.State.Running}}\t{{index .Config.Labels "' + SESSION_LABEL_KEY + '"}}'
        inspect_argv = ["docker", "inspect", "--format", fmt, *container_ids]
        rc, stdout_bytes, stderr_bytes = await run_docker_cli(inspect_argv)
        if rc != 0:
            raise SandboxBackendError(
                f"docker inspect failed (exit {rc}): "
                f"{stderr_bytes.decode('utf-8', errors='replace').strip()}"
            )
        out: list[ManagedSandboxRef] = []
        for line in stdout_bytes.decode("utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split("\t", 2)
            cid = parts[0].strip()
            running = len(parts) > 1 and parts[1].strip() == "true"
            sid = parts[2].strip() if len(parts) > 2 else ""
            out.append(ManagedSandboxRef(sandbox_id=cid, session_id=sid or None, running=running))
        return out

    async def force_remove(self, sandbox_id: str) -> None:
        """``docker rm --force`` a container by id. Logs but does not raise."""
        argv = ["docker", "rm", "--force", sandbox_id]
        try:
            rc, _, stderr_bytes = await run_docker_cli(argv)
        except SandboxBackendError as err:
            log.warning(
                "sandbox.force_remove_launch_failed",
                container_id=sandbox_id[:12],
                error=str(err),
            )
            return
        if rc != 0:
            log.warning(
                "sandbox.force_remove_nonzero",
                container_id=sandbox_id[:12],
                exit_code=rc,
                stderr=stderr_bytes.decode("utf-8", errors="replace").strip(),
            )

    async def is_alive(self, handle: SandboxHandle) -> bool:
        """``docker inspect`` the container; True iff State.Running is true.

        Probe failures (daemon hiccup, launch error, timeout) return
        False — better to take the cost of a re-provision than hand
        back a possibly-dead handle. Per #691, ``--rm`` containers that
        die between provision and the next call appear here as a
        nonzero ``rc`` (Docker's "No such container").
        """
        argv = [
            "docker",
            "inspect",
            "--format",
            "{{.State.Running}}",
            handle.sandbox_id,
        ]
        try:
            rc, stdout_bytes, _ = await run_docker_cli(argv)
        except Exception as err:
            # Total by contract (see Protocol docstring): ANY probe failure
            # — daemon hiccup, CLI launch error, 30s timeout — means we
            # couldn't confirm liveness, so report dead and let the caller
            # re-provision rather than hand back a possibly-dead handle.
            # ``CancelledError`` (BaseException, not Exception) is NOT caught,
            # so worker shutdown / job-timeout cancellation still propagates.
            log.warning(
                "sandbox.is_alive_probe_failed",
                session_id=handle.session_id,
                container_id=handle.sandbox_id[:12],
                error=str(err),
            )
            return False
        if rc != 0:
            # A container removed out of band (crash/OOM that left no corpse,
            # or an operator rm) reports nonzero + "No such container".
            log.info(
                "sandbox.is_alive_inspect_nonzero",
                session_id=handle.session_id,
                container_id=handle.sandbox_id[:12],
                exit_code=rc,
            )
            return False
        return stdout_bytes.decode("utf-8", errors="replace").strip() == "true"

    # ── durable-session-sandbox verbs (§5.2) ────────────────────────────────

    async def stop(self, sandbox_id: str) -> None:
        """``docker stop -t 5``. Idempotent on a stopped corpse / missing container."""
        argv = ["docker", "stop", "-t", "5", sandbox_id]
        rc, _stdout, stderr_bytes = await run_docker_cli(argv)
        if rc != 0:
            stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
            if _is_no_such_container(stderr):
                return
            log.warning("sandbox.stop_nonzero", container_id=sandbox_id[:12], stderr=stderr)

    async def snapshot(
        self,
        sandbox_id: str,
        tag: str,
        *,
        empty_floor_bytes: int,
        flatten_if_unique_bytes_over: int | None,
    ) -> SnapshotOutcome:
        """Commit (or flatten) ``sandbox_id``'s rootfs to ``tag`` (§5.2)."""
        # 1. Stop (idempotent — the corpse may already be stopped).
        await self.stop(sandbox_id)

        # 2. Inspect the corpse: parent image, writable-layer bytes, labels.
        parent_image, size_rw, labels = await self._inspect_container_for_snapshot(sandbox_id)
        env_keys = _split_label_list(labels.get(ENV_KEYS_LABEL_KEY))
        base_ref = labels.get(BASE_IMAGE_LABEL_KEY)

        # 3. Inspect the existing tag (absent → first snapshot).
        tag_fields = await self._inspect_image_fields(tag)

        # 4. LINEAGE GATE: proceed iff the tag is absent, or the tag is exactly
        #    the image this corpse was created from. A tag that has moved past
        #    the corpse's parent is content-equal crash residue (commit-then-
        #    crash-before-rm under the salvage-before-provision invariant) — the
        #    corpse carries nothing newer than the tag, so discard it.
        if tag_fields is not None and tag_fields[0] != parent_image:
            return SnapshotOutcome(
                kind="skipped_stale",
                image_id=tag_fields[0],
                unique_bytes=await self._unique_bytes(tag_fields, base_ref),
                depth=tag_fields[2],
            )

        # 5. Identity short-circuit: a writable layer at/below the empty floor
        #    produces content identical to the existing tag. On the containerd
        #    image store a no-write container reports SizeRw == 4096 (NOT 0),
        #    so the floor — not an == 0 test — is what keeps chat-only and
        #    read-only sessions from ever growing a chain.
        if size_rw is not None and size_rw <= empty_floor_bytes:
            if tag_fields is None:
                return SnapshotOutcome(kind="skipped_empty", image_id=None, unique_bytes=0, depth=0)
            return SnapshotOutcome(
                kind="skipped_empty",
                image_id=tag_fields[0],
                unique_bytes=await self._unique_bytes(tag_fields, base_ref),
                depth=tag_fields[2],
            )

        # 6. Decide commit vs flatten. Depth is a soft performance backstop
        #    (no hard layer wall on the containerd store); the per-session
        #    unique-bytes budget is the primary trigger.
        base_size = await self._image_size_or_zero(base_ref)
        parent_fields = await self._inspect_image_fields(parent_image)
        parent_size = parent_fields[1] if parent_fields else 0
        parent_depth = parent_fields[2] if parent_fields else 1
        rw = size_rw if size_rw is not None else 0
        projected_unique = max(0, parent_size - base_size) + rw
        over_budget = (
            flatten_if_unique_bytes_over is not None
            and projected_unique > flatten_if_unique_bytes_over
        )
        # Timeout scales with the bytes the daemon must move (resulting image
        # for commit; whole rootfs for export) so a huge layer never wedges
        # the worker in a permanent retry loop, while a hung daemon still trips.
        timeout_s = _SNAPSHOT_TIMEOUT_FLOOR_S + (parent_size + rw) * _SNAPSHOT_TIMEOUT_NS_PER_BYTE

        if parent_depth + 1 >= _FLATTEN_DEPTH_CEILING or over_budget:
            return await self._flatten(sandbox_id, tag, labels, timeout_s=timeout_s)
        return await self._commit(sandbox_id, tag, env_keys, base_ref, timeout_s=timeout_s)

    async def _commit(
        self,
        sandbox_id: str,
        tag: str,
        env_keys: list[str],
        base_ref: str | None,
        *,
        timeout_s: float,
    ) -> SnapshotOutcome:
        """``docker commit`` one layer, scrubbing exactly the run-injected env keys.

        Scrub scope is the ``aios.env_keys`` set ONLY (``ENV K=``). Scrubbing
        all of ``.Config.Env`` empties the base image's ``PATH``/``HOME`` and
        bricks every resumed container (Docker won't re-inject a default PATH
        for a present-but-empty key — verified). ``ENV K=`` empties rather than
        unsets; removed vars read as ``""`` until the next flatten strips
        config entirely.
        """
        argv = ["docker", "commit"]
        for key in env_keys:
            argv.extend(["--change", f"ENV {key}="])
        argv.extend([sandbox_id, tag])
        rc, _stdout, stderr_bytes = await run_docker_cli(argv, timeout_s=timeout_s)
        if rc != 0:
            raise SandboxBackendError(
                f"docker commit failed (exit {rc}) for {tag}: "
                f"{stderr_bytes.decode('utf-8', errors='replace').strip()}"
            )
        new = await self._inspect_image_fields(tag)
        if new is None:
            raise SandboxBackendError(f"committed image {tag} not found after commit")
        return SnapshotOutcome(
            kind="committed",
            image_id=new[0],
            unique_bytes=await self._unique_bytes(new, base_ref),
            depth=new[2],
        )

    async def _flatten(
        self,
        sandbox_id: str,
        tag: str,
        labels: dict[str, str],
        *,
        timeout_s: float,
    ) -> SnapshotOutcome:
        """``docker export | docker import`` — collapse the chain to one layer.

        Flatten applies overlay whiteouts (so "delete files to shrink" finally
        works) and strips ALL baked config (the definitive secret scrub). Import
        loses the base image's config, so restore ``WORKDIR``/``HOME``/``CMD``
        and re-stamp the managed labels. ``PATH`` is deliberately NOT restored.
        The restored ``CMD`` invokes ``tail`` by absolute path
        (``/usr/bin/tail``), so it no longer depends on PATH resolution — a
        PATH regression can't break container init (defense-in-depth after
        #925). The resumed container's env is re-injected fresh via
        ``docker run --env`` regardless, so config env doesn't affect runtime
        — only the artifact.
        """
        changes = [
            'CMD ["/usr/bin/tail","-f","/dev/null"]',
            "WORKDIR /workspace",
            "ENV HOME=/home/aios",
            f"LABEL {MANAGED_LABEL_KEY}={MANAGED_LABEL_VALUE}",
            f"LABEL {FLATTENED_LABEL_KEY}={FLATTENED_LABEL_VALUE}",
        ]
        for key in (
            INSTANCE_LABEL_KEY,
            SESSION_LABEL_KEY,
            ENV_KEYS_LABEL_KEY,
            BASE_IMAGE_LABEL_KEY,
        ):
            value = labels.get(key)
            if value is not None:
                changes.append(f"LABEL {key}={value}")
        consumer = ["docker", "import"]
        for change in changes:
            consumer.extend(["--change", change])
        consumer.extend(["-", tag])
        await run_docker_pipeline(["docker", "export", sandbox_id], consumer, timeout_s=timeout_s)
        new = await self._inspect_image_fields(tag)
        if new is None:
            raise SandboxBackendError(f"flattened image {tag} not found after import")
        # A flattened image is standalone — it shares no layers with the base,
        # so it is charged its FULL size (subtracting a base it doesn't share
        # would hide ~hundreds of MB from the accounting that must see the host
        # filling).
        return SnapshotOutcome(kind="flattened", image_id=new[0], unique_bytes=new[1], depth=new[2])

    async def list_managed_images(self, *, instance_id: str) -> list[ManagedImage]:
        """Enumerate managed images (incl. untagged residue) via ``docker images -a``."""
        ls_argv = [
            "docker",
            "images",
            "--all",
            "--no-trunc",
            "--quiet",
            "--filter",
            f"label={MANAGED_LABEL_KEY}={MANAGED_LABEL_VALUE}",
            "--filter",
            f"label={INSTANCE_LABEL_KEY}={instance_id}",
        ]
        rc, stdout_bytes, stderr_bytes = await run_docker_cli(ls_argv)
        if rc != 0:
            raise SandboxBackendError(
                f"docker images failed (exit {rc}): "
                f"{stderr_bytes.decode('utf-8', errors='replace').strip()}"
            )
        # ``docker images -q`` repeats an id once per tag; dedupe preserving order.
        seen: set[str] = set()
        image_ids: list[str] = []
        for line in stdout_bytes.decode("utf-8").splitlines():
            iid = line.strip()
            if iid and iid not in seen:
                seen.add(iid)
                image_ids.append(iid)
        if not image_ids:
            return []

        # Whole-Config form for labels — see ``_inspect_image_fields`` on why a
        # direct ``{{json .Config.Labels}}`` errors on label-less images under
        # Docker inspect's ``missingkey=error``.
        fmt = "{{.Id}}\t{{.Parent}}\t{{.Size}}\t{{json .RepoTags}}\t{{json .Config}}"
        rc, stdout_bytes, stderr_bytes = await run_docker_cli(
            ["docker", "image", "inspect", "--format", fmt, *image_ids]
        )
        if rc != 0:
            raise SandboxBackendError(
                f"docker image inspect failed (exit {rc}): "
                f"{stderr_bytes.decode('utf-8', errors='replace').strip()}"
            )
        out: list[ManagedImage] = []
        for line in stdout_bytes.decode("utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split("\t", 4)
            image_id = parts[0].strip()
            parent = parts[1].strip() if len(parts) > 1 else ""
            size = int(parts[2]) if len(parts) > 2 and parts[2].strip().isdigit() else 0
            repo_tags = _parse_json_str_list(parts[3]) if len(parts) > 3 else ()
            img_labels = _labels_from_config_json(parts[4]) if len(parts) > 4 else {}
            out.append(
                ManagedImage(
                    image_id=image_id,
                    repo_tags=repo_tags,
                    parent_id=parent or None,
                    size_bytes=size,
                    labels=img_labels,
                )
            )
        return out

    async def remove_image(self, ref: str) -> bool:
        """``docker rmi`` (no force). True on removed/already-gone, False on refusal."""
        rc, _stdout, stderr_bytes = await run_docker_cli(["docker", "rmi", ref])
        if rc == 0:
            return True
        stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
        if _is_no_such_image(stderr):
            return True  # idempotent: goal state (image gone) already reached
        log.info("sandbox.rmi_refused", ref=ref, stderr=stderr)
        return False

    async def image_size(self, image: str) -> int:
        fields = await self._inspect_image_fields(image)
        if fields is None:
            raise SandboxBackendError(f"image not found: {image}")
        return fields[1]

    async def image_labels(self, image: str) -> dict[str, str] | None:
        fields = await self._inspect_image_fields(image)
        return None if fields is None else fields[3]

    async def run_netns_sidecar(
        self,
        target_sandbox_id: str,
        *,
        image: str,
        script: str,
        timeout_seconds: int,
        max_output_bytes: int,
    ) -> CommandResult:
        """Apply/verify the network lockdown from an ephemeral operator-image sidecar.

        ``--network container:<id>`` shares the sandbox's netns; ``--cap-add
        NET_ADMIN`` lets the sidecar edit netfilter in that shared namespace;
        ``--rm`` removes it on exit (the rules persist in the netns, held by
        the sandbox). No restrictive seccomp is applied — this is an
        operator-trusted, short-lived container, and iptables needs the
        syscalls the default profile permits.
        """
        argv = [
            "docker",
            "run",
            "--rm",
            "--network",
            f"container:{target_sandbox_id}",
            "--cap-add",
            "NET_ADMIN",
            image,
            "bash",
            "-c",
            script,
        ]
        rc, stdout_bytes, stderr_bytes, timed_out = await run_subprocess_with_timeout(
            argv, timeout_s=float(timeout_seconds)
        )
        if timed_out:
            raise SandboxBackendError(
                f"network-lockdown sidecar timed out after {timeout_seconds}s for "
                f"{target_sandbox_id[:12]}"
            )
        stdout_str, out_truncated = _decode_and_truncate(stdout_bytes, max_output_bytes)
        stderr_str, err_truncated = _decode_and_truncate(stderr_bytes, max_output_bytes)
        return CommandResult(
            exit_code=rc,
            stdout=stdout_str,
            stderr=stderr_str,
            timed_out=False,
            truncated=out_truncated or err_truncated,
        )

    # ── snapshot-verb helpers ───────────────────────────────────────────────

    async def _inspect_container_for_snapshot(
        self, sandbox_id: str
    ) -> tuple[str, int | None, dict[str, str]]:
        """Return ``(parent_image_id, size_rw, labels)`` for a container.

        ``size_rw`` is ``None`` only when the daemon reports an unparseable
        value — the snapshot path then declines the empty short-circuit
        (commit rather than risk losing data).
        """
        fmt = "{{.Image}}\t{{.SizeRw}}\t{{json .Config.Labels}}"
        rc, stdout_bytes, stderr_bytes = await run_docker_cli(
            ["docker", "inspect", "--size", "--format", fmt, sandbox_id]
        )
        if rc != 0:
            raise SandboxBackendError(
                f"docker inspect (container) failed (exit {rc}) for {sandbox_id[:12]}: "
                f"{stderr_bytes.decode('utf-8', errors='replace').strip()}"
            )
        parts = stdout_bytes.decode("utf-8").rstrip("\n").split("\t", 2)
        parent_image = parts[0].strip()
        raw_rw = parts[1].strip() if len(parts) > 1 else ""
        size_rw = int(raw_rw) if raw_rw.isdigit() else None
        labels = _parse_json_labels(parts[2]) if len(parts) > 2 else {}
        return parent_image, size_rw, labels

    async def _inspect_image_fields(self, ref: str) -> tuple[str, int, int, dict[str, str]] | None:
        """Return ``(image_id, size_bytes, layer_depth, labels)`` or ``None`` if absent.

        Verified-negative: a confirmed "No such image" returns ``None``; any
        other nonzero exit raises (an indeterminate probe must not read as
        absence).

        Labels are extracted from the whole ``{{json .Config}}`` object rather
        than ``{{json .Config.Labels}}``: Docker's inspect applies
        ``missingkey=error`` to multi-action templates, so a direct
        ``.Config.Labels`` access errors on an image with NO labels (the base
        image) — ``map has no entry for key "Labels"`` — while the whole-Config
        form is safe for labeled and unlabeled images alike (verified).
        """
        fmt = "{{.Id}}\t{{.Size}}\t{{len .RootFS.Layers}}\t{{json .Config}}"
        rc, stdout_bytes, stderr_bytes = await run_docker_cli(
            ["docker", "image", "inspect", "--format", fmt, ref]
        )
        if rc != 0:
            stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
            if _is_no_such_image(stderr):
                return None
            raise SandboxBackendError(f"docker image inspect failed for {ref}: {stderr}")
        parts = stdout_bytes.decode("utf-8").rstrip("\n").split("\t", 3)
        image_id = parts[0].strip()
        size = int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else 0
        depth = int(parts[2]) if len(parts) > 2 and parts[2].strip().isdigit() else 1
        labels = _labels_from_config_json(parts[3]) if len(parts) > 3 else {}
        return image_id, size, depth, labels

    async def _image_size_or_zero(self, ref: str | None) -> int:
        """Base size for accounting; 0 when the base ref is unknown/absent.

        Over-counting (charging full size when the base can't be resolved) is
        safe for budget enforcement — it never under-reports the host filling.
        """
        if not ref:
            return 0
        try:
            return await self.image_size(ref)
        except SandboxBackendError:
            return 0

    async def _unique_bytes(
        self, image_fields: tuple[str, int, int, dict[str, str]], base_ref: str | None
    ) -> int:
        """Unique bytes for the accounting pointer: full size for a flattened
        (standalone) image, else ``tag.Size - base.Size``."""
        _image_id, size, _depth, labels = image_fields
        if labels.get(FLATTENED_LABEL_KEY) == FLATTENED_LABEL_VALUE:
            return size
        return max(0, size - await self._image_size_or_zero(base_ref))


def _split_label_list(value: str | None) -> list[str]:
    """Parse a comma-separated label value (e.g. ``aios.env_keys``) into names."""
    if not value:
        return []
    return [item for item in (part.strip() for part in value.split(",")) if item]


def _parse_json_labels(raw: str) -> dict[str, str]:
    """Parse a ``{{json .Config.Labels}}`` field into a label map.

    Docker emits ``null`` for an image/container with no labels and
    ``<no value>`` when the field is entirely absent — both map to ``{}``.
    """
    raw = raw.strip()
    if not raw or raw in ("null", "<no value>"):
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): str(v) for k, v in parsed.items()}


def _labels_from_config_json(raw: str) -> dict[str, str]:
    """Extract ``.Config.Labels`` from a ``{{json .Config}}`` field.

    Used instead of ``{{json .Config.Labels}}`` for IMAGE inspects: Docker's
    inspect applies ``missingkey=error`` to multi-action templates, so a direct
    ``.Config.Labels`` access errors on a label-less image (the base) with
    ``map has no entry for key "Labels"``. The whole-Config form is safe for
    both labeled and unlabeled images; ``Labels`` may be absent or ``null``,
    both of which map to ``{}``.
    """
    raw = raw.strip()
    if not raw or raw in ("null", "<no value>"):
        return {}
    try:
        config = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(config, dict):
        return {}
    labels = config.get("Labels")
    if not isinstance(labels, dict):
        return {}
    return {str(k): str(v) for k, v in labels.items()}


def _parse_json_str_list(raw: str) -> tuple[str, ...]:
    """Parse a ``{{json .RepoTags}}`` field into a tuple of tags (``()`` if none)."""
    raw = raw.strip()
    if not raw or raw in ("null", "<no value>"):
        return ()
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(str(item) for item in parsed)


def _is_no_such_container(stderr: str) -> bool:
    return "no such container" in stderr.lower()


def _is_no_such_image(stderr: str) -> bool:
    lowered = stderr.lower()
    return "no such image" in lowered or "no such object" in lowered


def _is_registry_image(image: str) -> bool:
    """Return True if *image* refers to a remote registry (not a bare local tag).

    Docker treats bare names (e.g. ``aios-sandbox:latest``) as
    ``docker.io/library/<name>``, so ``--pull always`` would trigger a Hub
    lookup that fails for locally-built images.  We only add the flag when the
    image clearly references a remote registry.

    Rules:
    - Single component (no ``/``) → bare name, local.
    - First component contains ``.`` or ``:`` → registry hostname (e.g.
      ``ghcr.io``, ``localhost:5000``).
    - First component is ``localhost`` → local registry, but still a daemon
      push/pull target → treat as registry.
    - Otherwise (e.g. ``myorg/myimage``) → Docker Hub short form, no explicit
      hostname → local-enough that ``--pull always`` is unsafe.

    Note: only the *first* path component is inspected, so a tag suffix on
    the final component (e.g. ``localhost:5000/foo:bar``) does not confuse
    the check.
    """
    parts = image.split("/")
    if len(parts) == 1:
        return False  # bare name like "ubuntu" or "aios-sandbox[:tag]"
    first = parts[0]
    return first == "localhost" or "." in first or ":" in first


def _format_volume(host_path: object, sandbox_path: str, read_only: bool) -> str:
    """Format a single ``--volume`` argument value."""
    suffix = ":ro" if read_only else ""
    return f"{host_path}:{sandbox_path}{suffix}"
