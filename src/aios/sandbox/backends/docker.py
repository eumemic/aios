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

from aios.logging import get_logger
from aios.sandbox._subprocess import run_docker_cli, run_subprocess_with_timeout
from aios.sandbox.backends.base import (
    INSTANCE_LABEL_KEY,
    MANAGED_LABEL_KEY,
    MANAGED_LABEL_VALUE,
    SESSION_LABEL_KEY,
    CommandResult,
    Limited,
    ManagedSandboxRef,
    SandboxBackendError,
    SandboxHandle,
    SandboxSpec,
)
from aios.sandbox.network import SANDBOX_NETWORK_NAME

log = get_logger("aios.sandbox.backends.docker")


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
            "--rm",
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

        # Limited policy needs NET_ADMIN so the iptables script (applied
        # later by the registry via aios.sandbox.setup) can install rules.
        if isinstance(spec.network_policy, Limited):
            argv.extend(["--cap-add", "NET_ADMIN"])

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

        # Keep stdin open so the container doesn't exit on empty stdin.
        argv.append("--interactive")

        # Environment.
        for key, value in spec.environment.items():
            argv.extend(["--env", f"{key}={value}"])

        # Only pull from registry for remote images; local/bare tags (dev builds) have no registry prefix
        if _is_registry_image(spec.image):
            argv.extend(["--pull", "always"])
        argv.append(spec.image)

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
        argv = [
            "docker",
            "exec",
            "--workdir",
            cwd,
            handle.sandbox_id,
            "bash",
            "-c",
            command,
        ]
        rc, stdout_bytes, stderr_bytes, timed_out = await run_subprocess_with_timeout(
            argv, timeout_s=float(timeout_seconds)
        )
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

    async def list_managed(self, *, instance_id: str) -> list[ManagedSandboxRef]:
        """List ``aios.managed=true`` containers for ``instance_id``."""
        ps_argv = [
            "docker",
            "ps",
            "--quiet",
            "--filter",
            f"label={MANAGED_LABEL_KEY}={MANAGED_LABEL_VALUE}",
            "--filter",
            f"label={INSTANCE_LABEL_KEY}={instance_id}",
        ]
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

        fmt = '{{.Id}}\t{{index .Config.Labels "' + SESSION_LABEL_KEY + '"}}'
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
            parts = line.split("\t", 1)
            cid = parts[0].strip()
            sid = parts[1].strip() if len(parts) > 1 else ""
            out.append(ManagedSandboxRef(sandbox_id=cid, session_id=sid or None))
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


def _is_registry_image(image: str) -> bool:
    """Return True if *image* refers to a remote registry (not a bare local tag).

    Docker treats bare names (e.g. ``aios-sandbox:latest``) as
    ``docker.io/library/<name>``, so ``--pull always`` would trigger a Hub
    lookup that fails for locally-built images.  We only add the flag when the
    image clearly references a remote registry.

    Rules (applied to the name part before any ``:tag`` suffix):
    - Single component (no ``/``) → bare name, local.
    - First component contains ``.`` or ``:`` → registry hostname (e.g.
      ``ghcr.io``, ``localhost:5000``).
    - First component is ``localhost`` → local registry, but still a daemon
      push/pull target → treat as registry.
    - Otherwise (e.g. ``myorg/myimage``) → Docker Hub short form, no explicit
      hostname → local-enough that ``--pull always`` is unsafe.
    """
    name = image.split(":")[0]  # strip tag
    parts = name.split("/")
    if len(parts) == 1:
        return False  # bare name like "ubuntu" or "aios-sandbox"
    first = parts[0]
    return first == "localhost" or "." in first or ":" in first


def _format_volume(host_path: object, sandbox_path: str, read_only: bool) -> str:
    """Format a single ``--volume`` argument value."""
    suffix = ":ro" if read_only else ""
    return f"{host_path}:{sandbox_path}{suffix}"
