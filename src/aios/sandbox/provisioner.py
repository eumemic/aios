"""Provision and tear down sandbox containers.

The lazy-provisioning model (see ``aios_v1_decisions.md``): a container is
created the first time a session's tool call needs one, not at session
creation or worker assignment. Creation runs ``docker run`` with the
standard aios image, a bind mount from the session's workspace directory
to ``/workspace``, the ``aios.managed=true`` label so the worker's orphan
reaper can find it, and a configurable network mode.

Teardown runs ``docker rm --force`` to stop + remove the container. The
workspace directory stays on disk across teardowns — the next wake will
reuse it when a fresh container is provisioned.

The module shells out to the ``docker`` CLI via
:func:`asyncio.create_subprocess_exec` (the Python analog of JavaScript's
``execFile`` — passes argv directly to the OS, does NOT invoke a shell,
no injection risk on the host side). aiodocker is available as a
dependency but its ergonomics for simple one-shots are worse than the
CLI.
"""

from __future__ import annotations

import asyncio

from aios.config import get_settings
from aios.logging import get_logger
from aios.sandbox.container import ContainerError, ContainerHandle
from aios.sandbox.volumes import ensure_workspace_dir

log = get_logger("aios.sandbox.provisioner")

# Label applied at ``docker run`` time. The worker's orphan reaper lists
# containers with this label at startup and removes any whose session_id
# does not match an active worker lease.
MANAGED_LABEL_KEY = "aios.managed"
MANAGED_LABEL_VALUE = "true"
SESSION_LABEL_KEY = "aios.session_id"


async def _run_docker(argv: list[str]) -> tuple[int, bytes, bytes]:
    """Run a ``docker`` CLI invocation and return (exit_code, stdout, stderr).

    Raises :class:`ContainerError` if the subprocess cannot be launched
    at all (missing docker binary, OS error). A nonzero exit from docker
    itself is returned as a regular tuple — callers decide whether it's
    fatal.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (OSError, FileNotFoundError) as host_err:
        raise ContainerError(f"failed to launch docker cli: {host_err}") from host_err
    stdout_bytes, stderr_bytes = await proc.communicate()
    return proc.returncode or 0, stdout_bytes, stderr_bytes


async def provision_for_session(session_id: str) -> ContainerHandle:
    """Create a fresh container for ``session_id`` and return a handle.

    Idempotency: this function does NOT check whether a container already
    exists for the session. Callers (specifically :mod:`aios.sandbox.registry`)
    are responsible for caching the handle and not calling this twice for
    the same session within a turn.

    Raises :class:`ContainerError` if ``docker run`` fails for any reason
    (image missing, daemon unreachable, resource limits, ...).
    """
    settings = get_settings()
    workspace_path = ensure_workspace_dir(session_id)

    argv = [
        "docker",
        "run",
        "--detach",
        "--rm",
        # Bind-mount the session's workspace into /workspace inside the
        # container. Must be an absolute path on the host.
        "--volume",
        f"{workspace_path}:/workspace",
        "--label",
        f"{MANAGED_LABEL_KEY}={MANAGED_LABEL_VALUE}",
        "--label",
        f"{SESSION_LABEL_KEY}={session_id}",
        "--network",
        settings.sandbox_network_mode,
        # Keep stdin open so the container doesn't exit on empty stdin.
        "--interactive",
        settings.docker_image,
    ]

    rc, stdout_bytes, stderr_bytes = await _run_docker(argv)
    if rc != 0:
        raise ContainerError(
            f"docker run failed (exit {rc}): "
            f"{stderr_bytes.decode('utf-8', errors='replace').strip()}"
        )

    container_id = stdout_bytes.decode("utf-8").strip()
    if not container_id:
        raise ContainerError("docker run returned an empty container id")

    log.info(
        "sandbox.provisioned",
        session_id=session_id,
        container_id=container_id[:12],
        workspace_path=str(workspace_path),
    )

    return ContainerHandle(
        session_id=session_id,
        container_id=container_id,
        workspace_path=workspace_path,
    )


async def release(handle: ContainerHandle) -> None:
    """Stop and remove ``handle``'s container.

    Uses ``docker rm --force`` which sends SIGKILL and removes the
    container in one step. If the container is already gone (daemon
    forgot it, user removed it manually), this is a no-op — we log but
    don't raise. The workspace directory on the host is NOT deleted.
    """
    argv = ["docker", "rm", "--force", handle.container_id]
    try:
        rc, _, stderr_bytes = await _run_docker(argv)
    except ContainerError as err:
        log.warning(
            "sandbox.release_launch_failed",
            session_id=handle.session_id,
            container_id=handle.container_id[:12],
            error=str(err),
        )
        return
    if rc != 0:
        log.warning(
            "sandbox.release_nonzero",
            session_id=handle.session_id,
            container_id=handle.container_id[:12],
            exit_code=rc,
            stderr=stderr_bytes.decode("utf-8", errors="replace").strip(),
        )
        return

    log.info(
        "sandbox.released",
        session_id=handle.session_id,
        container_id=handle.container_id[:12],
    )


async def list_managed_containers() -> list[tuple[str, str]]:
    """List all containers labelled ``aios.managed=true``.

    Returns a list of ``(container_id, session_id)`` tuples, where
    ``session_id`` is read from the container's ``aios.session_id`` label.
    Containers missing the session label are returned with an empty
    string as their session_id (shouldn't happen, but we're defensive).

    Used by the worker's orphan reaper at startup.
    """
    ps_argv = [
        "docker",
        "ps",
        "--quiet",
        "--filter",
        f"label={MANAGED_LABEL_KEY}={MANAGED_LABEL_VALUE}",
    ]
    rc, stdout_bytes, stderr_bytes = await _run_docker(ps_argv)
    if rc != 0:
        raise ContainerError(
            f"docker ps failed (exit {rc}): "
            f"{stderr_bytes.decode('utf-8', errors='replace').strip()}"
        )
    container_ids = [
        line.strip() for line in stdout_bytes.decode("utf-8").splitlines() if line.strip()
    ]
    if not container_ids:
        return []

    # For each id, inspect the session_id label.
    fmt = '{{.Id}}\t{{index .Config.Labels "' + SESSION_LABEL_KEY + '"}}'
    inspect_argv = ["docker", "inspect", "--format", fmt, *container_ids]
    rc, stdout_bytes, stderr_bytes = await _run_docker(inspect_argv)
    if rc != 0:
        raise ContainerError(
            f"docker inspect failed (exit {rc}): "
            f"{stderr_bytes.decode('utf-8', errors='replace').strip()}"
        )
    out: list[tuple[str, str]] = []
    for line in stdout_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        cid = parts[0].strip()
        sid = parts[1].strip() if len(parts) > 1 else ""
        out.append((cid, sid))
    return out


async def force_remove(container_id: str) -> None:
    """Force-remove a container by id. Used by the orphan reaper.

    Logs but does not raise on failure — a container that's already gone
    is fine.
    """
    argv = ["docker", "rm", "--force", container_id]
    try:
        rc, _, stderr_bytes = await _run_docker(argv)
    except ContainerError as err:
        log.warning(
            "sandbox.force_remove_launch_failed",
            container_id=container_id[:12],
            error=str(err),
        )
        return
    if rc != 0:
        log.warning(
            "sandbox.force_remove_nonzero",
            container_id=container_id[:12],
            exit_code=rc,
            stderr=stderr_bytes.decode("utf-8", errors="replace").strip(),
        )
