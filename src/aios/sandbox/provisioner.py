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
from aios.db import queries
from aios.logging import get_logger
from aios.models.environments import EnvironmentConfig, LimitedNetworking
from aios.sandbox.container import ContainerError, ContainerHandle

log = get_logger("aios.sandbox.provisioner")

# Label applied at ``docker run`` time. The worker's orphan reaper lists
# containers with this label at startup and removes any whose session_id
# does not match an active worker lease.
MANAGED_LABEL_KEY = "aios.managed"
MANAGED_LABEL_VALUE = "true"
SESSION_LABEL_KEY = "aios.session_id"

# Well-known hosts for public package registries.  Added to the iptables
# allowlist when ``allow_package_managers`` is True in limited networking.
PACKAGE_REGISTRY_HOSTS: frozenset[str] = frozenset(
    {
        # Python (pip)
        "pypi.org",
        "files.pythonhosted.org",
        # Node (npm)
        "registry.npmjs.org",
        # Rust (cargo)
        "crates.io",
        "static.crates.io",
        # Ruby (gem)
        "rubygems.org",
        # Go
        "proxy.golang.org",
        "sum.golang.org",
        # Debian/Ubuntu (apt)
        "deb.debian.org",
        "security.debian.org",
        # Common CDN used by package managers
        "github.com",
        "objects.githubusercontent.com",
    }
)


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


async def _load_environment_config(session_id: str) -> EnvironmentConfig | None:
    """Load the environment config for a session from the DB.

    Returns ``None`` if the session or environment doesn't exist (shouldn't
    happen in normal flow, but callers handle it gracefully).
    """
    from aios.harness import runtime

    pool = runtime.require_pool()
    async with pool.acquire() as conn:
        return await queries.get_environment_config_for_session(conn, session_id)


async def _load_workspace_path(session_id: str) -> str:
    """Load the workspace volume path for a session from the DB."""
    from aios.harness import runtime

    pool = runtime.require_pool()
    async with pool.acquire() as conn:
        return await queries.get_session_workspace_path(conn, session_id)


async def _load_session_env(session_id: str) -> dict[str, str]:
    """Load per-session environment variables from the DB."""
    from aios.harness import runtime

    pool = runtime.require_pool()
    async with pool.acquire() as conn:
        return await queries.get_session_env(conn, session_id)


async def provision_for_session(session_id: str) -> ContainerHandle:
    """Create a fresh container for ``session_id`` and return a handle.

    Idempotency: this function does NOT check whether a container already
    exists for the session. Callers (specifically :mod:`aios.sandbox.registry`)
    are responsible for caching the handle and not calling this twice for
    the same session within a turn.

    Raises :class:`ContainerError` if ``docker run`` fails for any reason
    (image missing, daemon unreachable, resource limits, ...).
    """
    from aios.sandbox.volumes import ensure_workspace_path

    settings = get_settings()
    raw_path = await _load_workspace_path(session_id)
    workspace_path = ensure_workspace_path(raw_path)
    env_config = await _load_environment_config(session_id)
    session_env = await _load_session_env(session_id)

    # Merge environment-level and session-level env vars (session wins).
    merged_env: dict[str, str] = {
        **(env_config.env if env_config and env_config.env else {}),
        **session_env,
    }

    # Determine if the environment uses limited networking.
    networking = env_config.networking if env_config else None
    needs_lockdown = isinstance(networking, LimitedNetworking)

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
    ]

    # Inject merged environment variables into the container.
    for key, value in merged_env.items():
        argv.extend(["--env", f"{key}={value}"])

    # Limited networking needs NET_ADMIN to apply iptables rules.
    if needs_lockdown:
        argv.extend(["--cap-add", "NET_ADMIN"])

    argv.append(settings.docker_image)

    rc, stdout_bytes, stderr_bytes = await _run_docker(argv)
    if rc != 0:
        raise ContainerError(
            f"docker run failed (exit {rc}): "
            f"{stderr_bytes.decode('utf-8', errors='replace').strip()}"
        )

    container_id = stdout_bytes.decode("utf-8").strip()
    if not container_id:
        raise ContainerError("docker run returned an empty container id")

    handle = ContainerHandle(
        session_id=session_id,
        container_id=container_id,
        workspace_path=workspace_path,
    )

    # Install packages while the network is still open.
    await _install_packages(handle, env_config, session_id)

    # Apply iptables lockdown after packages are installed.
    if needs_lockdown and isinstance(networking, LimitedNetworking):
        await _apply_network_lockdown(handle, networking, session_id)

    log.info(
        "sandbox.provisioned",
        session_id=session_id,
        container_id=container_id[:12],
        workspace_path=str(workspace_path),
        networking=networking.type if networking else "unrestricted",
    )

    return handle


async def _install_packages(
    handle: ContainerHandle,
    env_config: EnvironmentConfig | None,
    session_id: str,
) -> None:
    """Install packages from the environment config.

    Failures are logged but don't prevent container use — the model can
    retry or work around missing packages.
    """
    if env_config is None or not env_config.packages:
        return

    packages = env_config.packages

    install_cmds = {
        "apt": "apt-get update -qq && apt-get install -y -qq {}",
        "pip": "pip install -q {}",
        "npm": "npm install -g --silent {}",
        "cargo": "cargo install {}",
        "gem": "gem install {}",
        "go": "go install {}",
    }

    settings = get_settings()
    for manager, cmd_template in install_cmds.items():
        pkg_list = packages.get(manager)
        if not pkg_list:
            continue
        cmd = cmd_template.format(" ".join(pkg_list))
        result = await handle.run_command(
            cmd, timeout_seconds=120, max_output_bytes=settings.bash_max_output_bytes
        )
        if result.exit_code != 0:
            log.warning(
                "sandbox.package_install_failed",
                session_id=session_id,
                manager=manager,
                exit_code=result.exit_code,
                stderr=result.stderr[:500],
            )


# ── network lockdown ──────────────────────────────────────────────────────────


def build_iptables_script(allowed_hosts: set[str]) -> str:
    """Build a shell script that restricts outbound traffic via iptables.

    The script allows: loopback, established connections, DNS (port 53),
    and HTTP/HTTPS (ports 80/443) to the resolved IPs of each allowed
    host. Everything else is dropped.

    Hostnames are validated at the model layer (alphanumerics, dots, hyphens
    only) so embedding them in the script is safe.
    """
    lines = [
        "set -e",
        "",
        "# Flush existing OUTPUT rules",
        "iptables -F OUTPUT",
        "",
        "# Allow loopback",
        "iptables -A OUTPUT -o lo -j ACCEPT",
        "",
        "# Allow established/related connections",
        "iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT",
        "",
        "# Allow DNS (UDP and TCP port 53)",
        "iptables -A OUTPUT -p udp --dport 53 -j ACCEPT",
        "iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT",
    ]

    for host in sorted(allowed_hosts):
        lines.append("")
        lines.append(f"# Allow {host}")
        lines.append(
            f"for ip in $(getent ahosts {host} 2>/dev/null | awk '{{print $1}}' | sort -u); do"
        )
        lines.append('  iptables -A OUTPUT -d "$ip" -p tcp --dport 80 -j ACCEPT')
        lines.append('  iptables -A OUTPUT -d "$ip" -p tcp --dport 443 -j ACCEPT')
        lines.append("done")

    lines.append("")
    lines.append("# Drop everything else")
    lines.append("iptables -P OUTPUT DROP")

    return "\n".join(lines)


async def _apply_network_lockdown(
    handle: ContainerHandle,
    networking: LimitedNetworking,
    session_id: str,
) -> None:
    """Apply iptables rules to restrict outbound traffic.

    Called after package installation so that ``pip install`` etc. can
    reach registries before the lockdown takes effect.  Failures are
    logged as warnings — the container is still usable, but the model
    will see connection errors if it tries to reach blocked hosts.
    """
    allowed: set[str] = set(networking.allowed_hosts)
    if networking.allow_package_managers:
        allowed |= PACKAGE_REGISTRY_HOSTS

    script = build_iptables_script(allowed)
    settings = get_settings()
    result = await handle.run_command(
        script, timeout_seconds=30, max_output_bytes=settings.bash_max_output_bytes
    )

    if result.exit_code != 0:
        log.warning(
            "sandbox.network_lockdown_failed",
            session_id=session_id,
            exit_code=result.exit_code,
            stderr=result.stderr[:500],
        )
    else:
        log.info(
            "sandbox.network_lockdown_applied",
            session_id=session_id,
            allowed_host_count=len(allowed),
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
