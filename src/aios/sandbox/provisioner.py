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

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from aios.config import get_settings

if TYPE_CHECKING:
    from aios.config import Settings
from aios.db import queries
from aios.logging import get_logger
from aios.models.environments import EnvironmentConfig, LimitedNetworking
from aios.models.github_repositories import GithubRepositoryResourceEcho
from aios.models.memory_stores import MemoryStoreResourceEcho
from aios.sandbox.container import (
    ContainerError,
    ContainerHandle,
    mount_snapshot_from_echoes,
    run_subprocess_with_timeout,
)
from aios.sandbox.git_proxy import GitProxy
from aios.sandbox.github_clone import (
    GithubCloneError,
    ensure_cache_clone,
    ensure_session_working_tree,
)

log = get_logger("aios.sandbox.provisioner")

# Labels applied at ``docker run`` time. The worker's orphan reaper lists
# containers with the managed+instance labels at startup and removes any
# whose session_id does not match an active worker lease.
#
# The instance label scopes the reaper to containers belonging to this
# aios deployment — critical when multiple worktrees or deployed instances
# share a Docker daemon, otherwise worker A's startup would kill worker B's
# running sandboxes.
MANAGED_LABEL_KEY = "aios.managed"
MANAGED_LABEL_VALUE = "true"
INSTANCE_LABEL_KEY = "aios.instance_id"
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


# Bound every ``docker`` management call so a stalled daemon can't wedge
# the worker step path (per issue #179 / commit e675ed2).
_DOCKER_CLI_TIMEOUT_S = 30.0


async def _run_docker(
    argv: list[str], *, timeout_s: float = _DOCKER_CLI_TIMEOUT_S
) -> tuple[int, bytes, bytes]:
    """Run a ``docker`` CLI invocation and return (exit_code, stdout, stderr).

    Raises :class:`ContainerError` if the subprocess cannot be launched
    (missing binary, OS error) or if it runs longer than ``timeout_s``
    (stalled daemon). A nonzero exit from docker itself is returned as a
    regular tuple — callers decide whether it's fatal.
    """
    rc, stdout_bytes, stderr_bytes, timed_out = await run_subprocess_with_timeout(
        argv, timeout_s=timeout_s
    )
    if timed_out:
        raise ContainerError(f"docker cli timed out after {timeout_s}s: {' '.join(argv)}")
    return rc, stdout_bytes, stderr_bytes


async def _load_environment_config(session_id: str) -> EnvironmentConfig | None:
    """Load the environment config for a session from the DB.

    Returns ``None`` if the session or environment doesn't exist (shouldn't
    happen in normal flow, but callers handle it gracefully).
    """
    from aios.harness import runtime

    pool = runtime.require_pool()
    async with pool.acquire() as conn:
        return await queries.get_environment_config_for_session(conn, session_id)


async def _load_session_provisioning(session_id: str) -> tuple[str, dict[str, str]]:
    """Load workspace path and env from the session row in one query."""
    from aios.harness import runtime

    pool = runtime.require_pool()
    async with pool.acquire() as conn:
        return await queries.get_session_provisioning(conn, session_id)


async def _materialize_memory_mounts(
    session_id: str,
) -> list[MemoryStoreResourceEcho]:
    """Materialize each attached memory store and return the echo list.

    Returns the per-store metadata used to assemble ``--volume`` flags
    after this call returns. A separate helper (vs inline in
    ``provision_for_session``) keeps the unit-test mocking surface tight —
    tests mock this single function and don't need a live pool.
    """
    from aios.harness import runtime
    from aios.sandbox.memory_mounts import materialize_store_to_host

    pool = runtime.require_pool()
    async with pool.acquire() as conn:
        echoes = await queries.list_session_memory_store_echoes(conn, session_id)
        for echo in echoes:
            await materialize_store_to_host(conn, store_id=echo.memory_store_id)
    return list(echoes)


_PROXY_HOST_ALIAS = "host.docker.internal"


async def _materialize_github_clones(
    session_id: str,
) -> tuple[list[GithubRepositoryResourceEcho], GitProxy | None]:
    """Decrypt tokens, start a per-session :class:`GitProxy`, then run
    the host-side clone for each attached github_repository — rewriting
    each working tree's ``origin`` to the proxy URL so the bind-mounted
    ``.git/config`` inside the sandbox carries no credential.

    Returns the materialized echo list and the proxy (or ``None`` when
    the session has no github_repository attachments).

    Per-repo clone failures are non-fatal — they log + append a
    lifecycle event and drop the repo from the materialized list (its
    working tree won't be bind-mounted). The proxy stays alive for
    sibling repos.
    """
    from aios.harness import runtime
    from aios.sandbox.git_proxy import GitProxy, repo_key
    from aios.services import github_repositories as github_repo_service

    pool = runtime.require_pool()
    crypto_box = runtime.require_crypto_box()

    # Load echoes and decrypt all tokens up front so the proxy can be
    # initialized with the full token map before any clone runs.
    async with pool.acquire() as conn:
        echoes = await queries.list_session_github_repo_echoes(conn, session_id)
        if not echoes:
            return [], None
        echo_tokens: list[tuple[GithubRepositoryResourceEcho, str]] = []
        repos_map: dict[str, str] = {}
        for echo in echoes:
            token = await github_repo_service.get_session_token(
                conn, crypto_box, session_id, echo.id
            )
            echo_tokens.append((echo, token))
            repos_map[repo_key(echo.url)] = token

    proxy = GitProxy(repos_map)
    await proxy.start()

    materialized: list[GithubRepositoryResourceEcho] = []
    failures: list[tuple[GithubRepositoryResourceEcho, GithubCloneError]] = []
    for echo, token in echo_tokens:
        try:
            cache_dir = await ensure_cache_clone(echo.url, token)
            await ensure_session_working_tree(
                session_id=session_id,
                resource_id=echo.id,
                repo_url=echo.url,
                token=token,
                cache_dir=cache_dir,
                proxy_url=proxy.proxy_url(echo.url, host=_PROXY_HOST_ALIAS),
            )
        except GithubCloneError as err:
            failures.append((echo, err))
            continue
        materialized.append(echo)

    # Log + append failure events outside the conn block so we don't
    # hold one connection while acquiring a second from the same pool.
    if failures:
        from aios.services import sessions as sessions_service

        for failed_echo, failure in failures:
            log.warning(
                "sandbox.github_clone_failed",
                session_id=session_id,
                resource_id=failed_echo.id,
                repo_url=failed_echo.url,
                error=str(failure),
            )
            await sessions_service.append_event(
                pool,
                session_id,
                "lifecycle",
                {
                    "event": "github_clone_failed",
                    "resource_id": failed_echo.id,
                    "repo_url": failed_echo.url,
                    "message": str(failure),
                },
            )
    return materialized, proxy


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
    raw_path, session_env = await _load_session_provisioning(session_id)
    workspace_path = ensure_workspace_path(raw_path)
    env_config = await _load_environment_config(session_id)
    memory_echoes = await _materialize_memory_mounts(session_id)
    github_echoes, git_proxy = await _materialize_github_clones(session_id)

    # Once the proxy is running its serve task holds an open port +
    # in-memory token map. Anything that raises before we hand the
    # caller a ContainerHandle (which carries the proxy and would be
    # cleaned up by ``release``) must stop the proxy on the way out,
    # otherwise the port and tokens leak for the worker's lifetime.
    try:
        return await _provision_for_session_inner(
            session_id=session_id,
            settings=settings,
            workspace_path=workspace_path,
            env_config=env_config,
            session_env=session_env,
            memory_echoes=memory_echoes,
            github_echoes=github_echoes,
            git_proxy=git_proxy,
        )
    except BaseException:
        if git_proxy is not None:
            try:
                await git_proxy.stop()
            except Exception as cleanup_err:
                log.warning(
                    "sandbox.git_proxy_cleanup_after_provision_failure",
                    session_id=session_id,
                    error=str(cleanup_err),
                )
        raise


async def _provision_for_session_inner(
    *,
    session_id: str,
    settings: Settings,
    workspace_path: Path,
    env_config: EnvironmentConfig | None,
    session_env: dict[str, str],
    memory_echoes: list[MemoryStoreResourceEcho],
    github_echoes: list[GithubRepositoryResourceEcho],
    git_proxy: GitProxy | None,
) -> ContainerHandle:
    """The body of ``provision_for_session``. Split out so the caller's
    try/except cleanly wraps everything that might leak the proxy."""
    from aios.sandbox.volumes import memory_store_host_dir, session_repo_working_tree_dir

    merged_env: dict[str, str] = {
        **(env_config.env if env_config and env_config.env else {}),
        **session_env,
    }

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
        f"{INSTANCE_LABEL_KEY}={settings.instance_id}",
        "--label",
        f"{SESSION_LABEL_KEY}={session_id}",
        "--network",
        settings.sandbox_network_mode,
        # Keep stdin open so the container doesn't exit on empty stdin.
        "--interactive",
    ]

    # When a github_repository is attached, the container must be able to
    # reach the host-side credential proxy via host.docker.internal.
    # The explicit alias is required on Linux; on Docker Desktop the
    # name auto-resolves but `--add-host` is harmless.
    if git_proxy is not None:
        argv.extend(["--add-host", f"{_PROXY_HOST_ALIAS}:host-gateway"])

    # Bind-mount each attached memory store. ``:ro`` for read_only attaches
    # ensures the kernel rejects writes (bash + tools alike); ``:rw`` is the
    # read_write default. The host source is shared across all attached
    # sessions, so a tool/API write in one session is visible to all.
    for memory_echo in memory_echoes:
        host_dir = memory_store_host_dir(memory_echo.memory_store_id)
        mode = "ro" if memory_echo.access == "read_only" else "rw"
        argv.extend(["--volume", f"{host_dir}:/mnt/memory/{memory_echo.name}:{mode}"])

    for github_echo in github_echoes:
        repo_host_dir = session_repo_working_tree_dir(session_id, github_echo.id)
        argv.extend(["--volume", f"{repo_host_dir}:{github_echo.mount_path}:rw"])

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
        mount_snapshot=mount_snapshot_from_echoes(memory_echoes, github_echoes),
        git_proxy=git_proxy,
    )

    try:
        await _install_packages(handle, env_config, session_id)
        if needs_lockdown and isinstance(networking, LimitedNetworking):
            extra_host_ports: list[tuple[str, int]] = (
                [(_PROXY_HOST_ALIAS, git_proxy.port)] if git_proxy is not None else []
            )
            await _apply_network_lockdown(
                handle, networking, session_id, extra_host_ports=extra_host_ports
            )
    except BaseException:
        # Container is up but we failed to set it up. Tear it down so
        # we don't leak a docker container alongside the proxy.
        await release(handle)
        raise

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


def build_iptables_script(
    allowed_hosts: set[str],
    extra_host_ports: Sequence[tuple[str, int]] = (),
) -> str:
    """Build a shell script that restricts outbound traffic via iptables.

    The script allows: loopback, established connections, DNS (port 53),
    HTTP/HTTPS (ports 80/443) to the resolved IPs of each allowed host,
    and any additional ``(host, port)`` pairs in ``extra_host_ports``.
    Everything else is dropped.

    The extra-host-ports surface exists because the credential proxy
    binds to a non-standard ephemeral port; without it, in-container
    git traffic to the proxy would be dropped by the default policy.

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

    for host, port in extra_host_ports:
        lines.append("")
        lines.append(f"# Allow {host}:{port}")
        lines.append(
            f"for ip in $(getent ahosts {host} 2>/dev/null | awk '{{print $1}}' | sort -u); do"
        )
        lines.append(f'  iptables -A OUTPUT -d "$ip" -p tcp --dport {port} -j ACCEPT')
        lines.append("done")

    lines.append("")
    lines.append("# Drop everything else")
    lines.append("iptables -P OUTPUT DROP")

    return "\n".join(lines)


async def _apply_network_lockdown(
    handle: ContainerHandle,
    networking: LimitedNetworking,
    session_id: str,
    *,
    extra_host_ports: Sequence[tuple[str, int]] = (),
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

    script = build_iptables_script(allowed, extra_host_ports=extra_host_ports)
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
            extra_host_port_count=len(extra_host_ports),
        )


async def release(handle: ContainerHandle) -> None:
    """Stop and remove ``handle``'s container.

    Uses ``docker rm --force`` which sends SIGKILL and removes the
    container in one step. If the container is already gone (daemon
    forgot it, user removed it manually), this is a no-op — we log but
    don't raise. The workspace directory on the host is NOT deleted.

    The per-session credential proxy (if any) is stopped before docker
    rm so its in-memory tokens are released as soon as the session is
    no longer using them.
    """
    if handle.git_proxy is not None:
        try:
            await handle.git_proxy.stop()
        except Exception as err:
            # Best-effort cleanup — never let a stuck proxy block container
            # teardown. The server task is killed regardless on cancel.
            log.warning(
                "sandbox.git_proxy_stop_failed",
                session_id=handle.session_id,
                error=str(err),
            )

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
    """List containers for this aios instance.

    Returns ``(container_id, session_id)`` tuples for containers labelled
    both ``aios.managed=true`` and ``aios.instance_id=<current instance>``.
    ``session_id`` is read from the container's ``aios.session_id`` label;
    containers missing that label are returned with an empty string
    (shouldn't happen, but we're defensive).

    Used by the worker's orphan reaper at startup. The instance filter is
    load-bearing: without it, one worker's reaper would kill containers
    belonging to a concurrent worker on the same Docker daemon.
    """
    settings = get_settings()
    ps_argv = [
        "docker",
        "ps",
        "--quiet",
        "--filter",
        f"label={MANAGED_LABEL_KEY}={MANAGED_LABEL_VALUE}",
        "--filter",
        f"label={INSTANCE_LABEL_KEY}={settings.instance_id}",
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
