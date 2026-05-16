"""Backend-agnostic post-create setup for a sandbox.

After the registry calls ``backend.create(spec)`` and gets a
:class:`SandboxHandle` back, three setup steps run inside the sandbox to
bring it to a usable state:

1. :func:`ensure_workspace_runtime_dirs` — creates ``/workspace/.venv``
   and ``/workspace/.npm`` (the persistent install targets that survive
   idle release; see issue #227).
2. :func:`install_packages` — runs the apt/pip/npm/cargo/gem/go
   commands the environment config asked for.
3. :func:`apply_network_lockdown` — installs the iptables script that
   restricts outbound traffic when the network policy is
   :class:`Limited`.

Each step calls ``await backend.exec(handle, ...)`` rather than touching
Docker directly, so they work uniformly across backends. Failures are
logged but never raised — these are best-effort enrichments, not
correctness gates. The model can retry or work around missing tooling.

This module is the second seam (alongside ``backends.base``) that keeps
the registry and the orchestrator backend-agnostic.
"""

from __future__ import annotations

from collections.abc import Sequence

from aios.config import get_settings
from aios.logging import get_logger
from aios.models.environments import EnvironmentConfig, LimitedNetworking
from aios.sandbox.backends.base import SandboxBackend, SandboxHandle

log = get_logger("aios.sandbox.setup")


# Bind-mounted ``/workspace`` is the only path that survives idle release;
# pinning pip/npm installs into these subdirs lets long-lived sessions
# stop re-installing pdfminer / jq-cli-wrapper / etc. on every cold
# sandbox.  See issue #227 for the trade-offs vs PIP_TARGET / project-
# local node_modules / derived images.
WORKSPACE_VENV = "/workspace/.venv"
WORKSPACE_NPM = "/workspace/.npm"

# Environment variables the sandbox image needs to find the persistent
# venv/npm installs. Spec building merges these into ``SandboxSpec.environment``.
WORKSPACE_RUNTIME_ENV: dict[str, str] = {
    "VIRTUAL_ENV": WORKSPACE_VENV,
    "NPM_CONFIG_PREFIX": WORKSPACE_NPM,
    "NODE_PATH": f"{WORKSPACE_NPM}/lib/node_modules",
    # Hardcoded system PATH because docker --env doesn't expand $PATH;
    # the suffix matches the python:3.13-slim-bookworm image's default.
    "PATH": (
        f"{WORKSPACE_VENV}/bin:{WORKSPACE_NPM}/bin:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    ),
}


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


async def ensure_workspace_runtime_dirs(backend: SandboxBackend, handle: SandboxHandle) -> None:
    """Idempotently create ``/workspace/.venv`` and ``/workspace/.npm``.

    Both live under the workspace bind mount so they survive idle
    release; the venv pins ``pip install`` into a persistent
    ``site-packages``, and the npm prefix pins ``npm install -g``.  The
    ``[ -e .venv/bin/python ]`` guard makes venv creation a no-op on
    every cold provision after the first.

    Failures are logged but don't fail the provision — the model can
    still operate without the persistence layer (it just falls back to
    re-installing tools after every cold release).
    """
    settings = get_settings()
    cmd = (
        f"mkdir -p {WORKSPACE_NPM}/lib {WORKSPACE_NPM}/bin && "
        f"([ -e {WORKSPACE_VENV}/bin/python ] || python3 -m venv {WORKSPACE_VENV})"
    )
    result = await backend.exec(
        handle, cmd, timeout_seconds=60, max_output_bytes=settings.bash_max_output_bytes
    )
    if result.exit_code != 0:
        log.warning(
            "sandbox.workspace_runtime_dirs_setup_failed",
            session_id=handle.session_id,
            exit_code=result.exit_code,
            stderr=result.stderr[:500],
        )


async def install_packages(
    backend: SandboxBackend,
    handle: SandboxHandle,
    env_config: EnvironmentConfig | None,
) -> None:
    """Install packages from the environment config.

    Failures are logged but don't prevent sandbox use — the model can
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
        result = await backend.exec(
            handle, cmd, timeout_seconds=120, max_output_bytes=settings.bash_max_output_bytes
        )
        if result.exit_code != 0:
            log.warning(
                "sandbox.package_install_failed",
                session_id=handle.session_id,
                manager=manager,
                exit_code=result.exit_code,
                stderr=result.stderr[:500],
            )


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
    binds to a non-standard ephemeral port; without it, in-sandbox
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


async def apply_network_lockdown(
    backend: SandboxBackend,
    handle: SandboxHandle,
    networking: LimitedNetworking,
    *,
    extra_host_ports: Sequence[tuple[str, int]] = (),
) -> None:
    """Apply iptables rules to restrict outbound traffic.

    Called after package installation so that ``pip install`` etc. can
    reach registries before the lockdown takes effect.  Failures are
    logged as warnings — the sandbox is still usable, but the model
    will see connection errors if it tries to reach blocked hosts.
    """
    allowed: set[str] = set(networking.allowed_hosts)
    if networking.allow_package_managers:
        allowed |= PACKAGE_REGISTRY_HOSTS

    script = build_iptables_script(allowed, extra_host_ports=extra_host_ports)
    settings = get_settings()
    result = await backend.exec(
        handle, script, timeout_seconds=30, max_output_bytes=settings.bash_max_output_bytes
    )

    if result.exit_code != 0:
        log.warning(
            "sandbox.network_lockdown_failed",
            session_id=handle.session_id,
            exit_code=result.exit_code,
            stderr=result.stderr[:500],
        )
    else:
        log.info(
            "sandbox.network_lockdown_applied",
            session_id=handle.session_id,
            allowed_host_count=len(allowed),
            extra_host_port_count=len(extra_host_ports),
        )
