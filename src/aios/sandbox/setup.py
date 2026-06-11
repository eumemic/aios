"""Backend-agnostic post-create setup for a sandbox.

After the registry calls ``backend.create(spec)`` and gets a
:class:`SandboxHandle` back, three setup steps run inside the sandbox to
bring it to a usable state:

1. :func:`install_egress_ca` — installs the worker's egress-CA cert
   into the sandbox trust store (issue #875).
2. :func:`install_packages` — runs the apt/pip/npm/cargo/gem/go
   commands the environment config asked for.
3. :func:`apply_network_lockdown` — applies (and read-back verifies) the
   iptables egress rules when the network policy is :class:`Limited`, from
   an ephemeral operator-image sidecar joined to the sandbox's netns (§5.8)
   — NOT from the tenant-writable sandbox filesystem.

:data:`WORKSPACE_RUNTIME_ENV` carries the absolute system PATH that spec
building merges into every sandbox's environment (a load-bearing constant,
not a setup step — see its own docstring).

The first two steps call ``await backend.exec(handle, ...)`` rather than
touching Docker directly, so they work uniformly across backends; the
lockdown goes through :func:`SandboxBackend.run_netns_sidecar` (the sandbox
holds no ``NET_ADMIN``, so it cannot apply or subvert its own lockdown).

The first two steps are best-effort enrichments — a nonzero exit is
logged, never raised; the model can retry or work around missing tooling.
:func:`apply_network_lockdown` is different: when the policy is
:class:`Limited` it is a **security gate**, not an enrichment. A
:class:`Limited` sandbox whose iptables lockdown didn't apply is wide
open to the network, which silently violates the operator's intent (and
is especially dangerous combined with the per-environment image override
in #724 — a tenant-supplied image with a stripped-down ``iptables``/
``getent`` would otherwise downgrade to unrestricted networking without
anyone noticing). So that step **fails closed**: if the lockdown command
exits nonzero, or the backend exec itself errors, it raises
:class:`SandboxBackendError`, which the registry turns into a
sandbox teardown + aborted provision rather than handing back a sandbox
that can reach the whole internet.

This module is the second seam (alongside ``backends.base``) that keeps
the registry and the orchestrator backend-agnostic.
"""

from __future__ import annotations

from collections.abc import Sequence

from aios.config import get_settings
from aios.logging import get_logger
from aios.models.environments import EnvironmentConfig, LimitedNetworking
from aios.sandbox.backends.base import SandboxBackend, SandboxBackendError, SandboxHandle
from aios.sandbox.egress_ca import CA_CERT_SANDBOX_PATH, get_egress_ca

log = get_logger("aios.sandbox.setup")


# Hardcoded absolute system PATH because docker --env doesn't expand $PATH;
# the value matches the python:3.13-slim-bookworm image's default. The
# snapshot-resume/flatten path re-injects env via ``docker run --env`` with
# no config PATH, so this must be set explicitly or the keepalive CMD
# ``["tail","-f","/dev/null"]`` can't resolve ``tail`` (SEV-1 #935).
WORKSPACE_RUNTIME_ENV: dict[str, str] = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
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


async def install_egress_ca(backend: SandboxBackend, handle: SandboxHandle) -> None:
    """Install the worker's egress-CA cert into the sandbox trust store.

    Writes the PEM into the Debian drop-in directory and regenerates the
    aggregate bundle, so OpenSSL-based clients and Node (via the
    ``TRUST_STORE_ENV`` vars baked into the spec) trust leaf certs the
    secret-egress proxy will present for allowlisted hosts.

    A nonzero exit is logged, not raised, and doesn't fail the provision
    (a backend exec that itself errors still propagates, same as every
    sibling step) — until the egress proxy terminates TLS, a missing CA
    costs nothing, and after that it fails safe (the sandbox refuses the
    proxy's leaf rather than trusting anything extra). Revisit the
    posture when env-var credentials are attached (#876): a silently
    missing CA then turns into in-sandbox TLS verification failures on
    exactly the allowlisted hosts.

    The ``&&`` chain keeps the exit code all-or-nothing so a partial
    install (drop-in written, bundle not regenerated — Node would trust
    the CA while curl/python don't) still trips the warning. ``printf
    '%s'`` is load-bearing: the PEM starts with ``-----BEGIN``, which
    bash's printf would otherwise parse as an (invalid) option string.
    Single-quoting the PEM is safe because cryptography's PEM output is
    strictly base64 alphabet plus dashes/newlines — never a quote.
    """
    cert_pem = get_egress_ca().cert_pem
    cmd = (
        f"mkdir -p {CA_CERT_SANDBOX_PATH.rsplit('/', 1)[0]} && "
        f"printf '%s' '{cert_pem}' > {CA_CERT_SANDBOX_PATH} && "
        "update-ca-certificates"
    )
    settings = get_settings()
    result = await backend.exec(
        handle, cmd, timeout_seconds=60, max_output_bytes=settings.bash_max_output_bytes
    )
    if result.exit_code != 0:
        log.warning(
            "sandbox.egress_ca_install_failed",
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
    *,
    dnat_hosts: Sequence[str] = (),
    dnat_target: tuple[str, int] | None = None,
) -> str:
    """Build a shell script that restricts outbound traffic via iptables.

    The script allows: loopback, established connections, DNS (port 53),
    HTTP/HTTPS (ports 80/443) to the resolved IPs of each allowed host,
    and any additional ``(host, port)`` pairs in ``extra_host_ports``.
    Everything else is dropped.

    The extra-host-ports surface exists because the credential proxy
    binds to a non-standard ephemeral port; without it, in-sandbox
    git traffic to the proxy would be dropped by the default policy.

    When ``dnat_target`` is supplied alongside a non-empty ``dnat_hosts``,
    a nat-table OUTPUT section rewrites each credential host's resolved
    IPs on **dport 443 only** to the proxy endpoint (#878). The proxy
    alias is resolved to ``$PROXY_IP`` exactly ONCE at sidecar runtime
    (iptables ``--to-destination`` needs an IP, not a DNS name) and the
    whole nat block is guarded by ``if [ -n "$PROXY_IP" ]`` so a
    proxy-alias DNS miss emits no malformed rule — fail-closed: with no
    DNAT, the credential host's :443 still falls through to the filter
    table, which does NOT ACCEPT it unless the host is also in
    ``allowed_hosts``, so a proxy-resolution miss drops the egress rather
    than leaking the placeholder un-proxied. ``dnat_target`` of ``None``
    (the default) emits NO nat rules, preserving every existing caller.

    Hostnames are validated at the model layer (alphanumerics, dots, hyphens
    only) so embedding them in the script is safe; ``proxy_port`` is an int.
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

    if dnat_target is not None and dnat_hosts:
        proxy_alias, proxy_port = dnat_target
        lines.append("")
        lines.append("# Route credential-host HTTPS through the secret-egress proxy (#878)")
        # Resolve the proxy alias to an IP ONCE — iptables --to-destination
        # needs an IP, not a DNS name. The block is guarded on a non-empty
        # $PROXY_IP, so a proxy-alias DNS miss emits no malformed ":<port>"
        # rule; the credential host's :443 then falls through to the filter
        # table. If the host is not independently allow-listed on :443 it hits
        # the DROP policy — fail-closed, no un-proxied placeholder leak. Under
        # #879's `cred ⊆ env` the host is also in allowed_hosts (filter-
        # ACCEPTed), so keeping that overlap closed on a proxy miss is #879's
        # job, not this loop's. (In practice the alias is the load-bearing
        # WORKER_NETWORK_ALIAS, so a miss already means a non-functional
        # sandbox.)
        lines.append(
            f"PROXY_IP=$(getent ahosts {proxy_alias} 2>/dev/null "
            "| awk '{print $1}' | sort -u | head -n1)"
        )
        lines.append('if [ -n "$PROXY_IP" ]; then')
        for host in sorted(dnat_hosts):
            lines.append(
                f"  for ip in $(getent ahosts {host} 2>/dev/null "
                "| awk '{print $1}' | sort -u); do"
            )
            lines.append(
                '    iptables -t nat -A OUTPUT -d "$ip" -p tcp --dport 443 '
                f'-j DNAT --to-destination "$PROXY_IP:{proxy_port}"'
            )
            lines.append("  done")
        lines.append("fi")

    lines.append("")
    lines.append("# Drop everything else")
    lines.append("iptables -P OUTPUT DROP")

    return "\n".join(lines)


# Docker's embedded DNS, served inside every user-defined-network netns (the
# sandbox runs on the ``aios-sandbox`` user-defined bridge). The lockdown
# sidecar joins that netns but inherits the operator image's (typically empty)
# ``/etc/resolv.conf`` — Docker does NOT manage resolv.conf for a
# netns-joining container — so the sidecar script points itself at the
# embedded resolver before ``getent`` resolves the allowed hosts. A DNS miss
# fails CLOSED (the host gets no ACCEPT rule → blocked), never a bypass.
_EMBEDDED_DNS_ADDRESS = "127.0.0.11"

# Read-back assertion that the default OUTPUT policy is DROP — proves the
# lockdown actually took effect in the shared netns, not just that the apply
# script exited 0.
_LOCKDOWN_VERIFY_SCRIPT = "iptables -S OUTPUT | grep -qx -- '-P OUTPUT DROP'"


async def apply_network_lockdown(
    backend: SandboxBackend,
    handle: SandboxHandle,
    networking: LimitedNetworking,
    *,
    extra_host_ports: Sequence[tuple[str, int]] = (),
    dnat_hosts: Sequence[str] = (),
    dnat_target: tuple[str, int] | None = None,
) -> None:
    """Apply + verify iptables egress rules via an ephemeral operator-image sidecar.

    Called after package installation so ``pip install`` etc. can reach
    registries before the lockdown takes effect.

    ``dnat_hosts`` + ``dnat_target`` are threaded into
    :func:`build_iptables_script` to DNAT credential-host :443 egress through
    the secret-egress proxy (#878). The read-back verify reads the **filter**
    table (``iptables -S OUTPUT``) only, so the added nat-table OUTPUT rules
    don't affect it.

    **Off the tenant-writable filesystem (§5.8).** Under durable persistence,
    running the lockdown *inside* the sandbox (its own ``iptables``/``getent``)
    was a bypass: a tenant could replace ``/usr/sbin/iptables`` with ``exit 0``
    in an Unrestricted session, persist it in the snapshot, and have the
    fail-closed gate trust the poisoned binary's exit 0 when the environment
    later flipped to Limited. So the lockdown is applied from an **ephemeral
    sidecar** that joins the sandbox's netns but executes the *operator-trusted*
    image's binaries (:func:`SandboxBackend.run_netns_sidecar`), and the sandbox
    holds no ``NET_ADMIN`` — root-in-sandbox can no longer touch netfilter at
    all. This also closes the pre-existing ``iptables -F your own lockdown``
    hole.

    **Fails closed.** A Limited policy whose apply OR read-back verification
    fails (sidecar errors, nonzero exit, or ``OUTPUT`` policy not ``DROP``)
    raises :class:`SandboxBackendError`; the caller
    (:meth:`SandboxRegistry._provision`) tears the sandbox down and aborts the
    provision rather than handing back an open box.
    """
    allowed: set[str] = set(networking.allowed_hosts)
    if networking.allow_package_managers:
        allowed |= PACKAGE_REGISTRY_HOSTS

    iptables_script = build_iptables_script(
        allowed,
        extra_host_ports=extra_host_ports,
        dnat_hosts=dnat_hosts,
        dnat_target=dnat_target,
    )
    # Point the sidecar at the netns's embedded resolver before getent runs.
    apply_script = (
        f"printf 'nameserver {_EMBEDDED_DNS_ADDRESS}\\n' > /etc/resolv.conf 2>/dev/null || true\n"
        f"{iptables_script}"
    )
    settings = get_settings()

    try:
        result = await backend.run_netns_sidecar(
            handle.sandbox_id,
            image=settings.docker_image,
            script=apply_script,
            timeout_seconds=30,
            max_output_bytes=settings.bash_max_output_bytes,
        )
    except SandboxBackendError:
        # Don't swallow an infra failure into a wide-open sandbox: a Limited
        # policy whose lockdown couldn't even run must fail the provision.
        log.warning("sandbox.network_lockdown_sidecar_error", session_id=handle.session_id)
        raise

    if result.exit_code != 0:
        log.warning(
            "sandbox.network_lockdown_failed",
            session_id=handle.session_id,
            exit_code=result.exit_code,
            stderr=result.stderr[:500],
        )
        raise SandboxBackendError(
            f"network lockdown failed (exit {result.exit_code}) for session "
            f"{handle.session_id}; refusing to run a Limited sandbox with "
            f"unrestricted networking"
        )

    # Read-back verify the DROP policy actually landed in the shared netns.
    try:
        verify = await backend.run_netns_sidecar(
            handle.sandbox_id,
            image=settings.docker_image,
            script=_LOCKDOWN_VERIFY_SCRIPT,
            timeout_seconds=15,
            max_output_bytes=settings.bash_max_output_bytes,
        )
    except SandboxBackendError:
        log.warning("sandbox.network_lockdown_verify_error", session_id=handle.session_id)
        raise
    if verify.exit_code != 0:
        log.warning(
            "sandbox.network_lockdown_verify_failed",
            session_id=handle.session_id,
            exit_code=verify.exit_code,
        )
        raise SandboxBackendError(
            f"network lockdown verification failed for session {handle.session_id}: "
            "OUTPUT policy is not DROP after apply; refusing to run a Limited "
            "sandbox with unverified networking"
        )

    log.info(
        "sandbox.network_lockdown_applied",
        session_id=handle.session_id,
        allowed_host_count=len(allowed),
        extra_host_port_count=len(extra_host_ports),
        dnat_host_count=len(dnat_hosts),
    )
