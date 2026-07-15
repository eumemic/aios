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
from aios.sandbox.env_keys import PATH_ENV_KEY

log = get_logger("aios.sandbox.setup")


# Hardcoded absolute system PATH because docker --env doesn't expand $PATH;
# the value matches the python:3.13-slim-bookworm image's default. The
# snapshot-resume/flatten path re-injects env via ``docker run --env`` with
# no config PATH, so this must be set explicitly or the keepalive CMD
# ``["tail","-f","/dev/null"]`` can't resolve ``tail`` (SEV-1 #935).
WORKSPACE_RUNTIME_ENV: dict[str, str] = {
    PATH_ENV_KEY: "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
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
            owner_id=handle.owner_id,
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
                owner_id=handle.owner_id,
                manager=manager,
                exit_code=result.exit_code,
                stderr=result.stderr[:500],
            )


# Pick the legacy netfilter backend when it's available (#1022). gVisor's
# netstack (``runsc``) implements the *legacy* netfilter ABI, NOT nftables,
# but debian/ubuntu images default the ``iptables`` command to the nft
# backend via update-alternatives — so a bare ``iptables`` call inside a
# runsc netns fails with ``Failed to initialize nft: Protocol not supported``
# and the fail-closed gate refuses to provision the sandbox. The legacy binary
# ships in debian's ``iptables`` package as the ``iptables-legacy`` alternative,
# so we always prefer it when present and fall back to ``iptables`` on runc
# hosts whose (custom) image lacks it. Both the apply and the read-back verify
# scripts run this same preamble so they agree on which backend's table holds
# the rules — selecting different backends would let the verify read an empty
# table while the DROP policy sits in the other.
_IPTABLES_BACKEND_SELECT = (
    "if command -v iptables-legacy >/dev/null 2>&1; then IPT=iptables-legacy; else IPT=iptables; fi"
)


# Same legacy-vs-nft backend selection as ``_IPTABLES_BACKEND_SELECT`` but for
# the IPv6 ``ip6tables`` command (#1207). gVisor's netstack (``runsc``)
# implements the *legacy* netfilter ABI, so a bare ``ip6tables`` would fail with
# ``Failed to initialize nft: Protocol not supported`` and — under ``set -e`` —
# abort the entire lockdown apply, failing every Limited provision closed-noisily
# (a self-inflicted outage). We always prefer ``ip6tables-legacy`` when present
# and fall back to ``ip6tables`` on runc hosts whose image lacks it. Debian's
# ``iptables`` package ships BOTH the v4 and v6 legacy alternatives, so the
# operator sidecar image (settings.docker_image) that already carries
# ``iptables-legacy`` for the v4 path carries ``ip6tables-legacy`` too.
_IP6TABLES_BACKEND_SELECT = (
    "if command -v ip6tables-legacy >/dev/null 2>&1; then IP6T=ip6tables-legacy; "
    "else IP6T=ip6tables; fi"
)


# Belt-and-suspenders IPv6 egress denial (#1207). The IPv4-only egress lockdown
# rests on the ``aios-sandbox`` network being created without ``--ipv6`` so no
# v6 route exists — an implicit, undocumented invariant. The moment a v6 route
# appears (network recreated with ``--ipv6``, or a Docker default flips), the
# IPv4-only ``-P OUTPUT DROP`` is silently bypassable over IPv6 (fail-open).
# This block makes v6 egress impossible *by construction*: flush the v6 OUTPUT
# chain, allow only loopback (so any in-netns v6 localhost/DNS still works), and
# set the default OUTPUT policy to DROP — mirroring the v4 DROP. It is emitted
# only on the Limited lockdown path (total-egress-denial intent); the
# Unrestricted DNAT-only path deliberately leaves all egress open.
#
# This is the LOAD-BEARING prod protection for the IPv6 gap: it is applied
# per-session in the sidecar netns regardless of how the (already-running, never
# recreated — constraint #4) prod network was created. The ``--ipv6=false``
# network-create flag is the weakest of the three changes — redundant against
# the current Docker default and inert for the live network — so the real
# defense is this per-session DROP.
#
# The whole block is GUARDED on the v6 ``filter`` table being initializable
# (``"$IP6T" -S OUTPUT`` succeeds). On hosts where the ``ip6_tables`` kernel
# module is not loaded — common on CI runners and any IPv6-disabled host —
# ``ip6tables`` aborts with ``can't initialize ip6tables table 'filter': Table
# does not exist (do you need to insmod?)``. Under ``set -e`` that would abort
# the entire lockdown apply and fail every Limited provision closed-noisily — a
# self-inflicted outage triggered by the absence of the very v6 stack we are
# trying to lock down. But that absence is itself the security property: with no
# v6 ``filter`` table there is no v6 netfilter path to leak through, so skipping
# the DROP is safe. When the table IS present (a v6 route/stack exists — the
# exact case the DROP defends), the flush/loopback/DROP run and any failure
# there is a real error. We deliberately do NOT ``modprobe ip6_tables`` (the
# sidecar holds no module-load capability and forcing the module on just to drop
# would re-introduce a v6 surface where none existed).
_IP6TABLES_LOCKDOWN_LINES = (
    "",
    "# Belt-and-suspenders: deny ALL IPv6 egress (#1207). The IPv4 -P OUTPUT DROP",
    "# above is iptables-only; without this an IPv6 route would bypass it.",
    _IP6TABLES_BACKEND_SELECT,
    "# Guard on the v6 filter table being initializable: if ip6_tables is not",
    "# loaded (no v6 netfilter path to leak through) skip rather than abort under",
    "# set -e; when it IS present the DROP below is enforced and verified.",
    'if "$IP6T" -S OUTPUT >/dev/null 2>&1; then',
    '  "$IP6T" -F OUTPUT',
    "  # Allow v6 loopback so in-netns localhost/DNS still works; deny everything else.",
    '  "$IP6T" -A OUTPUT -o lo -j ACCEPT',
    '  "$IP6T" -P OUTPUT DROP',
    "else",
    '  echo "ip6tables filter table unavailable (ip6_tables not loaded); '
    'no IPv6 egress path to lock down — skipping v6 DROP" >&2',
    "fi",
)


# Emitted shell helper that resolves a hostname to its **IPv4 addresses only**,
# one per line. Centralizes the IPv4-only resolution shared by every host
# lookup in the lockdown scripts (the allowed-host loops, the extra-host-ports
# loop, the credential-host DNAT loop, and the proxy-alias lookup), so the
# IPv4-only invariant lives in exactly one place (#978).
#
# Why IPv4-only: every rule emitted by these scripts is an IPv4 ``iptables``
# command, and the secret-egress proxy binds the IPv4 ``WORKER_NETWORK_ALIAS``
# (it cannot intercept IPv6). ``getent ahosts`` returns BOTH A and AAAA
# records; feeding an AAAA literal to an IPv4-only ``iptables -d`` would error,
# and under ``set -e`` abort the whole apply. The sandbox network is currently
# IPv4-only so this is latent today, but if an IPv6-capable network is ever
# enabled it would break Limited networking on every IPv6-resolving host. Using
# ``getent ahostsv4`` makes only A records reach the rules; any AAAA/IPv6
# egress is simply dropped by the default policy (fail-closed) — which is the
# correct semantics for credential hosts too (IPv6 must never be sent
# un-proxied).
#
# A resolution miss prints nothing (the caller's ``for`` loop / ``$()`` capture
# sees no IPs), so the host gets no rule — fail-closed, never a bypass.
_RESOLVE_IPV4_FN = (
    "resolve_ipv4() { getent ahostsv4 \"$1\" 2>/dev/null | awk '{print $1}' | sort -u; }"
)


def _nat_dnat_lines(dnat_hosts: Sequence[str], dnat_target: tuple[str, int]) -> list[str]:
    """The nat-OUTPUT DNAT block: credential-host :443 → secret-egress proxy.

    Shared by the Limited lockdown script (:func:`build_iptables_script`) and
    the Unrestricted DNAT-only script (:func:`build_secret_egress_dnat_script`)
    so the rule shape — and the ``$PROXY_IP``-miss fail-open-to-placeholder
    guard — lives in exactly one place (#1153).

    The proxy alias is resolved to ``$PROXY_IP`` exactly ONCE at sidecar
    runtime (iptables ``--to-destination`` needs an IP, not a DNS name) and the
    whole block is guarded by ``if [ -n "$PROXY_IP" ]`` so a proxy-alias DNS
    miss emits no malformed rule. On such a miss the behavior is
    fail-open-to-placeholder, NOT fail-closed: the placeholder reaches the real
    upstream (an authentication failure, never a secret leak — the real secret
    never enters the container). See :func:`build_iptables_script` for the full
    rationale.

    Callers only invoke this when ``dnat_hosts`` is non-empty and a
    ``dnat_target`` is supplied.
    """
    proxy_alias, proxy_port = dnat_target
    lines = [
        "",
        "# Route credential-host HTTPS through the secret-egress proxy (#878)",
        # Resolve the proxy alias to an IP ONCE — iptables --to-destination
        # needs an IP, not a DNS name. The block is guarded on a non-empty
        # $PROXY_IP, so a proxy-alias DNS miss emits no malformed ":<port>"
        # rule. NOTE the resulting behavior: under #879's `cred ⊆ env` gate
        # (Limited) the credential host IS always in allowed_hosts and therefore
        # filter-ACCEPTed on :443 — so on a proxy miss, traffic flows DIRECTLY
        # to the real upstream carrying the opaque placeholder. That is
        # fail-open-to-placeholder (auth failures), never a secret leak: the
        # real secret never enters the container. (In practice the alias is the
        # load-bearing WORKER_NETWORK_ALIAS, so a miss already means a
        # non-functional sandbox.)
        f"PROXY_IP=$(resolve_ipv4 {proxy_alias} | head -n1)",
        'if [ -n "$PROXY_IP" ]; then',
    ]
    for host in sorted(dnat_hosts):
        lines.append(f"  for ip in $(resolve_ipv4 {host}); do")
        lines.append(
            '    "$IPT" -t nat -A OUTPUT -d "$ip" -p tcp --dport 443 '
            f'-j DNAT --to-destination "$PROXY_IP:{proxy_port}"'
        )
        lines.append("  done")
    lines.append("fi")
    return lines


def build_egress_resolve_script(hosts: Sequence[str] | set[str]) -> str:
    """Resolve refresh hosts inside the sandbox netns, one machine-readable row per IP."""
    lines = ["set -e", _RESOLVE_IPV4_FN]
    for host in sorted(set(hosts)):
        lines.append(f"for ip in $(resolve_ipv4 {host}); do printf '%s %s\\n' {host} \"$ip\"; done")
    return _RESOLV_PREAMBLE + "\n".join(lines)


def build_egress_refresh_script(
    *,
    old_ips: dict[str, set[str]],
    new_ips: dict[str, set[str]],
    credential_hosts: set[str],
    limited_hosts: set[str],
    dnat_target: tuple[str, int],
) -> str:
    """Atomically refresh generated egress rules without flushing Docker's tables.

    New rules are appended before superseded rules are deleted.  Every delete is
    the exact inverse of a rule this subsystem owns; no table restore/flush can
    disturb Docker's embedded-DNS chains or unrelated policy.

    Every operation is **idempotent** so a retried old→new delta never wedges
    under ``set -e`` and never accumulates duplicate rules: adds are guarded by
    an ``iptables -C`` existence check (append only when absent), and deletes
    tolerate an already-absent rule (``-D … || true``). A genuine ``-A``
    failure still aborts the script loudly (nonzero exit) so the caller keeps
    its last-good ``pinned`` state and retries the same delta next tick.
    """

    def _add(table_flag: str, rule: str) -> str:
        # Append-if-absent: -C exits 0 when the rule exists (skip the -A),
        # nonzero otherwise (2>/dev/null silences its "Bad rule" noise).
        return f'"$IPT"{table_flag} -C OUTPUT {rule} 2>/dev/null || "$IPT"{table_flag} -A OUTPUT {rule}'

    def _delete(table_flag: str, rule: str) -> str:
        # Delete-if-present: an already-absent rule must never abort the
        # script (set -e) — the delta may be a retry of a partial apply.
        return f'"$IPT"{table_flag} -D OUTPUT {rule} 2>/dev/null || true'

    proxy_ip, proxy_port = dnat_target
    # The rule tail after ``-d <ip>`` — byte-identical to the provision-time
    # DNAT shape so -C/-D match the installed rules exactly.
    dnat_tail = f"-p tcp --dport 443 -j DNAT --to-destination {proxy_ip}:{proxy_port}"
    lines = ["set -e", _IPTABLES_BACKEND_SELECT]
    for host in sorted(new_ips):
        added = new_ips[host] - old_ips.get(host, set())
        for ip in sorted(added):
            if host in limited_hosts:
                lines.append(_add("", f"-d {ip} -p tcp --dport 80 -j ACCEPT"))
                lines.append(_add("", f"-d {ip} -p tcp --dport 443 -j ACCEPT"))
            if host in credential_hosts:
                lines.append(_add(" -t nat", f"-d {ip} {dnat_tail}"))
    for host in sorted(old_ips):
        removed = old_ips[host] - new_ips.get(host, set())
        for ip in sorted(removed):
            if host in credential_hosts:
                lines.append(_delete(" -t nat", f"-d {ip} {dnat_tail}"))
            if host in limited_hosts:
                lines.append(_delete("", f"-d {ip} -p tcp --dport 80 -j ACCEPT"))
                lines.append(_delete("", f"-d {ip} -p tcp --dport 443 -j ACCEPT"))
    return "\n".join(lines)


def build_egress_dump_script() -> str:
    """Dump the netns's live OUTPUT rules (filter + nat) with section markers.

    Run at provision time, AFTER the apply sidecar, so the refresh state's
    ``pinned`` set can be seeded from the rules **actually installed** rather
    than from a second DNS resolve that may diverge from the apply script's
    own in-script ``resolve_ipv4`` (short-TTL/round-robin DNS). Read-only —
    never mutates the tables.
    """
    return "\n".join(
        [
            "set -e",
            _IPTABLES_BACKEND_SELECT,
            "echo '=filter='",
            '"$IPT" -S OUTPUT',
            "echo '=nat='",
            '"$IPT" -t nat -S OUTPUT',
        ]
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

    As a belt-and-suspenders measure (#1207) the script ALSO denies all IPv6
    egress: it flushes the ``ip6tables`` OUTPUT chain, allows v6 loopback, and
    sets ``-P OUTPUT DROP``. The IPv4 ``iptables`` DROP is IPv4-only, so without
    this an IPv6 route appearing on the sandbox network (currently created
    without ``--ipv6``) would silently bypass the lockdown over v6. The v6 path
    uses the same legacy-vs-nft backend selection as the v4 path so a bare
    ``ip6tables`` never aborts the apply under runsc's legacy-only netstack.

    The extra-host-ports surface exists because the credential proxy
    binds to a non-standard ephemeral port; without it, in-sandbox
    git traffic to the proxy would be dropped by the default policy.

    When ``dnat_target`` is supplied alongside a non-empty ``dnat_hosts``,
    a nat-table OUTPUT section rewrites each credential host's resolved
    IPs on **dport 443 only** to the proxy endpoint (#878). The proxy
    alias is resolved to ``$PROXY_IP`` exactly ONCE at sidecar runtime
    (iptables ``--to-destination`` needs an IP, not a DNS name) and the
    whole nat block is guarded by ``if [ -n "$PROXY_IP" ]`` so a
    proxy-alias DNS miss emits no malformed rule. On such a miss the
    behavior is fail-open-to-placeholder, NOT fail-closed: dnat_hosts ⊆
    networking.allowed_hosts (enforced by the #879 provision gate), so the
    credential host's :443 is still filter-ACCEPTed and traffic reaches
    the real upstream carrying the opaque placeholder — an authentication
    failure, never a secret leak (the real secret never enters the
    container). A non-functional credentialed sandbox is acceptable here;
    a WORKER_NETWORK_ALIAS miss already implies the proxy infrastructure
    is broken. ``dnat_target`` of ``None`` (the default) emits NO nat
    rules, preserving every existing caller.

    Hostnames are validated at the model layer (alphanumerics, dots, hyphens
    only) so embedding them in the script is safe; ``proxy_port`` is an int.
    """
    lines = [
        "set -e",
        "",
        _IPTABLES_BACKEND_SELECT,
        "",
        "# Resolve hosts IPv4-only so AAAA records never reach the IPv4 rules (#978)",
        _RESOLVE_IPV4_FN,
        "",
        "# Flush existing OUTPUT rules (filter + nat) for idempotent re-apply",
        '"$IPT" -F OUTPUT',
        '"$IPT" -t nat -F OUTPUT',
        "",
        "# Allow loopback",
        '"$IPT" -A OUTPUT -o lo -j ACCEPT',
        "",
        "# Allow established/related connections",
        '"$IPT" -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT',
        "",
        "# Allow DNS (UDP and TCP port 53)",
        '"$IPT" -A OUTPUT -p udp --dport 53 -j ACCEPT',
        '"$IPT" -A OUTPUT -p tcp --dport 53 -j ACCEPT',
    ]

    for host in sorted(allowed_hosts):
        lines.append("")
        lines.append(f"# Allow {host}")
        lines.append(f"for ip in $(resolve_ipv4 {host}); do")
        lines.append('  "$IPT" -A OUTPUT -d "$ip" -p tcp --dport 80 -j ACCEPT')
        lines.append('  "$IPT" -A OUTPUT -d "$ip" -p tcp --dport 443 -j ACCEPT')
        lines.append("done")

    for host, port in extra_host_ports:
        lines.append("")
        lines.append(f"# Allow {host}:{port}")
        lines.append(f"for ip in $(resolve_ipv4 {host}); do")
        lines.append(f'  "$IPT" -A OUTPUT -d "$ip" -p tcp --dport {port} -j ACCEPT')
        lines.append("done")

    if dnat_target is not None and dnat_hosts:
        # The nat-OUTPUT DNAT block lives in one place (#1153) so the Limited
        # lockdown script and the Unrestricted DNAT-only script emit a
        # byte-identical rule shape.
        lines.extend(_nat_dnat_lines(dnat_hosts, dnat_target))

    lines.append("")
    lines.append("# Drop everything else")
    lines.append('"$IPT" -P OUTPUT DROP')

    # Belt-and-suspenders: mirror the v4 DROP on IPv6 so the IPv4-only lockdown
    # cannot be bypassed over v6 if a v6 route ever appears (#1207).
    lines.extend(_IP6TABLES_LOCKDOWN_LINES)

    return "\n".join(lines)


def build_secret_egress_dnat_script(dnat_hosts: Sequence[str], dnat_target: tuple[str, int]) -> str:
    """Install ONLY the credential-host → proxy nat-OUTPUT DNAT (no lockdown).

    For an **Unrestricted** environment that nonetheless carries env-var
    credentials (#1153): the secret swap must fire, but general egress stays
    open. So this emits the same nat-OUTPUT DNAT block as the Limited lockdown
    (via the shared :func:`_nat_dnat_lines`) but leaves the filter OUTPUT policy
    at its default ``ACCEPT`` — there is NO ``-P OUTPUT DROP`` and NO per-host
    filter ``ACCEPT`` rules. The DNATed packet (now to ``$PROXY_IP:<port>``)
    traverses the default-ACCEPT filter OUTPUT and is forwarded; no explicit
    proxy ACCEPT is needed (and adding one would contradict the no-lockdown
    intent).

    Only the nat OUTPUT chain is flushed for idempotent re-apply — the filter
    OUTPUT chain is deliberately left untouched.

    Callers only invoke this with a non-empty ``dnat_hosts`` and a real
    ``dnat_target`` (the registry routes here only when there are credentials),
    so the nat block is always emitted.
    """
    return "\n".join(
        [
            "set -e",
            "",
            _IPTABLES_BACKEND_SELECT,
            "",
            "# Resolve hosts IPv4-only so AAAA records never reach the IPv4 rules (#978)",
            _RESOLVE_IPV4_FN,
            "",
            "# Flush nat OUTPUT for idempotent re-apply (do NOT touch filter OUTPUT)",
            '"$IPT" -t nat -F OUTPUT',
            *_nat_dnat_lines(dnat_hosts, dnat_target),
            # NO `-P OUTPUT DROP`, NO filter ACCEPTs — the filter policy stays
            # ACCEPT so general egress remains open under Unrestricted.
        ]
    )


# Docker's embedded DNS, served inside every user-defined-network netns (the
# sandbox runs on the ``aios-sandbox`` user-defined bridge). The lockdown
# sidecar joins that netns but inherits the operator image's (typically empty)
# ``/etc/resolv.conf`` — Docker does NOT manage resolv.conf for a
# netns-joining container — so the sidecar script points itself at the
# embedded resolver before ``getent`` resolves the allowed hosts. A DNS miss
# fails CLOSED (the host gets no ACCEPT rule → blocked), never a bypass.
_EMBEDDED_DNS_ADDRESS = "127.0.0.11"


# Point the netns-joining sidecar at the embedded resolver before any
# ``getent`` runs (Docker doesn't manage resolv.conf for a netns-joining
# container). Prepended to BOTH the Limited lockdown apply script and the
# Unrestricted DNAT-only apply script (#1153) so credential / allowed-host
# resolution works the same way in either mode.
_RESOLV_PREAMBLE = (
    f"printf 'nameserver {_EMBEDDED_DNS_ADDRESS}\\n' > /etc/resolv.conf 2>/dev/null || true\n"
)


# Read-back assertion that the default OUTPUT policy is DROP — proves the
# lockdown actually took effect in the shared netns, not just that the apply
# script exited 0.
def build_lockdown_verify_script(
    dnat_hosts: Sequence[str] = (), *, assert_drop: bool = True
) -> str:
    """Build the read-back verify script run by the lockdown sidecar.

    When ``assert_drop`` (the default), asserts the filter-table default OUTPUT
    policy is ``DROP`` — proof the lockdown actually landed in the shared netns,
    not merely that the apply script exited 0. It ALSO asserts the IPv6
    ``ip6tables`` OUTPUT policy is ``DROP`` (#1207): the apply installs a
    belt-and-suspenders v6 DROP, and leaving it unverified would re-create the
    exact "green verify while open" gap one layer down. The v6 assertion uses
    the same legacy-backend selection as the apply so it reads the right table
    under runsc. The DNAT-only Unrestricted path (#1153) passes
    ``assert_drop=False``: that script deliberately leaves the filter policy at
    ``ACCEPT`` and installs no v6 DROP, so there is no DROP (v4 or v6) to assert
    (asserting it would always fail).

    When ``dnat_hosts`` is non-empty it ALSO asserts the nat table carries at
    least one ``DNAT`` OUTPUT rule. Without this, a credential host whose
    ``getent`` returns zero IPs emits no DNAT rule and no error: apply exits 0,
    a filter-only verify passes, and the session runs WITHOUT DNAT for that host
    — the placeholder goes direct to the real upstream and auth fails with no
    operator signal (#984). Asserting nat coverage turns that silent omission
    into a fail-closed provision error. (Coverage is asserted at the table
    level, not per-host: any host resolving to zero IPs with NO other DNAT rule
    present fails the verify; the proxy-alias DNS-miss case — where the whole
    nat block is guarded out — is the documented fail-open-to-placeholder path
    in :func:`build_iptables_script` and is out of scope here.)

    Under DNAT-only (``assert_drop=False``) the caller always passes a
    non-empty ``dnat_hosts`` — it only runs when there are credentials — so the
    verify always carries a positive nat-DNAT assertion and never degenerates
    to a no-op.
    """
    # ``set -e`` so EVERY assertion is independently fatal regardless of order.
    # The sidecar runs this via ``bash -c <script>`` with NO ``-e``, so without
    # this the script's exit status is its LAST command — and the v6 read-back
    # block below ends in a guarded ``if ...; then ...; fi`` that returns 0 when
    # the v6 ``filter`` table is unavailable (the common CI / IPv6-disabled-host
    # case). That trailing 0 would MASK a failed earlier v4 ``-P OUTPUT DROP``
    # assertion: verify passes GREEN while the box is open over IPv4 — a fail-open
    # regression on the load-bearing v4 lockdown. ``set -e`` makes the v4 (and
    # nat) assertions abort the script the instant they fail, before the v6 block
    # can overwrite the exit status. The v6 block keeps its own internal ``if``
    # guard so a missing v6 table is still a graceful skip (the guard's condition
    # being false leaves ``$?`` at 0 and ``set -e`` does NOT fire on a tested
    # condition), not a failure.
    lines = ["set -e", _IPTABLES_BACKEND_SELECT]
    if assert_drop:
        lines.append("\"$IPT\" -S OUTPUT | grep -qx -- '-P OUTPUT DROP'")
        # Extend the read-back verify to v6 (#1207): without asserting the
        # ip6tables policy too, the new v6 DROP is itself unverified — re-creating
        # the exact "green verify while open" gap one layer down. Selects the same
        # legacy backend the apply wrote to, so the verify reads the right table
        # under runsc. The assertion is GUARDED the same way the apply is: when
        # the v6 ``filter`` table is not initializable (``ip6_tables`` not loaded
        # — no v6 netfilter path to leak through, so the apply correctly skipped
        # its DROP) there is no policy to read back and the verify passes. When
        # the table IS present, ``-S OUTPUT`` succeeds and the DROP policy must be
        # there (a missing DROP fails the verify, closing the "green verify while
        # open" gap for the case the DROP actually defends).
        lines.append(_IP6TABLES_BACKEND_SELECT)
        lines.append(
            'if v6_output="$("$IP6T" -S OUTPUT 2>/dev/null)"; then '
            "printf '%s\\n' \"$v6_output\" | grep -qx -- '-P OUTPUT DROP'; fi"
        )
    if dnat_hosts:
        lines.append("\"$IPT\" -t nat -S OUTPUT | grep -q -- '-j DNAT'")
    return "\n".join(lines)


async def apply_network_lockdown(
    backend: SandboxBackend,
    handle: SandboxHandle,
    networking: LimitedNetworking,
    *,
    extra_host_ports: Sequence[tuple[str, int]] = (),
    dnat_hosts: Sequence[str] = (),
    dnat_target: tuple[str, int] | None = None,
    runtime: str | None = None,
) -> None:
    """Apply + verify iptables egress rules via an ephemeral operator-image sidecar.

    Called after package installation so ``pip install`` etc. can reach
    registries before the lockdown takes effect.

    ``runtime`` (#1014) is the container runtime for the sidecar (e.g.
    ``runsc``), threaded by the registry from the sandbox's own provisioning
    spec so the sidecar always runs under the same runtime as the sandbox it
    locks down. The backend layer takes it as an explicit parameter — it never
    reads ambient config.

    ``dnat_hosts`` + ``dnat_target`` are threaded into
    :func:`build_iptables_script` to DNAT credential-host :443 egress through
    the secret-egress proxy (#878). The read-back verify always asserts the
    filter-table DROP policy and, when ``dnat_hosts`` is non-empty, ALSO
    asserts the nat table carries a ``DNAT`` OUTPUT rule
    (:func:`build_lockdown_verify_script`) so a host resolving to zero IPs
    fails closed instead of silently running without DNAT (#984).

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
    apply_script = _RESOLV_PREAMBLE + iptables_script
    settings = get_settings()

    try:
        result = await backend.run_netns_sidecar(
            handle.sandbox_id,
            image=settings.docker_image,
            script=apply_script,
            timeout_seconds=30,
            max_output_bytes=settings.bash_max_output_bytes,
            runtime=runtime,
        )
    except SandboxBackendError:
        # Don't swallow an infra failure into a wide-open sandbox: a Limited
        # policy whose lockdown couldn't even run must fail the provision.
        log.warning("sandbox.network_lockdown_sidecar_error", owner_id=handle.owner_id)
        raise

    if result.exit_code != 0:
        log.warning(
            "sandbox.network_lockdown_failed",
            owner_id=handle.owner_id,
            exit_code=result.exit_code,
            stderr=result.stderr[:500],
        )
        raise SandboxBackendError(
            f"network lockdown failed (exit {result.exit_code}) for session "
            f"{handle.owner_id}; refusing to run a Limited sandbox with "
            f"unrestricted networking"
        )

    # Read-back verify the DROP policy actually landed in the shared netns.
    try:
        verify = await backend.run_netns_sidecar(
            handle.sandbox_id,
            image=settings.docker_image,
            script=build_lockdown_verify_script(dnat_hosts),
            timeout_seconds=15,
            max_output_bytes=settings.bash_max_output_bytes,
            runtime=runtime,
        )
    except SandboxBackendError:
        log.warning("sandbox.network_lockdown_verify_error", owner_id=handle.owner_id)
        raise
    if verify.exit_code != 0:
        log.warning(
            "sandbox.network_lockdown_verify_failed",
            owner_id=handle.owner_id,
            exit_code=verify.exit_code,
        )
        raise SandboxBackendError(
            f"network lockdown verification failed for session {handle.owner_id}: "
            "OUTPUT policy is not DROP after apply; refusing to run a Limited "
            "sandbox with unverified networking"
        )

    log.info(
        "sandbox.network_lockdown_applied",
        owner_id=handle.owner_id,
        allowed_host_count=len(allowed),
        extra_host_port_count=len(extra_host_ports),
        dnat_host_count=len(dnat_hosts),
    )


async def apply_secret_egress_dnat(
    backend: SandboxBackend,
    handle: SandboxHandle,
    *,
    dnat_hosts: Sequence[str],
    dnat_target: tuple[str, int],
    runtime: str | None = None,
) -> None:
    """Install the credential-host → proxy DNAT in an OPEN-egress sandbox (#1153).

    The Unrestricted sibling of :func:`apply_network_lockdown`: for an
    Unrestricted (or no-networking-config) environment that nonetheless carries
    env-var credentials, the secret swap must fire — but general egress stays
    open. So this runs the same operator-image netns sidecar with the same
    fail-closed posture, but applies :func:`build_secret_egress_dnat_script`
    (DNAT-only; the filter OUTPUT policy is left at ``ACCEPT``) and verifies
    with ``assert_drop=False`` (assert the nat DNAT rule exists — fail-closed on
    a zero-IP credential host per #984 — but NOT a DROP policy, of which there
    is none).

    Deliberately **NOT** factored into a shared sidecar helper with
    :func:`apply_network_lockdown`: the two paths carry genuinely different
    error semantics. A Limited apply/verify failure is a *policy violation*
    ("refusing to run a Limited sandbox"); an Unrestricted DNAT apply/verify
    failure is a *plumbing failure* (the secret-egress proxy / sidecar is
    unavailable). The log events here are plumbing-specific
    (``sandbox.secret_egress_dnat_*``) so an operator alert never mis-attributes
    a proxy outage to a networking-policy violation.

    **Fails closed**, identically to the Limited path: on a sidecar infra error,
    a nonzero apply, or a failed read-back verify, :class:`SandboxBackendError`
    propagates and the registry tears the sandbox down rather than handing back
    a half-wired credentialed box whose swap silently doesn't fire.
    """
    apply_script = _RESOLV_PREAMBLE + build_secret_egress_dnat_script(dnat_hosts, dnat_target)
    settings = get_settings()

    try:
        result = await backend.run_netns_sidecar(
            handle.sandbox_id,
            image=settings.docker_image,
            script=apply_script,
            timeout_seconds=30,
            max_output_bytes=settings.bash_max_output_bytes,
            runtime=runtime,
        )
    except SandboxBackendError:
        # A credentialed sandbox whose swap chokepoint couldn't even be wired
        # must fail the provision, not hand back a box where the secret swap
        # silently never fires.
        log.warning("sandbox.secret_egress_dnat_sidecar_error", owner_id=handle.owner_id)
        raise

    if result.exit_code != 0:
        log.warning(
            "sandbox.secret_egress_dnat_failed",
            owner_id=handle.owner_id,
            exit_code=result.exit_code,
            stderr=result.stderr[:500],
        )
        raise SandboxBackendError(
            f"secret-egress DNAT failed (exit {result.exit_code}) for session "
            f"{handle.owner_id}; refusing to run an env-var-credentialed sandbox "
            f"whose secret-swap DNAT didn't install"
        )

    # Read-back verify the nat DNAT rule actually landed — there is NO DROP
    # policy to assert under DNAT-only (assert_drop=False).
    try:
        verify = await backend.run_netns_sidecar(
            handle.sandbox_id,
            image=settings.docker_image,
            script=build_lockdown_verify_script(dnat_hosts, assert_drop=False),
            timeout_seconds=15,
            max_output_bytes=settings.bash_max_output_bytes,
            runtime=runtime,
        )
    except SandboxBackendError:
        log.warning("sandbox.secret_egress_dnat_verify_error", owner_id=handle.owner_id)
        raise
    if verify.exit_code != 0:
        log.warning(
            "sandbox.secret_egress_dnat_verify_failed",
            owner_id=handle.owner_id,
            exit_code=verify.exit_code,
        )
        raise SandboxBackendError(
            f"secret-egress DNAT verification failed for session {handle.owner_id}: "
            "nat OUTPUT carries no DNAT rule after apply; refusing to run an "
            "env-var-credentialed sandbox whose secret-swap DNAT is unverified"
        )

    log.info(
        "sandbox.secret_egress_dnat_applied",
        owner_id=handle.owner_id,
        dnat_host_count=len(dnat_hosts),
    )
