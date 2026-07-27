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
from aios.models.environments import EnvironmentConfig, LimitedNetworking, PackageManager
from aios.sandbox.backends.base import SandboxBackend, SandboxBackendError, SandboxHandle
from aios.sandbox.credential_dns import CREDENTIAL_SENTINEL_IP
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

    install_cmds: dict[PackageManager, str] = {
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


# NAME-BASED credential-host interception (eumemic/aios#2042).
#
# THE DEFECT THIS REPLACES. The credential-host rules used to be generated only
# for RESOLVED (learned) addresses: ``for ip in $(resolve_ipv4 <host>)`` emitted
# one nat DNAT per address the sidecar's DNS query happened to return. A
# rotating pool — api.github.com serves a ~60s-TTL set and answers with only a
# SUBSET per query — makes that set a SAMPLE, never the pool. Under Unrestricted
# (filter policy ``ACCEPT``) an address no sampler ever returned matched no
# rule, egressed DIRECTLY to the real upstream carrying the literal
# ``AIOS_SECRET_PLACEHOLDER_*``, and came back ``401`` — the misleading
# "flaky auth, retry fixes it" signature. More probes shrink that window and
# CANNOT close it; every IP-keyed variant (a wider fence, a bigger sample)
# re-inherits the same defect one level down, because a sample is not a
# guarantee.
#
# THE FIX: stop keying policy on addresses. Every DNS query leaving the netns is
# redirected (nat OUTPUT, ``-I`` at the TOP of the chain so no in-netns resolver
# — including Docker's embedded 127.0.0.11 — can answer first) to the
# per-session worker-controlled resolver in :mod:`aios.sandbox.credential_dns`.
# That resolver answers a CREDENTIAL HOST with one fixed, non-routable sentinel
# address (:data:`CREDENTIAL_SENTINEL_IP`) and never forwards those names, so
# the sandbox cannot learn a real pool address for them at all; every other name
# is forwarded verbatim, so ordinary resolution is unchanged. The netns then
# needs exactly ONE credential rule, keyed on a constant THIS worker chose:
#
#     -t nat -A OUTPUT -d <sentinel> -p tcp --dport 443 \
#         -j DNAT --to-destination "$PROXY_IP:<proxy_port>"
#
# An address nobody ever sampled is now structurally incapable of bypassing the
# proxy — not because we enumerated it, but because the NAME can no longer
# resolve to it inside the sandbox. Addresses have stopped being what policy
# depends on.
#
# FAIL CLOSED, three ways:
#   * the sentinel is RFC 3927 link-local and routed nowhere, so a missing or
#     malformed DNAT kills the connection in the sandbox's own stack instead of
#     sending a placeholder to the real upstream. A broken rule can only DENY;
#   * a filter REJECT catches any other sentinel-addressed traffic (e.g. :80).
#     It is NOT an IP-keyed fence over a sampled set — the address it names is
#     our own constant, and it covers the host completely because the host has
#     exactly one address inside the sandbox now;
#   * the proxy-alias lookup is now a HARD ERROR (``exit 1``) rather than a
#     silently-skipped block: previously a ``$PROXY_IP`` miss guarded the whole
#     nat block out and every credential request went straight to the real
#     upstream. A sandbox whose credential egress cannot be protected must not
#     be able to send a credential, so the apply fails and the provision aborts.
#
# Behaviourally pinned by ``TestCredentialHostEgressVerdict``
# (tests/unit/test_networking.py), which resolves the host THROUGH the generated
# ruleset and replays the resulting packet — including a live pool address no
# sampler ever returned.


def _nat_dnat_lines(
    dnat_hosts: Sequence[str],
    dnat_target: tuple[str, int],
    dns_port: int,
    *,
    filter_accepts: bool = False,
    intercept_all_https: bool = False,
) -> list[str]:
    """The name-based credential-interception block (#2042).

    Shared by the Limited lockdown script (:func:`build_iptables_script`) and
    the Unrestricted DNAT-only script (:func:`build_secret_egress_dnat_script`)
    so both modes install a byte-identical chokepoint (#1153).

    Emits, in order:

    1. ``PROXY_IP`` — the proxy alias resolved ONCE at sidecar runtime
       (iptables ``--to-destination`` needs an IP, not a name). A miss is a
       HARD FAILURE now: it exits nonzero so the provision aborts, instead of
       guarding the block out and letting every credential request reach the
       real upstream with a literal placeholder.
    2. The DNS interception: udp+tcp ``:53`` DNATed to the worker-controlled
       resolver, inserted with ``-I`` at the TOP of nat OUTPUT so nothing in
       the netns can answer a credential name first.
    3. A destination-independent HTTPS DNAT.  The proxy forwards ordinary SNI
       hosts unchanged, swaps credential hosts, and refuses absent SNI.  This is
       the destination-side floor for raw, cached, DoH, and ``/etc/hosts``
       addresses that never passed through the credential resolver.
    4. A filter REJECT for any other sentinel-addressed packet (the ``:443``
       flow is already rewritten to the proxy by the nat table, which runs
       first, so this cannot catch it).

    ``dnat_hosts`` is no longer used to generate per-address rules; it is
    carried for the emitted comment (and to keep the callers' contract that the
    block is only emitted when there are credential hosts). ``filter_accepts``
    is set by the Limited path only: its ``-P OUTPUT DROP`` would otherwise
    drop the post-DNAT DNS flow to the proxy's resolver port.
    """
    proxy_alias, proxy_port = dnat_target
    lines = [
        "",
        "# Name-based credential-host interception (#2042): policy is keyed on the",
        "# NAME, never on a sampled address. Credential hosts: " + ", ".join(sorted(dnat_hosts)),
        # Resolve the proxy alias to an IP ONCE — iptables --to-destination
        # needs an IP, not a DNS name.
        f"PROXY_IP=$(resolve_ipv4 {proxy_alias} | head -n1)",
        'if [ -z "$PROXY_IP" ]; then',
        f'  echo "credential interception: proxy alias {proxy_alias} did not resolve; '
        'refusing to run a credentialed sandbox with unprotected egress" >&2',
        "  exit 1",
        "fi",
        "# All DNS out of this netns goes to the worker-controlled resolver. -I puts",
        "# these at the TOP of nat OUTPUT so no in-netns resolver answers first.",
        f'"$IPT" -t nat -I OUTPUT -p udp --dport 53 -j DNAT --to-destination "$PROXY_IP:{dns_port}"',
        f'"$IPT" -t nat -I OUTPUT -p tcp --dport 53 -j DNAT --to-destination "$PROXY_IP:{dns_port}"',
        "# Credential names resolve to the sentinel.",
        f'"$IPT" -t nat -A OUTPUT -d {CREDENTIAL_SENTINEL_IP} -p tcp --dport 443 '
        f'-j DNAT --to-destination "$PROXY_IP:{proxy_port}"',
    ]
    if intercept_all_https:
        lines.extend(
            [
                "# Unrestricted destination-side floor for raw/cached/DoH addresses.",
                f'"$IPT" -t nat -A OUTPUT -p tcp --dport 443 '
                f'-j DNAT --to-destination "$PROXY_IP:{proxy_port}"',
            ]
        )
    if filter_accepts:
        lines.extend(
            [
                "# Limited only: -P OUTPUT DROP would otherwise drop the post-DNAT DNS",
                "# flow to the worker resolver (filter sees the REWRITTEN destination).",
                f'"$IPT" -A OUTPUT -d "$PROXY_IP" -p udp --dport {dns_port} -j ACCEPT',
                f'"$IPT" -A OUTPUT -d "$PROXY_IP" -p tcp --dport {dns_port} -j ACCEPT',
            ]
        )
    lines.extend(
        [
            "# Fail closed: anything else addressed to the sentinel (e.g. :80) is",
            "# refused rather than left to leak. The :443 flow never reaches here —",
            "# nat runs first and has already rewritten it to the proxy.",
            f'"$IPT" -A OUTPUT -d {CREDENTIAL_SENTINEL_IP} -j REJECT '
            "--reject-with icmp-port-unreachable",
        ]
    )
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

    **Credential hosts no longer take part in this sweep (#2042).** They used
    to get one nat DNAT per newly-sampled address, and lose it again when the
    address aged out — the sampling machinery that made an unsampled address
    fail open in the first place. Interception is now keyed on the NAME (the
    single sentinel address every credential name resolves to inside the
    sandbox), so there is nothing per-address left to refresh, and re-adding
    per-address DNATs here would quietly restore an IP-keyed variant of the
    exact defect. ``credential_hosts`` is still accepted so callers keep their
    contract and so a host that is BOTH a credential host and an allowed
    Limited host still gets its filter ACCEPTs refreshed via ``limited_hosts``.
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
    # Legacy per-address credential DNAT shape, kept ONLY as a delete target:
    # a session provisioned before #2042 (or a snapshot resumed across the
    # upgrade) can still carry these, and the sweep should retire them. Nothing
    # here ever ADDS one.
    legacy_dnat_tail = f"-p tcp --dport 443 -j DNAT --to-destination {proxy_ip}:{proxy_port}"
    lines = ["set -e", _IPTABLES_BACKEND_SELECT]
    for host in sorted(new_ips):
        added = new_ips[host] - old_ips.get(host, set())
        for ip in sorted(added):
            if host in limited_hosts:
                lines.append(_add("", f"-d {ip} -p tcp --dport 80 -j ACCEPT"))
                if host not in credential_hosts:
                    lines.append(_add("", f"-d {ip} -p tcp --dport 443 -j ACCEPT"))
    for host in sorted(old_ips):
        removed = old_ips[host] - new_ips.get(host, set())
        for ip in sorted(removed):
            if host in credential_hosts:
                lines.append(_delete(" -t nat", f"-d {ip} {legacy_dnat_tail}"))
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
    dns_port: int | None = None,
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

    When ``dnat_target`` + ``dns_port`` are supplied alongside a non-empty
    ``dnat_hosts``, the name-based credential-interception block is emitted
    (:func:`_nat_dnat_lines`, #2042): all ``:53`` is DNATed to the
    worker-controlled resolver, and the single sentinel address every
    credential name now resolves to inside the sandbox is DNATed on ``:443``
    to the secret-egress proxy. NOTHING here is keyed on a sampled address any
    more. A proxy-alias DNS miss is a HARD apply failure (``exit 1``) — the old
    ``if [ -n "$PROXY_IP" ]`` guard silently skipped the whole block and sent
    every credential request to the real upstream carrying the literal
    placeholder; a sandbox that cannot protect a credential must not be able to
    send one. ``dnat_target``/``dns_port`` of ``None`` (the default) emits NO
    nat rules, preserving every existing caller.

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

    credential_hosts = set(dnat_hosts) if dnat_target is not None else set()
    for host in sorted(allowed_hosts):
        lines.append("")
        lines.append(f"# Allow {host}")
        lines.append(f"for ip in $(resolve_ipv4 {host}); do")
        lines.append('  "$IPT" -A OUTPUT -d "$ip" -p tcp --dport 80 -j ACCEPT')
        if host not in credential_hosts:
            # Raw-IP credential traffic must fall through to DROP rather than
            # bypass the name-based proxy via this sampled-address allowance.
            lines.append('  "$IPT" -A OUTPUT -d "$ip" -p tcp --dport 443 -j ACCEPT')
        lines.append("done")

    for host, port in extra_host_ports:
        lines.append("")
        lines.append(f"# Allow {host}:{port}")
        lines.append(f"for ip in $(resolve_ipv4 {host}); do")
        lines.append(f'  "$IPT" -A OUTPUT -d "$ip" -p tcp --dport {port} -j ACCEPT')
        lines.append("done")

    if dnat_target is not None and dnat_hosts and dns_port is not None:
        # The credential-interception block lives in one place (#1153/#2042) so
        # the Limited lockdown script and the Unrestricted DNAT-only script
        # install a byte-identical chokepoint. ``filter_accepts`` opens the
        # post-DNAT DNS flow to the worker resolver, which this script's
        # terminal ``-P OUTPUT DROP`` would otherwise drop.
        lines.extend(_nat_dnat_lines(dnat_hosts, dnat_target, dns_port, filter_accepts=True))

    lines.append("")
    lines.append("# Drop everything else")
    lines.append('"$IPT" -P OUTPUT DROP')

    # Belt-and-suspenders: mirror the v4 DROP on IPv6 so the IPv4-only lockdown
    # cannot be bypassed over v6 if a v6 route ever appears (#1207).
    lines.extend(_IP6TABLES_LOCKDOWN_LINES)

    return "\n".join(lines)


def build_secret_egress_dnat_script(
    dnat_hosts: Sequence[str], dnat_target: tuple[str, int], dns_port: int
) -> str:
    """Install ONLY the credential-host interception chokepoint (no lockdown).

    For an **Unrestricted** environment that nonetheless carries env-var
    credentials (#1153): the secret swap must fire, but general egress stays
    open. So this emits the same name-based interception block as the Limited
    lockdown (via the shared :func:`_nat_dnat_lines`) while leaving the filter
    OUTPUT policy at its default ``ACCEPT`` — there is NO ``-P OUTPUT DROP`` and
    NO per-allowed-host filter ``ACCEPT`` rules. The DNATed packet (now to
    ``$PROXY_IP:<port>``) traverses the default-ACCEPT filter OUTPUT and is
    forwarded, so no ``filter_accepts`` block is needed here (adding one would
    contradict the no-lockdown intent).

    **This is the path #2042 was filed against, and where the fix bites
    hardest.** Under the old IP-keyed shape the default-``ACCEPT`` policy meant
    an address no sampler returned egressed DIRECTLY with the literal
    placeholder. Now no credential name resolves to a real address inside the
    sandbox at all: it resolves to the sentinel, whose only route out is the
    nat DNAT to the proxy, and whose non-``:443`` traffic is REJECTed. The
    filter policy staying ``ACCEPT`` no longer helps an unsampled address —
    there is no such thing as an unsampled address here any more.

    Both the filter REJECT and the sentinel DNAT are emitted UNCONDITIONALLY
    (no resolution loop), so unlike every previous shape their coverage does not
    depend on what DNS happened to return.

    Only the nat OUTPUT chain is flushed for idempotent re-apply — the filter
    OUTPUT chain is deliberately left untouched, except for the single sentinel
    REJECT this function must own. That REJECT is deleted-then-appended so a
    re-apply cannot stack duplicates.

    Callers only invoke this with a non-empty ``dnat_hosts`` and a real
    ``dnat_target`` (the registry routes here only when there are credentials),
    so the block is always emitted.
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
            "# Flush nat OUTPUT for idempotent re-apply (do NOT touch filter OUTPUT:",
            "# under Unrestricted it carries the operator's / Docker's own rules).",
            '"$IPT" -t nat -F OUTPUT',
            "# Drop our own previous sentinel REJECT (if any) so a re-apply cannot",
            "# stack duplicates; the append below reinstates it.",
            f'"$IPT" -D OUTPUT -d {CREDENTIAL_SENTINEL_IP} -j REJECT '
            "--reject-with icmp-port-unreachable 2>/dev/null || true",
            *_nat_dnat_lines(dnat_hosts, dnat_target, dns_port, intercept_all_https=True),
            # NO `-P OUTPUT DROP`, NO per-allowed-host filter ACCEPTs — the
            # filter policy stays ACCEPT so general egress remains open.
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

    When ``dnat_hosts`` is non-empty it ALSO reads back every rule the
    name-based credential chokepoint depends on (#2042): the ``:53`` DNAT to
    the worker-controlled resolver (udp AND tcp), the sentinel ``:443`` DNAT to
    the secret-egress proxy, and the sentinel filter REJECT. Asserting merely
    that "some ``-j DNAT`` exists" (the pre-#2042 check) would pass on a
    half-installed chokepoint — DNS intercepted but the sentinel unrouted, or
    the reverse — which is a green verify over unprotected credential egress,
    the precise failure mode this issue exists to kill. Each assertion is
    independently fatal under ``set -e``, so a partial apply fails the
    provision instead of downgrading it silently. (This subsumes #984: a host
    that resolves to zero IPs is no longer even relevant, because no rule is
    keyed on a resolution any more.)

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
        # Read back the THREE rules that make the name-based chokepoint real
        # (#2042). Asserting only "some DNAT exists" would pass on a ruleset
        # that intercepts DNS but never redirects the sentinel (or vice versa)
        # — i.e. green verify while credential egress is unprotected. Each is
        # independently fatal under ``set -e``.
        lines.append(
            '"$IPT" -t nat -S OUTPUT | grep -q -- '
            f"'-d {CREDENTIAL_SENTINEL_IP}.*--dport 443 -j DNAT'"
        )
        # ``iptables -S`` canonicalizes port matches by inserting the protocol
        # module (for example ``-p udp -m udp --dport 53``), so allow tokens
        # between the protocol and destination port.  Matching them adjacently
        # made the read-back reject a correctly installed chokepoint on real
        # Docker hosts even though recording unit shims preserved input syntax.
        lines.append("\"$IPT\" -t nat -S OUTPUT | grep -q -- '-p udp .*--dport 53 .*-j DNAT'")
        lines.append("\"$IPT\" -t nat -S OUTPUT | grep -q -- '-p tcp .*--dport 53 .*-j DNAT'")
        lines.append(f"\"$IPT\" -S OUTPUT | grep -q -- '-d {CREDENTIAL_SENTINEL_IP}.*-j REJECT'")
    return "\n".join(lines)


async def apply_network_lockdown(
    backend: SandboxBackend,
    handle: SandboxHandle,
    networking: LimitedNetworking,
    *,
    extra_host_ports: Sequence[tuple[str, int]] = (),
    dnat_hosts: Sequence[str] = (),
    dnat_target: tuple[str, int] | None = None,
    dns_port: int | None = None,
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

    ``dnat_hosts`` + ``dnat_target`` + ``dns_port`` are threaded into
    :func:`build_iptables_script` to install the name-based credential
    chokepoint (#878, #2042): all ``:53`` to the worker-controlled resolver,
    and the sentinel every credential name resolves to redirected on ``:443``
    to the secret-egress proxy. The read-back verify always asserts the
    filter-table DROP policy and, when ``dnat_hosts`` is non-empty, ALSO
    asserts every rule of that chokepoint landed
    (:func:`build_lockdown_verify_script`), so a partial install fails the
    provision rather than running unprotected.

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
        dns_port=dns_port,
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
    dns_port: int,
    runtime: str | None = None,
) -> None:
    """Install the name-based credential chokepoint in an OPEN-egress sandbox.

    The Unrestricted sibling of :func:`apply_network_lockdown` (#1153): for an
    Unrestricted (or no-networking-config) environment that nonetheless carries
    env-var credentials, the secret swap must fire — but general egress stays
    open. So this runs the same operator-image netns sidecar with the same
    fail-closed posture, but applies :func:`build_secret_egress_dnat_script`
    (no lockdown; the filter OUTPUT policy is left at ``ACCEPT``) and verifies
    with ``assert_drop=False`` (assert the whole name-based chokepoint landed,
    but NOT a DROP policy, of which there is none).

    **This is the path #2042 was filed against.** The interception installed
    here is keyed on NAMES, not on addresses a DNS sample happened to return,
    so a credential host resolving to an address no sampler ever saw is still
    proxied: inside this sandbox that name resolves ONLY to the sentinel, and
    the sentinel's only route out is the DNAT to the proxy.

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
    apply_script = _RESOLV_PREAMBLE + build_secret_egress_dnat_script(
        dnat_hosts, dnat_target, dns_port
    )
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
