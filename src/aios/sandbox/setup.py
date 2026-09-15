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
``busybox`` would otherwise downgrade to unrestricted networking without
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
from dataclasses import dataclass

from aios.config import get_settings
from aios.logging import get_logger
from aios.models.environments import EnvironmentConfig, LimitedNetworking, PackageManager
from aios.sandbox.backends.base import SandboxBackend, SandboxBackendError, SandboxHandle
from aios.sandbox.credential_dns import CREDENTIAL_SENTINEL_IP
from aios.sandbox.egress_ca import CA_CERT_SANDBOX_PATH, get_egress_ca
from aios.sandbox.env_keys import PATH_ENV_KEY

log = get_logger("aios.sandbox.setup")


@dataclass(frozen=True)
class HostSkip:
    host: str
    reason: str


@dataclass(frozen=True)
class EgressProvisionResult:
    hosts_installed: tuple[str, ...] = ()
    hosts_skipped: tuple[HostSkip, ...] = ()


_EGRESS_INSTALLED_PREFIX = "AIOS_EGRESS_INSTALLED "
_EGRESS_SKIPPED_PREFIX = "AIOS_EGRESS_SKIPPED "


def _parse_egress_provision_result(stdout: str) -> EgressProvisionResult:
    installed: set[str] = set()
    skipped: dict[str, HostSkip] = {}
    for line in stdout.splitlines():
        if line.startswith(_EGRESS_INSTALLED_PREFIX):
            host = line.removeprefix(_EGRESS_INSTALLED_PREFIX)
            installed.add(host)
            skipped.pop(host, None)
        elif line.startswith(_EGRESS_SKIPPED_PREFIX):
            value = line.removeprefix(_EGRESS_SKIPPED_PREFIX)
            host, reason = value.split("\t", 1)
            if host not in installed:
                skipped[host] = HostSkip(host=host, reason=reason)
    return EgressProvisionResult(
        hosts_installed=tuple(sorted(installed)),
        hosts_skipped=tuple(skipped[host] for host in sorted(skipped)),
    )


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


# The loopback exclusion on the Unrestricted catch-all below. ``route_localnet``
# is on for this netns, so without it every in-sandbox ``https://127.0.0.1``
# (a dev server the model just started, a local test fixture) would be dragged
# out to the worker proxy and answered by whatever the SNI resolved to.
_LOOPBACK_CIDR = "127.0.0.0/8"


# Unrestricted-only IPv6 companion to the catch-all HTTPS DNAT (#2422).
#
# The whole credential chokepoint — the ``:53`` interception, the sentinel DNAT,
# the catch-all ``:443`` DNAT — is IPv4 ``iptables``, and the secret-egress proxy
# binds the IPv4 ``WORKER_NETWORK_ALIAS``. A sandbox with a v6 route therefore
# has an un-chokepointed second stack: resolve a credential host over v6 DNS (or
# just dial a known v6 literal), connect over v6, and the placeholder reaches the
# real upstream with no swap and no proxy — the same direct-IP bypass, one stack
# down. The Limited path already closes this with a blanket v6 ``-P OUTPUT DROP``
# (#1207); Unrestricted must stay open, so deny exactly the port the swap lives
# on and leave the rest of v6 egress alone.
#
# Inert today (the ``aios-sandbox`` network is created without ``--ipv6``, so no
# v6 route exists) and fail-closed the moment that stops being true — which is
# the point: the current safety rests on an implicit network-creation flag, and
# #1207 exists because that is not a property to depend on.
#
# Guarded exactly like :data:`_IP6TABLES_LOCKDOWN_LINES`: where the v6 ``filter``
# table will not initialize (``ip6_tables`` not loaded — the ordinary CI /
# IPv6-disabled-host case) there is no v6 netfilter path to leak through, so skip
# rather than abort the apply under ``set -e``. Delete-then-append so a re-apply
# cannot stack duplicates (this function does not flush the v6 chain — under
# Unrestricted it is not ours to flush).
_IP6TABLES_CREDENTIAL_HTTPS_DENY_LINES = (
    "",
    "# Deny IPv6 :443 (#2422): the credential chokepoint is IPv4-only, so an",
    "# HTTPS connection over v6 would reach a credential host un-proxied. Only",
    "# :443 is denied — the rest of v6 egress stays open (Unrestricted).",
    _IP6TABLES_BACKEND_SELECT,
    'if "$IP6T" -S OUTPUT >/dev/null 2>&1; then',
    '  "$IP6T" -D OUTPUT -p tcp --dport 443 -j DROP 2>/dev/null || true',
    '  "$IP6T" -A OUTPUT -p tcp --dport 443 -j DROP',
    "else",
    '  echo "ip6tables filter table unavailable (ip6_tables not loaded); no IPv6 '
    'path to a credential host — skipping v6 :443 deny" >&2',
    "fi",
)

# Docker's embedded DNS, served inside every user-defined-network netns (the
# sandbox runs on the ``aios-sandbox`` user-defined bridge). Every hostname the
# lockdown scripts resolve THROUGH DNS is resolved against THIS address and no
# other.
_EMBEDDED_DNS_ADDRESS = "127.0.0.11"


# The netns's own name table, consulted BEFORE DNS (see ``_RESOLVE_IPV4_FN``).
# A named constant so a test can retarget the lookup at a fixture file.
_HOSTS_FILE = "/etc/hosts"


# awk program that reads ``_HOSTS_FILE`` looking for ``name``: strip the
# comment, keep only lines whose first field is a dotted quad (so ``::1
# localhost`` and friends never reach an IPv4 ``iptables -d``), and print that
# address when ``name`` matches the canonical name or any alias on the line.
# ``next`` after the first hit so a line naming it twice prints once.
_HOSTS_LOOKUP_AWK = (
    '{ sub(/#.*/, "") } '
    "$1 ~ /^[0-9]+[.][0-9]+[.][0-9]+[.][0-9]+$/ "
    "{ for (i = 2; i <= NF; i++) if ($i == name) { print $1; next } }"
)


# awk program parsing busybox 1.35 ``nslookup`` output:
#
#     Server:\t\t127.0.0.11
#     Address:\t127.0.0.11:53
#     <blank>
#     Name:\texample.com
#     Address: 93.184.216.34
#
# The server block is skipped by requiring a preceding ``Name:`` line -- not by
# pattern-matching the address -- so the resolver's own address can never be
# mistaken for an answer and handed to ``iptables -d`` or picked up as
# ``$PROXY_IP``. AAAA answers print in the same ``Address:`` shape and are
# rejected by the dotted-quad test.
_NSLOOKUP_PARSE_AWK = (
    "/^Name:/ { answer = 1 } "
    '/^Address:/ && answer && $2 != "' + _EMBEDDED_DNS_ADDRESS + '" '
    "&& $2 ~ /^[0-9]+[.][0-9]+[.][0-9]+[.][0-9]+$/ { print $2 }"
)

_HOSTS_LOOKUP_CMD = (
    'awk -v name="$1" \'' + _HOSTS_LOOKUP_AWK + "' " + _HOSTS_FILE + " 2>/dev/null | sort -u"
)

_NSLOOKUP_CMD = (
    'busybox nslookup "$1" '
    + _EMBEDDED_DNS_ADDRESS
    + " 2>/dev/null | awk '"
    + _NSLOOKUP_PARSE_AWK
    + "' | sort -u"
)


# Emitted shell helper that resolves a hostname to its **IPv4 addresses only**,
# one per line. Centralizes the resolution shared by every host lookup in the
# lockdown scripts (the allowed-host loops, the extra-host-ports loop, the
# credential-host DNAT loop, and the proxy-alias lookup), so every invariant
# below lives in exactly one place (#978).
#
# HOSTS FIRST, THEN DNS (aios#2410) -- the ``files dns`` order glibc gives
# every other resolver in the container, because a DNS-only lookup cannot see a
# ``--add-host`` alias. ``aios-worker`` has TWO resolution paths (see
# :mod:`aios.sandbox.network`): the embedded DNS when the worker itself sits on
# the sandbox network, and ``/etc/hosts`` when it runs on the HOST -- the e2e
# and host-worker shape, where the sandbox is created with ``--add-host
# aios-worker:host-gateway`` and Docker writes that into the container's
# ``/etc/hosts`` WITHOUT publishing it to the embedded DNS at 127.0.0.11. Only
# the first path survives a DNS-only lookup, so on the second ``resolve_ipv4
# aios-worker`` answers nothing, ``PROXY_IP`` comes back empty, the whole
# ``if [ -n "$PROXY_IP" ]`` nat block is skipped, and the apply exits 0 with
# ``nat OUTPUT`` carrying no DNAT rule at all -- the credential-host redirect
# silently absent.
#
# Which ``/etc/hosts`` each sidecar shape reads:
#
#   * runc: the sidecar joins with ``--network container:<id>``, and Docker
#     bind-mounts the TARGET's ``/etc/hosts`` into it along with the rest of the
#     netns-owned files. So the sidecar reads the SANDBOX's file, ``--add-host``
#     alias included, and the lookup lands.
#   * runsc: the exec chroots into the read-only operator image first, so it
#     reads the OPERATOR image's ``/etc/hosts``, which carries no alias. The
#     lookup misses and falls through to DNS -- i.e. exactly the DNS-only
#     behaviour it has today. No regression, and no fix either: closing it means
#     injecting an address resolved outside the netns instead of looking one up
#     inside it, which is a different change and deliberately not attempted here.
#
# NOT a return to ``getent``, which stays banned (pinned by
# ``test_no_script_touches_resolv_conf_or_getent``). glibc reads the hosts file
# AND ``/etc/resolv.conf``, and the resolv.conf half is the unusable one: it
# cannot be written on either sidecar shape and BuildKit commits an EMPTY entry
# for any ``COPY`` to that path, verified against a CI-built image with a plain
# ``COPY`` and again with ``COPY --link`` (moby/buildkit#1267; DONE.md carries
# the evidence chain). Reading the hosts file with awk takes the ``files`` half
# of nsswitch without taking the ``dns`` half's dependency on a file we cannot
# supply; the ``dns`` half is busybox nslookup with the server as an ARGUMENT.
#
# ``busybox nslookup`` rather than any glibc path: busybox is already in the
# image as the runsc chroot's static entry binary, it takes the server as an
# argument, and being static it needs no dynamic loader -- so the runsc
# preamble can bind it directly instead of through the operator image's ld.so.
#
# TENANT-WRITABLE INPUT. On the runc shape the file above is the sandbox's, and
# root inside the sandbox can write it. A tenant entry cannot introduce a NAME
# (only operator-configured hosts are ever looked up) but it can choose the
# ADDRESS an allowed name resolves to, and therefore the address a rule is
# installed for. Provision-time applies are out of reach -- the lockdown lands
# before any tenant code runs -- so the exposed path is the refresh tick; see
# :func:`build_egress_resolve_script`, which carries the caveat in full.
#
# Why IPv4-only: every rule emitted by these scripts is an IPv4 ``iptables``
# command, and the secret-egress proxy binds the IPv4 ``WORKER_NETWORK_ALIAS``
# (it cannot intercept IPv6). Feeding an AAAA literal to an IPv4-only
# ``iptables -d`` would error, and under ``set -e`` abort the whole apply. The
# sandbox network is currently IPv4-only so this is latent today, but if an
# IPv6-capable network is ever enabled it would break Limited networking on
# every IPv6-resolving host. Keeping only dotted-quad answers means any
# AAAA/IPv6 egress is simply dropped by the default policy (fail-closed) --
# which is the correct semantics for credential hosts too (IPv6 must never be
# sent un-proxied).
#
# A resolution miss on BOTH steps prints nothing (the caller's ``for`` loop /
# ``$()`` capture sees no IPs), so the host gets no rule -- fail-closed, never a
# bypass. Both steps are pipelines ending in ``sort -u``, so neither a missing
# hosts file nor an nslookup failure can return nonzero and abort the caller's
# ``set -e`` script.
_RESOLVE_IPV4_FN = "\n".join(
    (
        "resolve_ipv4() {",
        f"  _hosts_ips=$({_HOSTS_LOOKUP_CMD})",
        '  if [ -n "$_hosts_ips" ]; then printf \'%s\\n\' "$_hosts_ips"; return 0; fi',
        f"  {_NSLOOKUP_CMD}",
        "}",
    )
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

_CREDENTIAL_DNS_SNAT_CHAIN = "AIOS_CRED_DNS_SNAT"


def _nat_dnat_lines(
    dnat_hosts: Sequence[str],
    dnat_target: tuple[str, int],
    dns_port: int,
    *,
    filter_accepts: bool = False,
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
    3. UDP+TCP source NAT for the redirected DNS flow. Docker's 127.0.0.11
       destination makes the kernel select a loopback source before OUTPUT
       DNAT; MASQUERADE replaces it so the packet can cross the bridge. The
       matching REPLY needs ``net.ipv4.conf.all.route_localnet=1`` on the
       sandbox container (``SandboxSpec.route_localnet``) — see the inline
       comment — plus the filter INPUT guard that keeps that sysctl from
       exposing this netns's loopback services to the sandbox bridge.
    4. One credential DNAT keyed on :data:`CREDENTIAL_SENTINEL_IP` — the single
       address every credential name now resolves to inside the sandbox.
    5. A filter REJECT for any other sentinel-addressed packet (the ``:443``
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
        # Docker tells the container to query its embedded resolver at
        # 127.0.0.11.  The kernel selects a loopback source before nat OUTPUT;
        # changing only the destination to the worker would therefore put a
        # 127/8-sourced packet on the bridge, where it is dropped as a martian.
        # Source-NAT the rewritten DNS flow so it can actually cross the bridge
        # and receive the answer.  A private chain makes reprovision idempotent
        # without flushing Docker's own POSTROUTING rules.
        f'"$IPT" -t nat -N {_CREDENTIAL_DNS_SNAT_CHAIN} 2>/dev/null || '
        f'"$IPT" -t nat -F {_CREDENTIAL_DNS_SNAT_CHAIN}',
        f'"$IPT" -t nat -C POSTROUTING -j {_CREDENTIAL_DNS_SNAT_CHAIN} 2>/dev/null || '
        f'"$IPT" -t nat -I POSTROUTING -j {_CREDENTIAL_DNS_SNAT_CHAIN}',
        f'"$IPT" -t nat -A {_CREDENTIAL_DNS_SNAT_CHAIN} -d "$PROXY_IP" '
        f"-p udp --dport {dns_port} -j MASQUERADE",
        f'"$IPT" -t nat -A {_CREDENTIAL_DNS_SNAT_CHAIN} -d "$PROXY_IP" '
        f"-p tcp --dport {dns_port} -j MASQUERADE",
        # The MASQUERADE above only repairs the REQUEST. The reply is un-SNATed
        # in nat PREROUTING back to a 127.0.0.1 DESTINATION before input
        # routing, which the kernel discards as a martian destination unless
        # net.ipv4.conf.all.route_localnet=1 — so the sandbox container is
        # started with that sysctl (``SandboxSpec.route_localnet``, set by the
        # spec builder for exactly the sessions that reach this block).
        #
        # route_localnet also makes 127.0.0.0/8 services in this netns
        # reachable from the (ICC-on) sandbox bridge, so pay for it here: drop
        # every NEW loopback-destined flow that did not arrive on lo. The
        # redirected DNS answer is ESTABLISHED by the time it reaches filter
        # INPUT (conntrack runs in PREROUTING, nat's LOCAL_IN source rewrite
        # runs after filter), so it is unaffected. Idempotent -C/-A: INPUT is
        # never flushed, so a reprovision must not append a second copy.
        "# route_localnet is on for this netns (#2422); keep it from exposing",
        "# loopback services to the bridge. The redirected DNS reply is",
        "# ESTABLISHED here, so only NEW inbound loopback flows are dropped.",
        "\"$IPT\" -C INPUT '!' -i lo -d 127.0.0.0/8 -m conntrack --ctstate NEW "
        "-j DROP 2>/dev/null || "
        "\"$IPT\" -A INPUT '!' -i lo -d 127.0.0.0/8 -m conntrack --ctstate NEW -j DROP",
        "# The ONE credential rule: every credential name resolves to this sentinel",
        "# inside the sandbox, so this covers the host completely (#2042).",
        f'"$IPT" -t nat -A OUTPUT -d {CREDENTIAL_SENTINEL_IP} -p tcp --dport 443 '
        f'-j DNAT --to-destination "$PROXY_IP:{proxy_port}"',
    ]
    # #2193 provision report. Under name-based interception coverage is
    # complete by construction — the one sentinel rule covers every credential
    # name — so each host is INSTALLED unconditionally. There is no per-host
    # skip left to report: the only way this block fails is the proxy-alias
    # miss above, which now exits nonzero and aborts the provision outright.
    for host in sorted(dnat_hosts):
        lines.append(f"echo '{_EGRESS_INSTALLED_PREFIX}{host}'")
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
    """Resolve refresh hosts inside the sandbox netns, one machine-readable row per IP.

    CAVEAT — TENANT-WRITABLE ``/etc/hosts`` ON THIS PATH. ``resolve_ipv4`` reads
    the hosts file before asking DNS (aios#2410, needed so a ``--add-host``
    alias resolves at all), and on the runc shape that file is the SANDBOX's,
    writable by root inside the container. The provision-time applies are out of
    reach — the lockdown lands before any tenant code runs — but this script
    runs on every refresh tick, by which point the tenant has had the container.
    So an entry written there names the address the refreshed ACCEPT/DNAT rule
    is installed for. The exposure is the allow-list's ADDRESSES, never its
    NAMES: only operator-configured hosts are ever looked up, so a tenant can
    point an already-allowed name at an address of their choosing, not admit a
    name of their choosing. Closing it means resolving OUTSIDE the tenant's
    reach and injecting the answer into the script instead of looking it up in
    the netns — the same change the runsc alias gap needs, and out of scope
    here.
    """
    lines = ["set -e", _RESOLVE_IPV4_FN]
    for host in sorted(set(hosts)):
        lines.append(f"for ip in $(resolve_ipv4 {host}); do printf '%s %s\\n' {host} \"$ip\"; done")
    return "\n".join(lines)


def egress_unread_hosts(
    *,
    new_ips: dict[str, set[str]],
    credential_hosts: set[str],
    limited_hosts: set[str],
) -> list[str]:
    """In-scope hosts ABSENT from ``new_ips`` — i.e. hosts whose IPs were not read.

    Absence and presence-with-an-empty-set are DIFFERENT facts: the first is
    "could not be read", the second is "read, and this host genuinely owns
    nothing". Only the second may drive a deletion.

    Exposed (rather than inlined into :func:`build_egress_refresh_script`) so
    the CALLER can act on the same signal the builder acts on. The builder can
    only decline to emit deletes; it cannot stop the caller from advancing its
    ``pinned`` bookkeeping past IPs whose rules were deliberately left
    installed. Both layers must read the identical predicate or the two
    disagree — which is how a refusal to delete silently becomes "the rule is
    installed and nothing remembers it exists".

    NOTE ON REACHABILITY (measured, not assumed): with today's sole in-tree
    caller this returns ``[]`` unconditionally — ``_seed_pinned_from_installed``
    writes a key for EVERY in-scope host and ``_merge_egress_resolutions`` only
    ever copies/``setdefault``s that dict, never deletes a key, and carries an
    unread host forward at its last-good pins (``if not fresh: continue``). A
    20k-tick randomized simulation of the merge (resolve failures, empty
    resolves, rotations, whole-sidecar failure) produced zero non-empty
    results. **Keep-last-good upstream is the actual live protection**; this
    predicate is defence-in-depth on a public helper whose contract would
    otherwise turn a missing key into a delete.
    """
    return sorted((credential_hosts | limited_hosts) - set(new_ips))


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
    # Legacy per-address credential DNAT shape, kept ONLY as a delete target
    # (#2042): a session provisioned before name-based interception — or a
    # snapshot resumed across the upgrade — can still carry these, so the sweep
    # retires them as they age out. Byte-identical to the shape those sessions
    # installed so -D matches exactly. Nothing here ever ADDS one.
    legacy_dnat_tail = f"-p tcp --dport 443 -j DNAT --to-destination {proxy_ip}:{proxy_port}"

    def _category_ips(host_ips: dict[str, set[str]], hosts: set[str]) -> set[str]:
        return set().union(*(host_ips.get(host, set()) for host in hosts))

    # FAIL CLOSED on an incomplete inventory. An in-scope host ABSENT from
    # ``new_ips`` is a host whose IPs could not be READ; a host present with an
    # empty set is a host that genuinely owns NONE. Those are different facts,
    # and conflating them (``.get(host, set())``) drops the unread host's live
    # IPs into the ``old - new`` difference — so one transient/partial resolve
    # would DELETE firewall rules that are still in force. Deletions are
    # therefore refused entirely while any in-scope host is unread; adds are
    # unaffected because an add only ever widens what is already permitted.
    #
    # The SAME predicate is read by the caller (``_merge_egress_resolutions``),
    # which must also hold its ``pinned`` bookkeeping when it fires — see
    # :func:`egress_unread_hosts`.
    unread_hosts = egress_unread_hosts(
        new_ips=new_ips, credential_hosts=credential_hosts, limited_hosts=limited_hosts
    )

    old_credential_ips = _category_ips(old_ips, credential_hosts)
    new_credential_ips = _category_ips(new_ips, credential_hosts)
    old_limited_ips = _category_ips(old_ips, limited_hosts)
    new_limited_ips = _category_ips(new_ips, limited_hosts)

    lines = ["set -e", _IPTABLES_BACKEND_SELECT]
    for ip in sorted(new_limited_ips - old_limited_ips):
        lines.append(_add("", f"-d {ip} -p tcp --dport 80 -j ACCEPT"))
        lines.append(_add("", f"-d {ip} -p tcp --dport 443 -j ACCEPT"))
    # No credential DNAT add. Per-address DNAT churn — one rule per newly
    # sampled IP, dropped again when the address ages out — IS the sampling
    # machinery that let an unsampled address fail open (#2042). Interception
    # is keyed on the name now (one sentinel rule installed at provision), so
    # re-adding these here would quietly restore an IP-keyed variant of it.
    if unread_hosts:
        # Surfaced, not silent: the emitted script itself records why no
        # delete pass ran, so an operator reading the sidecar script sees the
        # refusal rather than an unexplained absence of deletions.
        lines.append(
            "# egress refresh: deletions REFUSED — incomplete host inventory "
            f"(unread: {' '.join(unread_hosts)})"
        )
        return "\n".join(lines)
    # The sentinel is NEVER a delete target. Inside the sandbox every
    # credential name resolves to it, so the stamp's read-back attributes the
    # one provisioned sentinel DNAT to the credential hosts and it lands in
    # ``pinned`` like any other address — and ``legacy_dnat_tail`` is
    # byte-identical to that rule, so a single tick whose resolve came back
    # without it would age the chokepoint out and delete it, with nothing here
    # ever adding it back. Excluding it by construction means no sweep can
    # retire name-based interception (#2042).
    for ip in sorted(old_credential_ips - new_credential_ips - {CREDENTIAL_SENTINEL_IP}):
        lines.append(_delete(" -t nat", f"-d {ip} {legacy_dnat_tail}"))
    for ip in sorted(old_limited_ips - new_limited_ips):
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

    A host that is BOTH an allowed host and a credential host gets no
    per-address filter ``ACCEPT`` (#2422) — see the comment on the allowed-host
    loop. Its only route out of the netns is sentinel → DNAT → proxy, so a
    direct-IP connection to a real address of it is refused by the terminal
    ``-P OUTPUT DROP`` instead of leaving un-proxied with the placeholder.

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

    # A credential host gets NO per-address filter ACCEPT (#2422). The sidecar
    # resolves allowed hosts here, BEFORE the :53 interception below is
    # installed, so for a host that is both allowed and credential-bearing this
    # loop would sample the host's REAL addresses and ACCEPT them on :443. No
    # legitimate in-sandbox client ever reaches those addresses — inside the
    # netns that NAME resolves only to the sentinel, whose route out is the
    # DNAT to the secret-egress proxy — so the only traffic such a rule can
    # admit is a client dialling a real credential-host address it holds out of
    # band, which leaves un-proxied carrying the literal placeholder. Without
    # the ACCEPT the terminal ``-P OUTPUT DROP`` refuses it.
    #
    # Unlike the Unrestricted path (:func:`build_secret_egress_dnat_script`)
    # this is a denial, not a redirect: the secret-egress proxy runs STRICT
    # under Limited (it refuses any SNI outside the credential set), so routing
    # every :443 through it would break egress to ordinary allowed hosts.
    #
    # NAMED RESIDUAL: an address SHARED with a different allowed host still
    # gets that host's ACCEPT, so a direct-IP connection to a credential host
    # co-tenanted on it remains un-proxied. Closing that needs the proxy to
    # carry the Limited allow-set so all :443 can be routed through it, which
    # is a proxy-side change, not a ruleset one.
    credential_hosts = (
        set(dnat_hosts)
        if dnat_target is not None and dnat_hosts and dns_port is not None
        else set()
    )
    for host in sorted(allowed_hosts - credential_hosts):
        lines.append("")
        lines.append(f"# Allow {host}")
        lines.append(f"ips=$(resolve_ipv4 {host})")
        lines.append(
            f"if [ -z \"$ips\" ]; then printf '%s\\t%s\\n' '{_EGRESS_SKIPPED_PREFIX}{host}' 'no IPv4 address'; "
            f"else echo '{_EGRESS_INSTALLED_PREFIX}{host}'; fi"
        )
        lines.append("for ip in $ips; do")
        lines.append('  "$IPT" -A OUTPUT -d "$ip" -p tcp --dport 80 -j ACCEPT')
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
    placeholder. No credential name resolves to a real address inside the
    sandbox any more: it resolves to the sentinel, whose only route out is the
    nat DNAT to the proxy, and whose non-``:443`` traffic is REJECTed.

    **That closes the resolving client; the catch-all below closes every other
    one (#2422).** Name-keyed interception is only reached by a client that
    ASKS — it rewrites what a name resolves to. A client that never asks, and
    dials a real credential-host address it holds out of band (a published
    GitHub API address, an address cached before provision, one read back from
    a DoH/DoT lookup this netns does not intercept), matched nothing here and
    left through the ``ACCEPT`` policy carrying the literal placeholder. So
    this path ALSO redirects **every** outbound ``tcp:443`` to the proxy:

        -t nat -A OUTPUT ! -d 127.0.0.0/8 -p tcp --dport 443 \
            -j DNAT --to-destination "$PROXY_IP:<proxy_port>"

    The proxy is the right place for that traffic to land because it is already
    name-keyed in the same way the netns rules are: it reads the host from the
    TLS ClientHello SNI, terminates and swaps ONLY for a credential host, and
    under Unrestricted blind-relays any other SNI to a worker-resolved,
    SSRF-checked, pinned upstream (``SecretEgressProxy._dispatch``). So general
    HTTPS egress stays open — it is relayed, not filtered — while the swap can
    no longer be skipped by choosing a destination address. **The destination
    address stops deciding anything; the name in the ClientHello decides, and
    the sandbox cannot present a credential host's name without reaching the
    swap.** A ``:443`` connection carrying NO SNI (an IP-literal HTTPS URL, or
    a non-TLS service on 443) is refused by the proxy rather than relayed —
    fail-closed, and the only case this narrows, deliberately: an SNI-less
    connection is exactly the one whose intent cannot be established.

    Loopback is excluded (:data:`_LOOPBACK_CIDR`) so an in-sandbox
    ``https://127.0.0.1`` service is not dragged out to the worker.

    IPv6 (:data:`_IP6TABLES_CREDENTIAL_HTTPS_DENY_LINES`) denies ``tcp:443``
    only — every rule above is IPv4 and the proxy binds IPv4, so a v6 route
    would otherwise reopen the identical bypass one stack down.

    The filter REJECT, the sentinel DNAT and the catch-all are all emitted
    UNCONDITIONALLY (no resolution loop), so unlike every previous shape their
    coverage does not depend on what DNS happened to return.

    Only the nat OUTPUT chain is flushed for idempotent re-apply — the filter
    OUTPUT chain is deliberately left untouched, except for the single sentinel
    REJECT this function must own. That REJECT (and the v6 ``:443`` DROP) is
    deleted-then-appended so a re-apply cannot stack duplicates.

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
            *_nat_dnat_lines(dnat_hosts, dnat_target, dns_port),
            "",
            "# Direct-IP catch-all (#2422). The sentinel rule above only covers a",
            "# client that ASKED DNS; one that dials a real credential-host address",
            "# it holds out of band matches nothing and leaves through the ACCEPT",
            "# policy with the literal placeholder. Send EVERY :443 to the proxy,",
            "# which keys on the ClientHello SNI (a name) and blind-relays a",
            "# non-credential one, so open egress survives but the destination",
            "# address no longer decides whether the swap fires.",
            f"\"$IPT\" -t nat -A OUTPUT '!' -d {_LOOPBACK_CIDR} -p tcp --dport 443 "
            f'-j DNAT --to-destination "$PROXY_IP:{dnat_target[1]}"',
            *_IP6TABLES_CREDENTIAL_HTTPS_DENY_LINES,
            # NO `-P OUTPUT DROP`, NO per-allowed-host filter ACCEPTs — the
            # filter policy stays ACCEPT so general egress remains open.
        ]
    )


# The internal / link-local / metadata / CGNAT destination ranges a browser
# container is denied outbound (jarbot#106). A deliberately NARROW, targeted L3
# subset for the SSRF / credential-theft threat — NOT the full internal-IP
# predicate in ``aios.tools.url_safety`` (which also blocks multicast, reserved,
# and TEST-NET; those are non-SSRF and unroutable from this isolated bridge, so
# they are intentionally omitted rather than carried as near-dead rules).
# Link-local 169.254.0.0/16 covers the cloud-metadata endpoint (169.254.169.254);
# the three RFC1918 blocks cover every private network INCLUDING the docker
# bridge gateway (so "reach the host via the gateway" is already denied — no
# separate gateway rule); 100.64.0.0/10 is CGNAT, which ``ipaddress`` does NOT
# fold into ``is_private``, so it must be listed explicitly. 127.0.0.0/8 is
# deliberately ABSENT: the container's embedded DNS resolver is 127.0.0.11
# (netns-local loopback) and the browser needs it to resolve the public web.
# This is a DESTINATION-IP filter: an internal service reachable on a PUBLIC IP
# is not covered here and must rely on its own auth (e.g. the API bearer key).
# IPv4-only: ``ensure_browser_network`` enforces the ``aios-browser`` network is
# ``--ipv6=false`` (hard-failing otherwise), so a browser container gets no v6
# address or route and v6 egress is impossible — no v6 rules needed.
_BROWSER_DENY_INTERNAL_CIDRS = (
    "169.254.0.0/16",
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "100.64.0.0/10",
)


def build_browser_deny_internal_script() -> str:
    """Build the L3 deny-internal egress script for a browser container.

    The browser renders UNTRUSTED web content, so — unlike the session/run
    lockdown, which is a default-DROP allow-list (:func:`build_iptables_script`)
    — general (public) egress must stay OPEN. This is the default-ACCEPT sibling
    of :func:`build_secret_egress_dnat_script`: it leaves the filter OUTPUT
    policy at ``ACCEPT`` (NO ``-P OUTPUT DROP``) and only appends targeted
    ``DROP`` rules for the ranges in :data:`_BROWSER_DENY_INTERNAL_CIDRS`. That
    closes cloud-metadata theft (169.254.169.254) and internal-service SSRF at
    L3 — on the resolved *destination IP*, so it holds against the DNS rebinding
    the driver's userspace navigate guard (navigate-time, hostname-based) cannot
    catch — while leaving the public web reachable.

    Only the filter OUTPUT chain is flushed (for idempotent re-apply); the nat
    table is untouched. Nothing here resolves a hostname: the rules are static
    CIDRs.
    """
    lines = [
        "set -e",
        "",
        _IPTABLES_BACKEND_SELECT,
        "",
        "# Clear filter OUTPUT (empty on the fresh container this always runs on),",
        "# leave the policy at ACCEPT (general egress stays open), do NOT touch nat.",
        '"$IPT" -F OUTPUT',
        "",
        "# Deny egress to the internal / link-local / metadata / CGNAT ranges; every",
        "# other destination (the public web) stays allowed by the ACCEPT policy.",
    ]
    lines += [f'"$IPT" -A OUTPUT -d {cidr} -j DROP' for cidr in _BROWSER_DENY_INTERNAL_CIDRS]
    return "\n".join(lines)


def build_browser_deny_internal_verify_script() -> str:
    """Read-back verify that every deny-internal DROP rule actually landed.

    Proof the rules took effect in the shared netns, not merely that the apply
    script exited 0 — the browser's analog of the ``-P OUTPUT DROP`` read-back
    in :func:`build_lockdown_verify_script`. ``set -e`` makes each missing-rule
    grep independently fatal. There is NO policy assertion: the deny-internal
    path deliberately leaves the filter policy at ``ACCEPT``. ``grep -F`` so the
    CIDR dots/slash are matched literally, not as a regexp.
    """
    lines = ["set -e", _IPTABLES_BACKEND_SELECT]
    lines += [
        f"\"$IPT\" -S OUTPUT | grep -qF -- '-d {cidr} -j DROP'"
        for cidr in _BROWSER_DENY_INTERNAL_CIDRS
    ]
    return "\n".join(lines)


# ``iptables -S`` does not echo the apply command back: it re-prints each rule
# through iptables' own formatter, and that formatter's spelling varies by
# backend and version. Two normalizations bite the read-back verify below:
#
#   * a single host address comes back ``/32``-canonicalized on most backends
#     (``-d 169.254.53.53`` applied → ``-d 169.254.53.53/32`` printed), and
#   * a ``--dport`` match is printed with the protocol match module it
#     implicitly loaded (``-p tcp --dport 53`` applied → ``-p tcp -m tcp
#     --dport 53`` printed).
#
# A verify grep written against the APPLY spelling therefore never matches the
# READ-BACK spelling — it fails on a correctly-installed ruleset, aborting the
# provision with the verify's (static) error text. These EREs tolerate both
# spellings of each field while still requiring every semantic field of the
# rule, so the read-back stays fail-closed. Same convention as
# ``registry.py``'s ``_EGRESS_RULE_RE``, which parses the same output.
_SENTINEL_RE = CREDENTIAL_SENTINEL_IP.replace(".", r"\.") + "(/32)?"
# Same escaping for the catch-all's negated loopback match. iptables prints a
# CIDR back verbatim (no /32 canonicalization to tolerate — it is already a
# prefix), so only the dots need quoting.
_LOOPBACK_CIDR_RE = _LOOPBACK_CIDR.replace(".", r"\.")


# Read-back assertion that the default OUTPUT policy is DROP — proves the
# lockdown actually took effect in the shared netns, not just that the apply
# script exited 0.
def build_lockdown_verify_script(
    dnat_hosts: Sequence[str] = (),
    *,
    dns_port: int | None = None,
    assert_drop: bool = True,
    assert_https_catch_all: bool = False,
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
    the worker-controlled resolver (udp AND tcp), the matching source NAT that
    lets Docker's loopback-sourced DNS packets cross the bridge (udp AND tcp),
    the sentinel ``:443`` DNAT to the secret-egress proxy, and the sentinel
    filter REJECT. Asserting merely
    that "some ``-j DNAT`` exists" (the pre-#2042 check) would pass on a
    half-installed chokepoint — DNS intercepted but the sentinel unrouted, or
    the reverse — which is a green verify over unprotected credential egress,
    the precise failure mode this issue exists to kill. Each assertion is
    independently fatal under ``set -e``, so a partial apply fails the
    provision instead of downgrading it silently. (This subsumes #984: a host
    that resolves to zero IPs is no longer even relevant, because no rule is
    keyed on a resolution any more.)

    Those read-back greps match the spelling ``iptables -S`` *prints*, which is not
    the spelling the apply script *wrote* (#2422) — see ``_SENTINEL_RE``. A
    grep written against the apply spelling fails on a correctly installed
    chokepoint, and because the callers' error text is static ("OUTPUT policy
    is not DROP", "nat OUTPUT carries no DNAT rule") it mis-reports which
    assertion failed.

    ``assert_https_catch_all`` is set by the DNAT-only Unrestricted caller
    (#2422) and reads back the catch-all that closes the direct-IP bypass: the
    ``! -d 127.0.0.0/8 -p tcp --dport 443 -j DNAT`` rule, plus (guarded, like
    the v6 assertions above) the v6 ``:443`` DROP. Left unverified these would
    be the "green verify while open" gap again — the sentinel rules can all
    land while the rule that covers a client which never asked DNS is missing,
    and the verdict on that sandbox is unprotected credential egress. It is a
    separate flag rather than ``not assert_drop`` so the Limited path can never
    acquire the assertion by accident: Limited closes the same bypass by
    WITHHOLDING an ACCEPT, and has no catch-all to read back.

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
        if dns_port is None:
            raise ValueError("dns_port is required when verifying credential-host interception")
        # Read back every rule that makes the name-based chokepoint real
        # (#2042). Asserting only "some DNAT exists" would pass on a ruleset
        # that intercepts DNS but never redirects the sentinel (or vice versa)
        # — i.e. green verify while credential egress is unprotected. Each is
        # independently fatal under ``set -e``.
        # Each grep is an ERE (``-qE``) matching the READ-BACK spelling of the
        # rule, not the apply spelling — see ``_SENTINEL_RE`` above for why the
        # two differ and why matching the apply spelling fails on a correctly
        # installed chokepoint.
        lines.append(
            '"$IPT" -t nat -S OUTPUT | grep -qE -- '
            f"'-d {_SENTINEL_RE} -p tcp( -m tcp)? --dport 443 -j DNAT'"
        )
        lines.append(
            "\"$IPT\" -t nat -S OUTPUT | grep -qE -- '-p udp( -m udp)? --dport 53 -j DNAT'"
        )
        lines.append(
            "\"$IPT\" -t nat -S OUTPUT | grep -qE -- '-p tcp( -m tcp)? --dport 53 -j DNAT'"
        )
        lines.append(
            f"\"$IPT\" -t nat -S POSTROUTING | grep -q -- '-j {_CREDENTIAL_DNS_SNAT_CHAIN}'"
        )
        for proto in ("udp", "tcp"):
            lines.append(
                f'"$IPT" -t nat -S {_CREDENTIAL_DNS_SNAT_CHAIN} | grep -qE -- '
                f"'-p {proto}( -m {proto})? --dport {dns_port} -j MASQUERADE'"
            )
        lines.append(f"\"$IPT\" -S OUTPUT | grep -qE -- '-d {_SENTINEL_RE} -j REJECT'")
    if assert_https_catch_all:
        # READ-BACK spelling again (see ``_SENTINEL_RE``): iptables re-prints
        # ``--dport`` with the protocol match module it implicitly loaded.
        lines.append(
            '"$IPT" -t nat -S OUTPUT | grep -qE -- '
            f"'! -d {_LOOPBACK_CIDR_RE} -p tcp( -m tcp)? --dport 443 -j DNAT'"
        )
        # Guarded exactly like the v6 assertion above: no v6 filter table means
        # the apply correctly skipped its DROP and there is nothing to read back.
        lines.append(_IP6TABLES_BACKEND_SELECT)
        lines.append(
            'if v6_output="$("$IP6T" -S OUTPUT 2>/dev/null)"; then '
            "printf '%s\\n' \"$v6_output\" | "
            "grep -qE -- '-p tcp( -m tcp)? --dport 443 -j DROP'; fi"
        )
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
) -> EgressProvisionResult:
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
    running the lockdown *inside* the sandbox (its own ``iptables``/``busybox``)
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
    apply_script = iptables_script
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
            script=build_lockdown_verify_script(dnat_hosts, dns_port=dns_port),
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
    return _parse_egress_provision_result(result.stdout)


async def apply_secret_egress_dnat(
    backend: SandboxBackend,
    handle: SandboxHandle,
    *,
    dnat_hosts: Sequence[str],
    dnat_target: tuple[str, int],
    dns_port: int,
    runtime: str | None = None,
) -> EgressProvisionResult:
    """Install the name-based credential chokepoint in an OPEN-egress sandbox (#1153).

    The Unrestricted sibling of :func:`apply_network_lockdown` (#1153): for an
    Unrestricted (or no-networking-config) environment that nonetheless carries
    env-var credentials, the secret swap must fire — but general egress stays
    open. So this runs the same operator-image netns sidecar with the same
    fail-closed posture, but applies :func:`build_secret_egress_dnat_script`
    (no lockdown; the filter OUTPUT policy is left at ``ACCEPT``) and verifies
    with ``assert_drop=False`` (assert the whole name-based chokepoint landed,
    but NOT a DROP policy, of which there is none) plus
    ``assert_https_catch_all=True`` (assert the direct-IP catch-all landed —
    #2422).

    **This is the path #2042 was filed against.** The interception installed
    here is keyed on NAMES, not on addresses a DNS sample happened to return,
    so a credential host resolving to an address no sampler ever saw is still
    proxied: inside this sandbox that name resolves ONLY to the sentinel, and
    the sentinel's only route out is the DNAT to the proxy.

    **And the path #2422's High was filed against.** Name-keying only binds a
    client that ASKS DNS; the script's catch-all ``:443`` DNAT binds the one
    that doesn't, by sending every HTTPS connection to the proxy, which keys on
    the ClientHello SNI and blind-relays a non-credential name so open egress
    survives.

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
    apply_script = build_secret_egress_dnat_script(dnat_hosts, dnat_target, dns_port)
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
            script=build_lockdown_verify_script(
                dnat_hosts,
                dns_port=dns_port,
                assert_drop=False,
                # Read back the direct-IP catch-all too (#2422) — the sentinel
                # rules can all land while the rule covering a client that
                # never asked DNS is missing, which is a green verify over
                # unprotected credential egress.
                assert_https_catch_all=True,
            ),
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
    return _parse_egress_provision_result(result.stdout)


async def apply_browser_deny_internal(backend: SandboxBackend, handle: SandboxHandle) -> None:
    """Apply + verify a browser container's L3 deny-internal egress (jarbot#106).

    Called right after the browser container is created. Runs the same
    operator-image netns sidecar as :func:`apply_network_lockdown` — the browser
    container itself holds no ``NET_ADMIN``, so root-in-container can neither
    flush nor poison the rules — but applies
    :func:`build_browser_deny_internal_script` (default-ACCEPT + targeted DROP)
    and verifies every DROP rule landed (there is no DROP *policy* to assert).

    Deliberately NOT factored into a shared helper with the session/run egress
    orchestrators (matching the :func:`apply_secret_egress_dnat` precedent): the
    browser path carries its own ``sandbox.browser_egress_*`` log events so an
    operator alert never mis-attributes a browser-plane egress failure to a
    session networking-policy violation.

    Takes NO ``runtime``: a browser is provisioned only under the default
    container runtime — the registry rejects a custom runtime before create,
    because this netns-sidecar iptables path does not initialize under runsc's
    netstack — so the sidecar always runs under the default runtime too.

    **Fails closed**: on a sidecar infra error, a nonzero apply, or a failed
    read-back verify, :class:`SandboxBackendError` propagates and the registry
    tears the just-created container down rather than handing back a browser
    whose untrusted web content can reach the cloud-metadata endpoint or
    internal services.
    """
    settings = get_settings()
    try:
        result = await backend.run_netns_sidecar(
            handle.sandbox_id,
            image=settings.docker_image,
            script=build_browser_deny_internal_script(),
            timeout_seconds=30,
            max_output_bytes=settings.bash_max_output_bytes,
        )
    except SandboxBackendError:
        log.warning("sandbox.browser_egress_sidecar_error", owner_id=handle.owner_id)
        raise

    if result.exit_code != 0:
        log.warning(
            "sandbox.browser_egress_failed",
            owner_id=handle.owner_id,
            exit_code=result.exit_code,
            stderr=result.stderr[:500],
        )
        raise SandboxBackendError(
            f"browser deny-internal egress failed (exit {result.exit_code}) for "
            f"{handle.owner_id}; refusing to run a browser whose untrusted web content "
            f"can reach internal/metadata endpoints"
        )

    try:
        verify = await backend.run_netns_sidecar(
            handle.sandbox_id,
            image=settings.docker_image,
            script=build_browser_deny_internal_verify_script(),
            timeout_seconds=15,
            max_output_bytes=settings.bash_max_output_bytes,
        )
    except SandboxBackendError:
        log.warning("sandbox.browser_egress_verify_error", owner_id=handle.owner_id)
        raise
    if verify.exit_code != 0:
        log.warning(
            "sandbox.browser_egress_verify_failed",
            owner_id=handle.owner_id,
            exit_code=verify.exit_code,
        )
        raise SandboxBackendError(
            f"browser deny-internal egress verification failed for {handle.owner_id}: "
            "an internal-range DROP rule is missing after apply; refusing to run a "
            "browser with unverified egress isolation"
        )

    log.info("sandbox.browser_egress_applied", owner_id=handle.owner_id)
