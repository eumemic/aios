"""The egress scripts must resolve through an explicitly-named DNS server.

Every hostname in a Limited allow-list, every credential host behind the
secret-egress DNAT and the proxy alias itself are turned into iptables rules by
one emitted shell function, ``setup.build_resolve_ipv4_fn``. Whatever answers it
decides which addresses get an ACCEPT (or a DNAT) rule, so *which resolver it
asks* is a security property, not a detail.

It used to ask glibc (``getent ahostsv4``), and glibc reads the servers out of
``/etc/resolv.conf`` and nothing else. Neither sidecar shape has a usable one:

* runc — Docker writes no ``resolv.conf`` for a ``--network container:<id>``
  sidecar, so it inherits the IMAGE's file;
* runsc — the exec chroots into the operator image mounted READ-ONLY, so it
  reads that image's file and nothing can write another one.

And the file cannot be baked either: BuildKit treats ``/etc/resolv.conf`` as
runtime-managed and commits an EMPTY entry for any ``COPY`` to that path —
observed on a CI-built image with a plain ``COPY`` and again with ``COPY
--link`` (aios#2410, moby/buildkit#1267; DONE.md carries the evidence chain).

So the resolver is passed as an argument instead: ``busybox nslookup <host>
127.0.0.11``.

That alone was not enough, because it also dropped the ``files`` half of the
lookup. ``getent`` read ``/etc/hosts`` before it read a nameserver; DNS-only
does not, and a ``--add-host`` alias lives ONLY in ``/etc/hosts`` — Docker
never publishes it to the embedded resolver. The sandbox is created with
``--add-host aios-worker:host-gateway``, so a DNS-only ``resolve_ipv4
aios-worker`` answers nothing, ``$PROXY_IP`` comes back empty, the guarded nat
block is skipped, and the apply exits 0 having installed no DNAT rule at all.
So the helper reads the hosts file FIRST and asks DNS only on a miss.

That hosts-first arm then had a tenant problem of its own. On the runc sidecar
shape the file it reads is the SANDBOX's, and root inside the sandbox can write
it — harmless while the lockdown is being applied (no tenant process exists
yet), not harmless on the periodic egress refresh, which runs once the tenant
owns the container and installs an ACCEPT for whatever the lookup returns. So
the two are now different arms of one resolver (``setup.ResolveScope``): both
consult an OPERATOR table baked in by the worker first, and only the provision
arm goes on to read the netns's hosts file. The operator table is also what
makes the ``--add-host`` alias resolvable under runsc at all, where the exec
chroots into the operator image and never sees the sandbox's file.

These are the source-level pins for all of that contract. The
end-to-end oracle — that the shipped image's busybox really answers this way
against Docker's embedded DNS — is ``tests/e2e/test_sandbox_image_contract.py``,
which needs a daemon.
"""

from __future__ import annotations

import stat
import subprocess
from pathlib import Path

import pytest

from aios.sandbox import setup
from aios.sandbox.backends.docker import _RUNSC_OPERATOR_STATIC_COMMANDS
from aios.sandbox.credential_dns import CREDENTIAL_SENTINEL_IP

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOCKERFILE = _REPO_ROOT / "docker" / "Dockerfile.sandbox"


def _scripts() -> dict[str, str]:
    """Every script `aios.sandbox.setup` hands to ``run_netns_sidecar``."""
    dnat_target = ("aios-worker", 49152)
    dns_port = 53535
    return {
        "limited_lockdown_apply": setup.build_iptables_script(
            {"example.com"},
            [("extra.example.com", 8080)],
            dnat_hosts=["api.secret.com"],
            dnat_target=dnat_target,
            dns_port=dns_port,
        ),
        "dnat_only_apply": setup.build_secret_egress_dnat_script(
            ["api.secret.com"], dnat_target, dns_port
        ),
        "egress_resolve_provision": setup.build_egress_resolve_script(
            ["example.com"], scope=setup.ResolveScope.PROVISION
        ),
        "egress_resolve_refresh": setup.build_egress_resolve_script(
            ["example.com"], scope=setup.ResolveScope.REFRESH
        ),
    }


@pytest.mark.parametrize("scope", list(setup.ResolveScope))
def test_resolver_is_named_explicitly(scope: setup.ResolveScope) -> None:
    """The emitted helper passes the embedded resolver's address as an argument."""
    helper = setup.build_resolve_ipv4_fn(scope=scope)
    assert f'busybox nslookup "$1" {setup._EMBEDDED_DNS_ADDRESS}' in helper


@pytest.mark.parametrize("name", sorted(_scripts()))
def test_no_script_touches_resolv_conf_or_getent(name: str) -> None:
    """Nothing on the lockdown path may depend on a resolver config FILE.

    A reintroduced ``getent`` silently resolves through whichever
    ``/etc/resolv.conf`` the sidecar happened to inherit; a reintroduced
    ``printf ... > /etc/resolv.conf`` preamble silently writes nothing under
    runsc (read-only operator mount) and is swallowed by its ``|| true``. Both
    failure modes are invisible until Limited egress blackholes in production.
    """
    script = _scripts()[name]
    assert "/etc/resolv.conf" not in script
    assert "getent" not in script


def test_busybox_is_shadowed_as_a_static_operator_command() -> None:
    """The runsc preamble must bind ``busybox`` to the operator image.

    ``resolve_ipv4`` runs inside the TENANT's mount namespace under runsc, so an
    unbound ``busybox`` resolves through ``PATH`` to a tenant-writable file that
    gets to choose every address the allow-list ends up containing. It belongs
    in the STATIC family: busybox carries no ``PT_INTERP``, so running it
    through the operator image's ``ld.so`` (as every dynamic command is) would
    fail outright.
    """
    assert "busybox" in _RUNSC_OPERATOR_STATIC_COMMANDS


def test_dockerfile_bakes_no_resolver() -> None:
    """No ``COPY`` to ``/etc/resolv.conf`` — BuildKit commits it empty.

    This is the regression guard for the fix: a well-meaning "the chroot needs a
    nameserver, just COPY one in" lands a 0-byte file in the layer, which reads
    as a resolver that exists and works right up until nothing resolves.
    """
    dockerfile = _DOCKERFILE.read_text()
    assert "/etc/resolv.conf" not in dockerfile, (
        "docker/Dockerfile.sandbox references /etc/resolv.conf again — BuildKit "
        "commits an EMPTY entry for any COPY to that path (aios#2410); the "
        "resolver is named explicitly in setup.build_resolve_ipv4_fn instead"
    )


# The real busybox 1.35 ``nslookup`` shape: the resolver's own address first,
# then the answers. ``$2`` is the queried name, ``$3`` the server.
_BUSYBOX_STUB = (
    "#!/bin/sh\n"
    '[ "$1" = nslookup ] || { echo "busybox: unknown applet $1" >&2; exit 1; }\n'
    'case "$2" in\n'
    "  missing.example.com)\n"
    '    printf \'Server:\\t\\t%s\\nAddress:\\t%s:53\\n\\n\' "$3" "$3"\n'
    "    echo \"nslookup: can't resolve '$2': Name or service not known\" >&2\n"
    "    exit 1;;\n"
    "esac\n"
    "printf 'Server:\\t\\t%s\\nAddress:\\t%s:53\\n\\nName:\\t%s\\n"
    'Address: 127.0.0.11\\nAddress: 203.0.113.7\\nAddress: 2001:db8::1\\nAddress: 198.51.100.9\\n\' "$3" "$3" "$2"\n'
)


def _resolve(
    host: str,
    tmp_path: Path,
    hosts: str | None = "",
    *,
    scope: setup.ResolveScope = setup.ResolveScope.PROVISION,
    operator_hosts: dict[str, str] | None = None,
) -> list[str]:
    """Run the REAL emitted helper against a stub busybox and a fixture hosts file.

    ``hosts`` is the ``/etc/hosts`` the helper sees; ``None`` writes no file at
    all. Retargeting ``setup._HOSTS_FILE`` is what keeps this hermetic — without
    it the helper would read the *test machine's* hosts file and the DNS
    assertions below would depend on whatever a developer or a CI image left
    there.
    """
    stub = tmp_path / "busybox"
    stub.write_text(_BUSYBOX_STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    hosts_file = tmp_path / "hosts"
    if hosts is not None:
        hosts_file.write_text(hosts)
    helper = setup.build_resolve_ipv4_fn(operator_hosts=operator_hosts or {}, scope=scope).replace(
        setup._HOSTS_FILE, str(hosts_file)
    )
    proc = subprocess.run(
        ["bash", "-c", f"set -e\n{helper}\nresolve_ipv4 {host}"],
        env={"PATH": f"{tmp_path}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.split()


def test_only_ipv4_answers_survive_the_parse(tmp_path: Path) -> None:
    """A answers only — never the AAAA, never the resolver's own address.

    The AAAA would be handed to an IPv4-only ``iptables -d``, which errors and,
    under ``set -e``, aborts the whole apply — a Limited sandbox that never
    finishes provisioning the moment an allowed host gains a v6 record. The
    server's address is worse: silently ACCEPTing 127.0.0.11, or picking it up
    as ``$PROXY_IP`` and DNATing every credential host at the resolver.
    """
    assert _resolve("example.com", tmp_path) == ["198.51.100.9", "203.0.113.7"]


def test_a_resolution_miss_yields_nothing(tmp_path: Path) -> None:
    """Fail closed: an unresolvable host gets no addresses, so it gets no rule."""
    assert _resolve("missing.example.com", tmp_path) == []


# --- hosts-first ------------------------------------------------------------
#
# The DNS oracles above all use names the stub resolver answers, so they cannot
# see the regression that motivated this round: a name that exists ONLY in
# ``/etc/hosts``. Every test below queries a name the stub busybox would either
# fail on or answer differently, so the hosts file is the only thing that can
# produce the assertion.

_ADD_HOST_LINE = "172.17.0.1\taios-worker\n"


def test_add_host_alias_resolves_without_dns(tmp_path: Path) -> None:
    """The regression, pinned: ``--add-host`` names resolve, DNS or no DNS.

    ``missing.example.com`` is the one name the stub resolver FAILS on, so a
    DNS-only helper returns nothing here — which is exactly what happened to
    ``aios-worker`` in the e2e host-gateway shape: Docker writes an
    ``--add-host`` alias into ``/etc/hosts`` and never publishes it to the
    embedded resolver at 127.0.0.11.
    """
    hosts = "10.0.0.9\tmissing.example.com\n"
    assert _resolve("missing.example.com", tmp_path, hosts) == ["10.0.0.9"]


def test_hosts_file_is_consulted_before_dns(tmp_path: Path) -> None:
    """Order, not just presence: a hosts entry SHORT-CIRCUITS the DNS query.

    The stub answers ``example.com`` with 203.0.113.7/198.51.100.9. Seeing only
    the hosts address proves the helper did not merge the two or ask DNS first
    — the same ``files`` before ``dns`` order glibc gives everything else in the
    container, so the firewall's idea of a name matches the container's.
    """
    assert _resolve("example.com", tmp_path, "10.0.0.9\texample.com\n") == ["10.0.0.9"]


def test_dns_still_answers_when_the_hosts_file_has_no_entry(tmp_path: Path) -> None:
    """Hosts-first is a PREFIX, not a replacement: an unlisted name falls through."""
    assert _resolve("example.com", tmp_path, _ADD_HOST_LINE) == ["198.51.100.9", "203.0.113.7"]


def test_dns_still_answers_when_there_is_no_hosts_file(tmp_path: Path) -> None:
    """A missing hosts file is a miss, never an abort.

    ``awk`` exits nonzero on an unreadable file; the caller runs under
    ``set -e``, so a lookup that propagated that status would abort the entire
    lockdown apply on a sidecar shape that happens to ship no ``/etc/hosts``.
    """
    assert _resolve("example.com", tmp_path, None) == ["198.51.100.9", "203.0.113.7"]


def test_hosts_aliases_and_comments_are_parsed_like_the_resolver_would(tmp_path: Path) -> None:
    """Canonical name and aliases both match; a commented-out line does not.

    A ``#`` comment that still parsed as an entry would install an ACCEPT for an
    address an operator believed they had removed.
    """
    hosts = "10.0.0.9\tcanonical.example.com alias.example.com\n# 10.0.0.250\tmissing.example.com\n"
    assert _resolve("alias.example.com", tmp_path, hosts) == ["10.0.0.9"]
    # Commented out, so the lookup misses and falls through to DNS — which the
    # stub fails for this name. An entry that survived its ``#`` would surface
    # here as 10.0.0.250.
    assert _resolve("missing.example.com", tmp_path, hosts) == []


def test_hosts_ipv6_entries_never_reach_the_ipv4_rules(tmp_path: Path) -> None:
    """The dotted-quad filter applies to the hosts half too.

    Every ``/etc/hosts`` carries ``::1 localhost``-shaped lines. An IPv6 literal
    from that half would be handed to an IPv4-only ``iptables -d``, which errors
    and aborts the apply under ``set -e`` — the same failure the DNS half's AAAA
    filter exists to prevent.
    """
    hosts = "::1\tmissing.example.com\nfe80::1\tmissing.example.com\n"
    assert _resolve("missing.example.com", tmp_path, hosts) == []


def test_partial_name_matches_are_not_answers(tmp_path: Path) -> None:
    """Fields are compared whole — a hosts entry may not widen a NEIGHBOURING name."""
    hosts = "10.0.0.9\tnot-missing.example.com missing.example.com.evil\n"
    assert _resolve("missing.example.com", tmp_path, hosts) == []


# ``iptables`` shim: record every mutating call, answer the script's guards.
_IPTABLES_RECORDER = (
    "#!/bin/sh\n"
    'printf "%s\\n" "$*" >> "$RULE_LOG"\n'
    'for a in "$@"; do case "$a" in -S) exit 0;; -C) exit 1;; -D) exit 1;; esac; done\n'
    "exit 0\n"
)


def test_hosts_only_proxy_alias_still_installs_the_dnat(tmp_path: Path) -> None:
    """The end of the causal chain the regression broke, at script level.

    ``aios-worker`` is the ``dnat_target`` alias and exists only in the hosts
    file (the stub resolver fails it, as Docker's embedded DNS does for an
    ``--add-host`` name). A DNS-only lookup leaves ``$PROXY_IP`` empty, and
    post-#2042 that is a HARD apply failure (``exit 1``) rather than a silently
    skipped nat block — either way the sandbox never gets its chokepoint.
    Asserting the installed rules, not the lookup, is what makes this an oracle
    for that failure rather than a restatement of the helper.
    """
    stub = tmp_path / "busybox"
    # The alias must be a name this resolver FAILS, or the DNS half could
    # satisfy the assertion on its own.
    stub.write_text(_BUSYBOX_STUB.replace("missing.example.com)", "aios-worker)"))
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    for name in ("iptables", "iptables-legacy", "ip6tables", "ip6tables-legacy"):
        shim = tmp_path / name
        shim.write_text(_IPTABLES_RECORDER)
        shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    hosts_file = tmp_path / "hosts"
    hosts_file.write_text(_ADD_HOST_LINE)
    rule_log = tmp_path / "rules.log"

    script = setup.build_secret_egress_dnat_script(
        ["api.secret.com"], ("aios-worker", 49152), 53535
    ).replace(setup._HOSTS_FILE, str(hosts_file))
    proc = subprocess.run(
        ["bash", "-c", script],
        env={"PATH": f"{tmp_path}:/usr/bin:/bin", "RULE_LOG": str(rule_log)},
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    rules = rule_log.read_text().splitlines()
    # Every DNAT is targeted at the proxy address the HOSTS file supplied — the
    # alias never reached DNS. Post-#2042 the credential host itself is not
    # resolved at all: the chokepoint is the :53 interception plus the sentinel
    # and direct-IP catch-all redirects, all keyed on ``$PROXY_IP``.
    proxy = "172.17.0.1"
    assert [r for r in rules if "DNAT" in r] == [
        f"-t nat -I OUTPUT -p udp --dport 53 -j DNAT --to-destination {proxy}:53535",
        f"-t nat -I OUTPUT -p tcp --dport 53 -j DNAT --to-destination {proxy}:53535",
        f"-t nat -A OUTPUT -d {CREDENTIAL_SENTINEL_IP} -p tcp --dport 443 "
        f"-j DNAT --to-destination {proxy}:49152",
        f"-t nat -A OUTPUT ! -d 127.0.0.0/8 -p tcp --dport 443 "
        f"-j DNAT --to-destination {proxy}:49152",
    ], rules


# --- operator table + resolve scope (aios#2410 fixround) --------------------
#
# The hosts-first arm above is correct input at PROVISION time and tenant input
# at REFRESH time, because the same file changes hands. These pin the two arms
# apart, and pin the operator table that sits in front of both.

_OPERATOR_TABLE = {"aios-worker": "192.168.65.2"}


def test_operator_table_answers_before_the_hosts_file(tmp_path: Path) -> None:
    """The worker-computed table wins over anything written inside the netns.

    This is the ordering the fix rests on. A tenant with root in the sandbox can
    rewrite ``/etc/hosts``; they cannot rewrite a ``case`` arm the worker baked
    into the script before the container existed. If a hosts entry could shadow
    the table, the refresh hardening below would be the only protection left and
    the runsc alias gap would reopen at provision time.
    """
    poisoned = "10.0.0.9\taios-worker\n"
    assert _resolve("aios-worker", tmp_path, poisoned, operator_hosts=_OPERATOR_TABLE) == [
        "192.168.65.2"
    ]


def test_operator_table_answers_before_dns(tmp_path: Path) -> None:
    """...and before the resolver, which does not know ``--add-host`` names.

    ``missing.example.com`` is the one name the stub resolver fails on, standing
    in for what the embedded DNS does with an ``--add-host`` alias: nothing.
    Under runsc that was the whole story — the operator chroot reads the
    operator image's hosts file, which has no alias either — so this is the
    lookup that used to leave ``$PROXY_IP`` empty and the credential DNAT
    uninstalled.
    """
    table = {"missing.example.com": "203.0.113.200"}
    assert _resolve("missing.example.com", tmp_path, None, operator_hosts=table) == [
        "203.0.113.200"
    ]


@pytest.mark.parametrize("scope", list(setup.ResolveScope))
def test_operator_table_answers_on_every_scope(tmp_path: Path, scope: setup.ResolveScope) -> None:
    """Both arms consult it — dropping the hosts file must not drop the alias.

    The refresh arm resolves fewer sources, not fewer operator names: an
    operator-supplied name that stopped resolving there would start evicting its
    own rules three ticks later.
    """
    assert _resolve("aios-worker", tmp_path, None, scope=scope, operator_hosts=_OPERATOR_TABLE) == [
        "192.168.65.2"
    ]


def test_refresh_scope_never_reads_the_hosts_file(tmp_path: Path) -> None:
    """THE HIGH, PINNED. A tenant hosts entry cannot steer a refreshed rule.

    Provision-time, ``10.0.0.9 example.com`` short-circuits DNS (the test above
    asserts exactly that, and it is correct there: the file is still what Docker
    wrote). By refresh time the tenant has owned the container for as long as
    the session has run, so the same entry is an attacker-chosen address for an
    operator-chosen name — and the refresh tick installs an ACCEPT for whatever
    comes back, widening the Limited allow-list to an address the operator never
    allowed. The refresh arm answers from DNS instead, so the write is inert.
    """
    poisoned = "10.0.0.9\texample.com\n"
    assert _resolve("example.com", tmp_path, poisoned, scope=setup.ResolveScope.REFRESH) == [
        "198.51.100.9",
        "203.0.113.7",
    ]


def test_refresh_scope_emits_no_hosts_file_read_at_all() -> None:
    """Not merely outranked — absent. The refresh script never names the file.

    An ordering assertion alone would still pass if the read happened and lost;
    this is the stronger statement, and the one a future edit to the resolver
    has to break deliberately.
    """
    refresh = setup.build_egress_resolve_script(["example.com"], scope=setup.ResolveScope.REFRESH)
    provision = setup.build_egress_resolve_script(
        ["example.com"], scope=setup.ResolveScope.PROVISION
    )
    assert setup._HOSTS_FILE not in refresh
    assert setup._HOSTS_FILE in provision


def test_the_operator_table_validates_what_it_bakes_in() -> None:
    """Both halves of every entry are checked, because both are interpolated.

    The name lands in a shell ``case`` pattern and the address lands in an
    ``iptables -d`` argument; an unvalidated AAAA literal in particular would
    abort the whole apply under ``set -e``.
    """
    with pytest.raises(ValueError):
        setup.build_resolve_ipv4_fn(
            operator_hosts={"aios-worker; rm -rf /": "10.0.0.1"},
            scope=setup.ResolveScope.PROVISION,
        )
    with pytest.raises(ValueError):
        setup.build_resolve_ipv4_fn(
            operator_hosts={"aios-worker": "2001:db8::1"},
            scope=setup.ResolveScope.PROVISION,
        )


def test_an_empty_operator_table_is_still_a_valid_script(tmp_path: Path) -> None:
    """The worker-in-container shape supplies no names and must still resolve.

    A shell function body may not be empty, so the emitted table degenerates to
    ``:`` rather than to a syntax error that would fail every apply on the
    deployment that needs the table least.
    """
    assert _resolve("example.com", tmp_path, None, operator_hosts={}) == [
        "198.51.100.9",
        "203.0.113.7",
    ]


def test_operator_table_proxy_alias_installs_the_dnat_without_hosts(tmp_path: Path) -> None:
    """THE MEDIUM, PINNED. The runsc chroot never sees ``--add-host``.

    Same oracle as ``test_hosts_only_proxy_alias_still_installs_the_dnat``, but
    the alias is supplied by the worker-baked table and the hosts file is
    poisoned: if the table lost, the apply would DNAT at 10.0.0.9 (or, with no
    hosts file and no DNS, hard-fail with an empty ``$PROXY_IP``). That is the
    runsc host-worker shape: operator-image ``/etc/hosts`` has no alias, Docker
    does not publish ``--add-host`` to the embedded DNS, and the credential
    redirect has nowhere to point unless the worker injects the address.
    """
    stub = tmp_path / "busybox"
    stub.write_text(_BUSYBOX_STUB.replace("missing.example.com)", "aios-worker)"))
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    for name in ("iptables", "iptables-legacy", "ip6tables", "ip6tables-legacy"):
        shim = tmp_path / name
        shim.write_text(_IPTABLES_RECORDER)
        shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    hosts_file = tmp_path / "hosts"
    hosts_file.write_text("10.0.0.9\taios-worker\n")
    rule_log = tmp_path / "rules.log"

    script = setup.build_secret_egress_dnat_script(
        ["api.secret.com"],
        ("aios-worker", 49152),
        53535,
        operator_hosts=_OPERATOR_TABLE,
    ).replace(setup._HOSTS_FILE, str(hosts_file))
    proc = subprocess.run(
        ["bash", "-c", script],
        env={"PATH": f"{tmp_path}:/usr/bin:/bin", "RULE_LOG": str(rule_log)},
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    rules = rule_log.read_text().splitlines()
    proxy = "192.168.65.2"
    assert [r for r in rules if "DNAT" in r] == [
        f"-t nat -I OUTPUT -p udp --dport 53 -j DNAT --to-destination {proxy}:53535",
        f"-t nat -I OUTPUT -p tcp --dport 53 -j DNAT --to-destination {proxy}:53535",
        f"-t nat -A OUTPUT -d {CREDENTIAL_SENTINEL_IP} -p tcp --dport 443 "
        f"-j DNAT --to-destination {proxy}:49152",
        f"-t nat -A OUTPUT ! -d 127.0.0.0/8 -p tcp --dport 443 "
        f"-j DNAT --to-destination {proxy}:49152",
    ], rules
    assert not any("10.0.0.9" in r for r in rules), rules
