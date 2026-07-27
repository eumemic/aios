"""Unit tests for networking model validation and sandbox lockdown logic."""

from __future__ import annotations

import asyncio
import os
import socket
import struct
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aios.models.environments import (
    EnvironmentConfig,
    LimitedNetworking,
    UnrestrictedNetworking,
)
from aios.sandbox.backends.base import (
    CommandResult,
    Mount,
    SandboxBackendError,
    SandboxSpec,
)
from aios.sandbox.backends.docker import DockerBackend
from aios.sandbox.credential_dns import CREDENTIAL_SENTINEL_IP, CredentialDnsResolver
from aios.sandbox.setup import (
    apply_network_lockdown,
    apply_secret_egress_dnat,
    build_iptables_script,
    build_lockdown_verify_script,
    build_secret_egress_dnat_script,
)
from tests.helpers.sandbox import FakeBackend, make_handle

# ── model validation ──────────────────────────────────────────────────────────


class TestUnrestrictedNetworking:
    def test_round_trip(self) -> None:
        n = UnrestrictedNetworking()
        assert n.type == "unrestricted"
        assert n.model_dump() == {"type": "unrestricted"}

    def test_from_dict(self) -> None:
        n = UnrestrictedNetworking.model_validate({"type": "unrestricted"})
        assert n.type == "unrestricted"


class TestLimitedNetworking:
    def test_round_trip(self) -> None:
        n = LimitedNetworking(
            type="limited",
            allowed_hosts=["api.example.com", "cdn.example.com"],
            allow_package_managers=True,
        )
        assert n.type == "limited"
        assert n.allowed_hosts == ["api.example.com", "cdn.example.com"]
        assert n.allow_package_managers is True

    def test_defaults(self) -> None:
        n = LimitedNetworking(type="limited")
        assert n.allowed_hosts == []
        assert n.allow_package_managers is False

    def test_rejects_empty_hostname(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            LimitedNetworking(type="limited", allowed_hosts=[""])

    def test_rejects_hostname_with_protocol(self) -> None:
        with pytest.raises(ValueError, match="invalid hostname"):
            LimitedNetworking(type="limited", allowed_hosts=["https://example.com"])

    def test_rejects_hostname_with_shell_metacharacters(self) -> None:
        with pytest.raises(ValueError, match="invalid hostname"):
            LimitedNetworking(type="limited", allowed_hosts=["example.com; rm -rf /"])

    def test_rejects_hostname_with_trailing_newline(self) -> None:
        # The hostname regex is anchored ^...$; `re.match` would forgive a
        # single trailing newline and let it through into the iptables script.
        with pytest.raises(ValueError, match="invalid hostname"):
            LimitedNetworking(type="limited", allowed_hosts=["example.com\n"])

    def test_rejects_hostname_with_path(self) -> None:
        with pytest.raises(ValueError, match="invalid hostname"):
            LimitedNetworking(type="limited", allowed_hosts=["example.com/path"])

    def test_rejects_hostname_too_long(self) -> None:
        with pytest.raises(ValueError, match="hostname too long"):
            LimitedNetworking(type="limited", allowed_hosts=["a" * 254])

    def test_accepts_valid_hostnames(self) -> None:
        hosts = ["example.com", "sub.domain.co.uk", "a-b-c.example.org", "123.45.67.89"]
        n = LimitedNetworking(type="limited", allowed_hosts=hosts)
        assert n.allowed_hosts == hosts


class TestEnvironmentConfigNetworking:
    def test_defaults_to_none(self) -> None:
        config = EnvironmentConfig()
        assert config.networking is None

    def test_unrestricted_round_trip(self) -> None:
        config = EnvironmentConfig(networking=UnrestrictedNetworking())
        assert isinstance(config.networking, UnrestrictedNetworking)

    def test_limited_round_trip(self) -> None:
        config = EnvironmentConfig(
            networking=LimitedNetworking(
                type="limited",
                allowed_hosts=["api.example.com"],
                allow_package_managers=True,
            )
        )
        assert isinstance(config.networking, LimitedNetworking)
        assert config.networking.allowed_hosts == ["api.example.com"]


# ── iptables script construction ──────────────────────────────────────────────


class TestBuildIptablesScript:
    def test_drops_everything_else(self) -> None:
        script = build_iptables_script(allowed_hosts={"api.example.com"})
        assert '"$IPT" -P OUTPUT DROP' in script

    def test_includes_each_allowed_host(self) -> None:
        script = build_iptables_script(allowed_hosts={"api.example.com", "cdn.example.com"})
        assert "resolve_ipv4 api.example.com" in script
        assert "resolve_ipv4 cdn.example.com" in script

    def test_flushes_filter_output_chain(self) -> None:
        script = build_iptables_script(allowed_hosts={"api.example.com"})
        assert '"$IPT" -F OUTPUT' in script

    def test_flushes_nat_output_chain_for_idempotent_reapply(self) -> None:
        """#984: a future re-apply path (e.g. credential rotation refreshing the
        lockdown without a full netns recycle) would accumulate duplicate DNAT
        entries unless the nat OUTPUT chain is flushed alongside the filter
        chain. Flushing both makes re-apply idempotent."""
        script = build_iptables_script(
            allowed_hosts={"api.example.com"},
            dnat_hosts=["api.secret.com"],
            dnat_target=("aios-worker", 49152),
            dns_port=53535,
        )
        assert '"$IPT" -t nat -F OUTPUT' in script
        # The nat flush precedes the DNAT rules, so re-running the script
        # replaces (not appends to) the existing nat OUTPUT chain.
        assert script.index('"$IPT" -t nat -F OUTPUT') < script.index("-j DNAT")

    def test_loopback_and_dns_always_allowed(self) -> None:
        script = build_iptables_script(allowed_hosts=set())
        assert '"$IPT" -A OUTPUT -o lo -j ACCEPT' in script
        assert '"$IPT" -A OUTPUT -p udp --dport 53 -j ACCEPT' in script
        assert '"$IPT" -A OUTPUT -p tcp --dport 53 -j ACCEPT' in script

    def test_extra_host_ports_added(self) -> None:
        script = build_iptables_script(
            allowed_hosts=set(),
            extra_host_ports=[("aios-worker", 8765)],
        )
        assert "aios-worker:8765" in script
        assert "--dport 8765 -j ACCEPT" in script

    # ── legacy-vs-nft backend selection (#1022, gVisor netstack) ──────────────

    def test_selects_legacy_binary_when_present(self) -> None:
        """gVisor's netstack implements legacy netfilter, NOT nftables, but
        debian/ubuntu images default ``iptables`` to the nft backend. The
        sidecar script must prefer ``iptables-legacy`` when it is installed
        (and fall back to ``iptables`` on hosts whose image lacks it)."""
        script = build_iptables_script(allowed_hosts={"api.example.com"})
        # The preamble detects the legacy binary and falls back to plain iptables.
        assert "command -v iptables-legacy" in script
        # Once the binary is selected, every rule invokes the SELECTED binary
        # via the shell variable — never a bare ``iptables`` command.
        assert '"$IPT" -P OUTPUT DROP' in script
        assert '"$IPT" -F OUTPUT' in script
        assert '"$IPT" -A OUTPUT -o lo -j ACCEPT' in script

    def test_no_bare_iptables_calls(self) -> None:
        """Every netfilter command goes through the ``$IPT`` selector so apply
        and verify agree on the backend; a bare ``iptables ...`` call
        (line start or after a guard) would silently use the nft default."""
        script = build_iptables_script(
            allowed_hosts={"api.example.com"},
            dnat_hosts=["api.secret.com"],
            dnat_target=("aios-worker", 49152),
            dns_port=53535,
        )
        for line in script.splitlines():
            stripped = line.strip()
            # The detection preamble names the binaries; that's expected.
            if "command -v iptables-legacy" in stripped:
                continue
            assert not stripped.startswith("iptables "), (
                f"bare iptables call would use the nft default: {line!r}"
            )

    # ── nat-table DNAT to the secret-egress proxy (#878) ──────────────────────

    def test_sentinel_dnat_rule_emitted_on_443(self) -> None:
        """ONE credential DNAT, keyed on the sentinel — not one per resolved IP.

        Post-#2042 the credential hosts no longer appear in a resolution loop
        at all: every credential name resolves to
        :data:`CREDENTIAL_SENTINEL_IP` inside the sandbox, so a single rule
        covers all of them completely.
        """
        script = build_iptables_script(
            allowed_hosts=set(),
            dnat_hosts=["api.secret.com", "data.secret.com"],
            dnat_target=("aios-worker", 49152),
            dns_port=53535,
        )
        assert '"$IPT" -t nat -A OUTPUT' in script
        assert (
            f"-d {CREDENTIAL_SENTINEL_IP} -p tcp --dport 443 "
            '-j DNAT --to-destination "$PROXY_IP:49152"'
        ) in script
        assert "PROXY_IP=$(resolve_ipv4 aios-worker" in script
        # No credential host is resolved to generate rules any more — that
        # sampling IS the #2042 defect.
        assert "resolve_ipv4 api.secret.com" not in script
        assert "resolve_ipv4 data.secret.com" not in script

    def test_dns_intercepted_to_worker_resolver(self) -> None:
        """All sandbox :53 is redirected to this session's resolver, inserted at
        the TOP of nat OUTPUT so no in-netns resolver answers a credential name
        first (#2042)."""
        script = build_iptables_script(
            allowed_hosts=set(),
            dnat_hosts=["api.secret.com"],
            dnat_target=("aios-worker", 49152),
            dns_port=53535,
        )
        for proto in ("udp", "tcp"):
            assert (
                f'"$IPT" -t nat -I OUTPUT -p {proto} --dport 53 -j DNAT '
                '--to-destination "$PROXY_IP:53535"'
            ) in script

    def test_proxy_alias_miss_is_a_hard_failure(self) -> None:
        """A ``$PROXY_IP`` miss must abort the apply, not skip the block.

        The old code guarded the whole nat section behind ``if [ -n
        "$PROXY_IP" ]``, so an alias miss silently produced a credentialed
        sandbox with NO interception and every request went to the real
        upstream carrying the literal placeholder. Fail closed instead.
        """
        script = build_iptables_script(
            allowed_hosts=set(),
            dnat_hosts=["api.secret.com"],
            dnat_target=("aios-worker", 49152),
            dns_port=53535,
        )
        assert 'if [ -z "$PROXY_IP" ]; then' in script
        assert "  exit 1" in script

    def test_dnat_not_redirect(self) -> None:
        script = build_iptables_script(
            allowed_hosts=set(),
            dnat_hosts=["api.secret.com", "data.secret.com"],
            dnat_target=("aios-worker", 49152),
            dns_port=53535,
        )
        assert "DNAT" in script
        assert "--to-port" not in script
        assert "REDIRECT" not in script

    def test_no_nat_rules_without_dnat_target(self) -> None:
        # The unconditional nat flush is always present (#984 idempotency); what
        # must be absent without a dnat_target is the DNAT rule *addition*.
        script = build_iptables_script(
            allowed_hosts={"api.example.com"},
            dnat_hosts=["api.secret.com"],
        )
        assert "-t nat -A OUTPUT" not in script
        assert "DNAT" not in script

    def test_no_nat_rules_without_dnat_hosts(self) -> None:
        script = build_iptables_script(
            allowed_hosts=set(),
            dnat_hosts=[],
            dnat_target=("aios-worker", 49152),
            dns_port=53535,
        )
        assert "-t nat -A OUTPUT" not in script
        assert "DNAT" not in script

    def test_existing_callers_unchanged(self) -> None:
        script = build_iptables_script(allowed_hosts={"api.example.com"})
        assert "-t nat -A OUTPUT" not in script
        assert "DNAT" not in script
        assert '"$IPT" -P OUTPUT DROP' in script

    def test_allowed_host_gets_accept_not_dnat(self) -> None:
        script = build_iptables_script(
            allowed_hosts={"plain.example.com"},
            dnat_hosts=["api.secret.com"],
            dnat_target=("aios-worker", 49152),
            dns_port=53535,
        )
        assert '"$IPT" -A OUTPUT -d "$ip" -p tcp --dport 443 -j ACCEPT' in script
        assert "resolve_ipv4 plain.example.com" in script
        # The plain allowed host is never DNAT'd. Three DNATs are expected now
        # (#2042): udp/tcp :53 to the resolver + the single sentinel :443.
        assert script.count("-j DNAT --to-destination") == 3
        assert '-d "$ip" -p tcp --dport 443 -j DNAT' not in script

    def test_dnat_only_on_443_not_80(self) -> None:
        script = build_iptables_script(
            allowed_hosts=set(),
            dnat_hosts=["api.secret.com", "data.secret.com"],
            dnat_target=("aios-worker", 49152),
            dns_port=53535,
        )
        assert "--dport 80 -j DNAT" not in script


# ── IPv6 belt-and-suspenders egress DROP (#1207) ──────────────────────────────


class TestIPv6EgressLockdown:
    """The Limited lockdown mirrors the v4 ``-P OUTPUT DROP`` on ip6tables so
    the IPv4-only lockdown cannot be bypassed over IPv6 if a v6 route ever
    appears (network recreated with ``--ipv6``, or a Docker default flips)."""

    def test_emits_ip6tables_output_drop(self) -> None:
        script = build_iptables_script(allowed_hosts={"api.example.com"})
        assert '"$IP6T" -P OUTPUT DROP' in script

    def test_v6_block_guarded_on_table_availability(self) -> None:
        """#1207 fix: the v6 flush/loopback/DROP must NOT abort the whole apply
        under ``set -e`` when the ``ip6_tables`` kernel module is absent (the v6
        ``filter`` table cannot initialize — common on CI runners and any
        IPv6-disabled host). A missing v6 netfilter table means there is no v6
        egress path to leak through, so the block is skipped, not fatal; when the
        table IS present the DROP is enforced. Without this guard every Limited
        provision in the docker e2e shard fails closed on such hosts."""
        script = build_iptables_script(allowed_hosts={"api.example.com"})
        # The v6 rules run only inside an ``if "$IP6T" -S OUTPUT`` guard.
        assert 'if "$IP6T" -S OUTPUT >/dev/null 2>&1; then' in script
        guard_idx = script.index('if "$IP6T" -S OUTPUT >/dev/null 2>&1; then')
        drop_idx = script.index('"$IP6T" -P OUTPUT DROP')
        assert guard_idx < drop_idx, "v6 DROP must be inside the table-available guard"
        # There must be an else-branch that does not abort (no bare exit/false).
        assert "\nelse\n" in script or "\nelse \n" in script or "else" in script

    def test_flushes_ip6tables_output_chain(self) -> None:
        script = build_iptables_script(allowed_hosts={"api.example.com"})
        assert '"$IP6T" -F OUTPUT' in script
        # The flush precedes the DROP policy so re-apply is idempotent.
        assert script.index('"$IP6T" -F OUTPUT') < script.index('"$IP6T" -P OUTPUT DROP')

    def test_allows_v6_loopback(self) -> None:
        """Total v6 egress denial, but loopback stays open so any in-netns
        v6 localhost/DNS still works (the spec's 'flush + DROP with loopback
        allowed' form)."""
        script = build_iptables_script(allowed_hosts={"api.example.com"})
        assert '"$IP6T" -A OUTPUT -o lo -j ACCEPT' in script
        # Loopback ACCEPT precedes the DROP policy (rules are order-independent
        # for a policy, but the ACCEPT rule must exist alongside it).
        assert script.index('"$IP6T" -A OUTPUT -o lo -j ACCEPT') < script.index(
            '"$IP6T" -P OUTPUT DROP'
        )

    def test_selects_legacy_ip6tables_backend(self) -> None:
        """runsc's netstack speaks the legacy netfilter ABI, not nft. A bare
        ``ip6tables`` under ``set -e`` would error on runsc and abort the whole
        lockdown apply, failing every Limited provision closed-noisily. So the
        v6 path mirrors the v4 legacy-backend selection."""
        script = build_iptables_script(allowed_hosts={"api.example.com"})
        assert "command -v ip6tables-legacy" in script

    def test_no_bare_ip6tables_calls(self) -> None:
        """Every v6 netfilter command goes through the ``$IP6T`` selector so it
        never silently uses the nft default (which fails on runsc)."""
        script = build_iptables_script(
            allowed_hosts={"api.example.com"},
            dnat_hosts=["api.secret.com"],
            dnat_target=("aios-worker", 49152),
            dns_port=53535,
        )
        for line in script.splitlines():
            stripped = line.strip()
            # The detection preamble names the binaries; that's expected.
            if "command -v ip6tables-legacy" in stripped:
                continue
            assert not stripped.startswith("ip6tables "), (
                f"bare ip6tables call would use the nft default: {line!r}"
            )

    def test_v6_drop_present_with_dnat(self) -> None:
        """The v6 DROP is on the lockdown (filter-DROP) path regardless of
        credential DNAT, which only touches the v4 nat table."""
        script = build_iptables_script(
            allowed_hosts={"plain.example.com"},
            extra_host_ports=[("aios-worker", 8765)],
            dnat_hosts=["api.secret.com"],
            dnat_target=("aios-worker", 49152),
            dns_port=53535,
        )
        assert '"$IP6T" -P OUTPUT DROP' in script

    def test_dnat_only_script_has_no_v6_drop(self) -> None:
        """The Unrestricted DNAT-only path leaves general egress open, so it
        must NOT install a v6 DROP (which would deny v6 egress in an otherwise
        open box)."""
        script = build_secret_egress_dnat_script(
            dnat_hosts=["api.secret.com"],
            dnat_target=("aios-worker", 49152),
            dns_port=53535,
        )
        assert "ip6tables" not in script
        assert "-P OUTPUT DROP" not in script


# ── IPv4-only host resolution (#978) ──────────────────────────────────────────


class TestIPv4OnlyResolution:
    """Every host lookup in the lockdown scripts must resolve IPv4-only.

    ``getent ahosts`` returns BOTH A and AAAA records; the emitted rules are
    all IPv4 ``iptables`` commands and the script runs under ``set -e``, so an
    AAAA literal fed to ``iptables -d`` would error and abort the whole apply
    the moment an IPv6-capable sandbox network is enabled. Resolving with
    ``getent ahostsv4`` keeps only A records flowing into the IPv4 rules; IPv6
    egress is left to the default DROP policy (fail-closed). The proxy binds the
    IPv4 ``WORKER_NETWORK_ALIAS`` and cannot intercept IPv6, so IPv4-only DNAT
    is also the correct credential-host semantics.
    """

    def _all_scripts(self) -> dict[str, str]:
        return {
            "lockdown_plain": build_iptables_script(allowed_hosts={"api.example.com"}),
            "lockdown_extra_ports": build_iptables_script(
                allowed_hosts=set(), extra_host_ports=[("aios-worker", 8765)]
            ),
            "lockdown_dnat": build_iptables_script(
                allowed_hosts={"plain.example.com"},
                extra_host_ports=[("aios-worker", 8765)],
                dnat_hosts=["api.secret.com"],
                dnat_target=("aios-worker", 49152),
                dns_port=53535,
            ),
            "dnat_only": build_secret_egress_dnat_script(
                dnat_hosts=["api.secret.com"],
                dnat_target=("aios-worker", 49152),
                dns_port=53535,
            ),
        }

    def test_no_dual_stack_getent_ahosts(self) -> None:
        """No script may use the dual-stack ``getent ahosts`` (which also
        returns AAAA); every lookup must use the IPv4-only ``getent ahostsv4``.
        """
        for name, script in self._all_scripts().items():
            for line in script.splitlines():
                # ``ahostsv4`` contains ``ahosts`` as a substring, so match the
                # exact dual-stack token (``ahosts`` followed by a space).
                assert "getent ahosts " not in line, (
                    f"{name}: dual-stack getent leaks AAAA into IPv4 rules: {line!r}"
                )

    def test_helper_resolves_ipv4_only(self) -> None:
        """The shared helper is defined and uses ``getent ahostsv4``."""
        for name, script in self._all_scripts().items():
            if "resolve_ipv4 " not in script:
                continue
            assert "resolve_ipv4()" in script, f"{name}: helper used but not defined"
            assert "getent ahostsv4" in script, f"{name}: helper is not IPv4-only"

    def test_allowed_host_loop_uses_helper(self) -> None:
        script = build_iptables_script(allowed_hosts={"api.example.com"})
        assert "for ip in $(resolve_ipv4 api.example.com); do" in script
        assert "getent ahostsv4" in script

    def test_extra_host_ports_loop_uses_helper(self) -> None:
        script = build_iptables_script(
            allowed_hosts=set(), extra_host_ports=[("aios-worker", 8765)]
        )
        assert "for ip in $(resolve_ipv4 aios-worker); do" in script

    def test_proxy_alias_uses_helper_and_credential_hosts_are_not_resolved(self) -> None:
        """The proxy ALIAS is still resolved (iptables needs an IP for
        ``--to-destination``); the credential HOSTS no longer are (#2042)."""
        script = build_iptables_script(
            allowed_hosts=set(),
            dnat_hosts=["api.secret.com"],
            dnat_target=("aios-worker", 49152),
            dns_port=53535,
        )
        assert "PROXY_IP=$(resolve_ipv4 aios-worker | head -n1)" in script
        assert "resolve_ipv4 api.secret.com" not in script

    def test_dnat_only_script_defines_and_uses_helper(self) -> None:
        script = build_secret_egress_dnat_script(
            dnat_hosts=["api.secret.com"],
            dnat_target=("aios-worker", 49152),
            dns_port=53535,
        )
        assert "resolve_ipv4()" in script
        assert "getent ahostsv4" in script
        # Only the proxy alias is resolved; the credential host is not.
        assert "PROXY_IP=$(resolve_ipv4 aios-worker | head -n1)" in script
        assert "resolve_ipv4 api.secret.com" not in script

    def test_helper_defined_before_first_use(self) -> None:
        """The helper definition must precede every call so the script runs
        under ``set -e`` without a 'command not found'."""
        for name, script in self._all_scripts().items():
            if "resolve_ipv4 " not in script:
                continue
            assert script.index("resolve_ipv4()") < script.index("$(resolve_ipv4 "), (
                f"{name}: helper called before it is defined"
            )

    def test_helper_emitted_once_per_script(self) -> None:
        """The dedup'd helper is emitted exactly once (centralizes the fix)."""
        for name, script in self._all_scripts().items():
            assert script.count("resolve_ipv4()") <= 1, f"{name}: helper defined twice"


# ── DNAT-only Unrestricted swap chokepoint (no lockdown, #1153) ───────────────


class TestBuildSecretEgressDnatScript:
    """The Unrestricted DNAT-only script installs the credential-host → proxy
    swap chokepoint while leaving general egress open (no ``-P OUTPUT DROP``)."""

    def test_emits_sentinel_dnat_and_dns_interception(self) -> None:
        """The Unrestricted chokepoint: :53 to the resolver, sentinel :443 to
        the proxy — and NO per-credential-host resolution (#2042)."""
        script = build_secret_egress_dnat_script(
            dnat_hosts=["api.secret.com", "data.secret.com"],
            dnat_target=("aios-worker", 49152),
            dns_port=53535,
        )
        assert (
            f'"$IPT" -t nat -A OUTPUT -d {CREDENTIAL_SENTINEL_IP} -p tcp --dport 443 '
            '-j DNAT --to-destination "$PROXY_IP:49152"'
        ) in script
        for proto in ("udp", "tcp"):
            assert (
                f'"$IPT" -t nat -I OUTPUT -p {proto} --dport 53 -j DNAT '
                '--to-destination "$PROXY_IP:53535"'
            ) in script
        assert "PROXY_IP=$(resolve_ipv4 aios-worker" in script
        assert "route_localnet" not in script
        assert f"ip address replace {CREDENTIAL_SENTINEL_IP}/32 dev lo" in script
        assert script.index("ip address replace") < script.index(
            f"-d {CREDENTIAL_SENTINEL_IP} -p tcp --dport 443"
        )
        assert "ip route replace" not in script
        assert "resolve_ipv4 api.secret.com" not in script
        assert "resolve_ipv4 data.secret.com" not in script

    def test_sentinel_reject_is_unconditional(self) -> None:
        """The fail-closed filter REJECT for the sentinel is emitted flat — not
        inside a resolution loop — so its coverage cannot depend on what DNS
        returned (the #2042 defect in its previous, IP-keyed form)."""
        script = build_secret_egress_dnat_script(
            dnat_hosts=["api.secret.com"],
            dnat_target=("aios-worker", 49152),
            dns_port=53535,
        )
        reject = (
            f'"$IPT" -A OUTPUT -d {CREDENTIAL_SENTINEL_IP} -j REJECT '
            "--reject-with icmp-port-unreachable"
        )
        assert reject in script
        # Not nested in any `for ip in $(...)` loop.
        assert "for ip in" not in script

    def test_no_drop_policy(self) -> None:
        # The whole point: general egress stays open under Unrestricted.
        script = build_secret_egress_dnat_script(
            dnat_hosts=["api.secret.com"],
            dnat_target=("aios-worker", 49152),
            dns_port=53535,
        )
        assert "-P OUTPUT DROP" not in script

    def test_no_filter_accept_rules(self) -> None:
        # No per-host filter ACCEPTs and no loopback/DNS/established ACCEPTs —
        # the filter policy is left at its default ACCEPT (no lockdown).
        script = build_secret_egress_dnat_script(
            dnat_hosts=["api.secret.com"],
            dnat_target=("aios-worker", 49152),
            dns_port=53535,
        )
        assert "-A OUTPUT -o lo -j ACCEPT" not in script
        assert "--dport 53 -j ACCEPT" not in script
        assert "ESTABLISHED,RELATED -j ACCEPT" not in script
        assert "--dport 443 -j ACCEPT" not in script

    def test_flushes_nat_output_only_not_filter(self) -> None:
        script = build_secret_egress_dnat_script(
            dnat_hosts=["api.secret.com"],
            dnat_target=("aios-worker", 49152),
            dns_port=53535,
        )
        assert '"$IPT" -t nat -F OUTPUT' in script
        # The filter OUTPUT chain is deliberately NOT flushed (would disturb a
        # mode the operator left open).
        assert '"$IPT" -F OUTPUT' not in script

    def test_uses_selected_backend_no_bare_iptables(self) -> None:
        script = build_secret_egress_dnat_script(
            dnat_hosts=["api.secret.com"],
            dnat_target=("aios-worker", 49152),
            dns_port=53535,
        )
        assert "command -v iptables-legacy" in script
        for line in script.splitlines():
            stripped = line.strip()
            if "command -v iptables-legacy" in stripped:
                continue
            assert not stripped.startswith("iptables "), (
                f"bare iptables call would use the nft default: {line!r}"
            )

    def test_dnat_rule_shape_matches_lockdown_script(self) -> None:
        # The shared ``_nat_dnat_lines`` helper means the DNAT rule shape is
        # byte-identical to the Limited lockdown's — proven here by extracting
        # the DNAT line from each and comparing.
        dnat_only = build_secret_egress_dnat_script(
            dnat_hosts=["api.secret.com"],
            dnat_target=("aios-worker", 49152),
            dns_port=53535,
        )
        lockdown = build_iptables_script(
            allowed_hosts=set(),
            dnat_hosts=["api.secret.com"],
            dnat_target=("aios-worker", 49152),
            dns_port=53535,
        )

        def _dnat_line(script: str) -> str:
            return next(line for line in script.splitlines() if "-j DNAT --to-destination" in line)

        assert _dnat_line(dnat_only) == _dnat_line(lockdown)


# ── read-back verify script asserts DROP + DNAT coverage (#984) ───────────────


class TestBuildLockdownVerifyScript:
    def test_asserts_filter_drop_policy(self) -> None:
        script = build_lockdown_verify_script()
        assert "\"$IPT\" -S OUTPUT | grep -qx -- '-P OUTPUT DROP'" in script

    def test_no_dnat_assertion_without_dnat_hosts(self) -> None:
        """With no credential hosts, there's no nat coverage to assert — the
        verify must not reference the nat table."""
        script = build_lockdown_verify_script(dnat_hosts=[])
        assert "-t nat" not in script
        assert "DNAT" not in script

    def test_asserts_every_rule_of_the_name_based_chokepoint(self) -> None:
        """#984 + #2042: the read-back must prove the WHOLE chokepoint landed.

        Asserting only that "some ``-j DNAT`` exists" (the pre-#2042 check)
        passes on a half-installed ruleset — DNS intercepted but the sentinel
        unrouted, or the reverse — i.e. a green verify over unprotected
        credential egress. All four rules are asserted individually.
        """
        script = build_lockdown_verify_script(dnat_hosts=["api.secret.com"])
        assert (
            '"$IPT" -t nat -S OUTPUT | grep -q -- '
            f"'-d {CREDENTIAL_SENTINEL_IP}.*--dport 443 -j DNAT'"
        ) in script
        assert ("\"$IPT\" -t nat -S OUTPUT | grep -q -- '-p udp .*--dport 53 .*-j DNAT'") in script
        assert ("\"$IPT\" -t nat -S OUTPUT | grep -q -- '-p tcp .*--dport 53 .*-j DNAT'") in script
        assert f"\"$IPT\" -S OUTPUT | grep -q -- '-d {CREDENTIAL_SENTINEL_IP}.*-j REJECT'" in script
        # The filter-table DROP assertion is still present.
        assert "OUTPUT DROP" in script

    def test_uses_selected_backend_no_bare_iptables(self) -> None:
        """Both assertions go through the ``$IPT`` selector so the verify reads
        the same netfilter backend the apply wrote to (#1022)."""
        script = build_lockdown_verify_script(dnat_hosts=["api.secret.com"])
        assert "command -v iptables-legacy" in script
        for line in script.splitlines():
            stripped = line.strip()
            if "command -v iptables-legacy" in stripped:
                continue
            assert not stripped.startswith("iptables "), (
                f"verify uses a bare iptables (nft default): {line!r}"
            )

    def test_asserts_ip6tables_drop_policy(self) -> None:
        """#1207: the v6 DROP installed by the apply must itself be verified —
        without asserting ``ip6tables -S OUTPUT`` shows ``-P OUTPUT DROP`` the
        new v6 DROP is unverified, re-creating the 'green verify while open' gap
        one layer down. The assertion goes through ``$IP6T -S OUTPUT`` and
        ``grep`` for the DROP policy."""
        script = build_lockdown_verify_script()
        assert '"$IP6T" -S OUTPUT' in script
        assert "grep -qx -- '-P OUTPUT DROP'" in script

    def test_v6_verify_guarded_on_table_availability(self) -> None:
        """#1207 fix: the v6 read-back must NOT hard-fail when the ``ip6_tables``
        kernel module is absent (the v6 ``filter`` table cannot initialize, so
        the apply correctly skipped its DROP and there is no policy to read
        back). The assertion is guarded on ``$IP6T -S OUTPUT`` succeeding, so a
        missing module passes the verify (nothing to secure) while a present
        table still requires the DROP. Without this guard the docker e2e shard
        fails on CI runners that don't load ip6_tables."""
        script = build_lockdown_verify_script()
        # The v6 read-back is captured under a conditional, not a bare pipe that
        # would propagate ip6tables' init failure as the script's exit status.
        assert 'if v6_output="$("$IP6T" -S OUTPUT 2>/dev/null)"; then' in script
        # Still asserts the DROP policy when the table IS readable.
        assert "grep -qx -- '-P OUTPUT DROP'" in script

    def test_v6_verify_uses_legacy_backend_no_bare_ip6tables(self) -> None:
        """The v6 read-back selects the same legacy backend the apply wrote to,
        so it reads the right table under runsc — and never a bare ip6tables."""
        script = build_lockdown_verify_script(dnat_hosts=["api.secret.com"])
        assert "command -v ip6tables-legacy" in script
        for line in script.splitlines():
            stripped = line.strip()
            if "command -v ip6tables-legacy" in stripped:
                continue
            assert not stripped.startswith("ip6tables "), (
                f"verify uses a bare ip6tables (nft default): {line!r}"
            )

    def test_assert_drop_false_omits_drop_assertion(self) -> None:
        # The DNAT-only Unrestricted path (#1153) leaves the filter policy at
        # ACCEPT, so the verify must NOT assert a DROP policy (it would always
        # fail) — but still asserts nat DNAT coverage. The v6 DROP assertion is
        # likewise omitted (the DNAT-only path installs no v6 DROP).
        script = build_lockdown_verify_script(dnat_hosts=["api.secret.com"], assert_drop=False)
        assert "OUTPUT DROP" not in script
        assert "IP6T" not in script
        assert (
            '"$IPT" -t nat -S OUTPUT | grep -q -- '
            f"'-d {CREDENTIAL_SENTINEL_IP}.*--dport 443 -j DNAT'"
        ) in script

    def test_assert_drop_true_is_default(self) -> None:
        # Backward-compat: the Limited callers pass no assert_drop and must keep
        # getting the DROP assertion.
        assert "OUTPUT DROP" in build_lockdown_verify_script(dnat_hosts=["api.secret.com"])

    def test_emits_set_e_first(self) -> None:
        """Every assertion must be independently fatal. The sidecar runs the
        script via ``bash -c`` with NO ``-e`` flag, so the script's exit status
        defaults to its LAST command — the trailing guarded v6 ``if`` that
        returns 0 when the v6 table is unavailable. Without ``set -e`` at the top
        that trailing 0 masks a failed earlier v4 DROP assertion (fail-open).
        ``set -e`` must be the first line so the v4 (and nat) assertions abort the
        script the instant they fail."""
        for script in (
            build_lockdown_verify_script(),
            build_lockdown_verify_script(dnat_hosts=["api.secret.com"]),
            build_lockdown_verify_script(dnat_hosts=["api.secret.com"], assert_drop=False),
        ):
            assert script.splitlines()[0] == "set -e"

    def _run_verify(self, *, v4_policy: str, v6_mode: str, dnat_hosts: Sequence[str] = ()) -> int:
        """Run the generated verify script under ``bash -c`` (exactly as the
        sidecar does — no ``-e`` on the call) against fake legacy
        binaries, returning its exit code.

        ``v4_policy``/``v6_mode`` are the ``-S OUTPUT`` policies the fakes report
        (``"DROP"``/``"ACCEPT"``); ``v6_mode="unavailable"`` makes ``ip6tables-S
        OUTPUT`` fail to initialize (the no-``ip6_tables``-module / CI case).
        """
        script = build_lockdown_verify_script(dnat_hosts=dnat_hosts)
        bindir = tempfile.mkdtemp()
        v4 = (
            f"#!/usr/bin/env bash\n"
            f"if [ \"$1\" = '-S' ] && [ \"$2\" = 'OUTPUT' ]; then echo '-P OUTPUT {v4_policy}'; exit 0; fi\n"
            f"# nat -S OUTPUT carries a DNAT rule so the nat assertion (if any) passes\n"
            f"if [ \"$1\" = '-t' ] && [ \"$2\" = 'nat' ]; then echo '-A OUTPUT -j DNAT --to-destination 1.2.3.4:443'; exit 0; fi\n"
            f"exit 0\n"
        )
        if v6_mode == "unavailable":
            v6_body = "echo \"ip6tables: can't initialize table 'filter'\" >&2; exit 3;"
        else:
            v6_body = f"echo '-P OUTPUT {v6_mode}'; exit 0;"
        v6 = (
            f"#!/usr/bin/env bash\n"
            f"if [ \"$1\" = '-S' ] && [ \"$2\" = 'OUTPUT' ]; then {v6_body} fi\n"
            f"exit 0\n"
        )
        for name, body in (("iptables-legacy", v4), ("ip6tables-legacy", v6)):
            p = os.path.join(bindir, name)
            with open(p, "w") as f:
                f.write(body)
            os.chmod(p, 0o755)
        env = dict(os.environ)
        env["PATH"] = bindir + os.pathsep + env["PATH"]
        return subprocess.run(
            ["bash", "-c", script], env=env, capture_output=True, text=True
        ).returncode

    def test_v4_drop_absent_fails_even_when_v6_table_unavailable(self) -> None:
        """REGRESSION (#1207 verify-ordering fail-open): when the v6 ``filter``
        table is unavailable (no ``ip6_tables`` module — the common CI /
        IPv6-disabled-host case) the trailing guarded v6 ``if`` returns 0. Without
        ``set -e`` that trailing 0 OVERWRITES a failed earlier v4 ``-P OUTPUT
        DROP`` assertion, so verify passes GREEN while the box is open over IPv4.
        With ``set -e`` the v4 assertion aborts the script before the v6 block can
        mask it — fail-closed."""
        assert self._run_verify(v4_policy="ACCEPT", v6_mode="unavailable") != 0

    def test_v4_drop_present_passes_when_v6_table_unavailable(self) -> None:
        """The graceful-skip path still passes: v4 locked down + no v6 stack to
        secure → verify is GREEN (the v6 ``if`` guard skips, ``set -e`` does not
        fire on a tested condition)."""
        assert self._run_verify(v4_policy="DROP", v6_mode="unavailable") == 0

    def test_v6_drop_absent_fails_when_v6_table_present(self) -> None:
        """When the v6 table IS present (the case the v6 DROP defends), a missing
        v6 ``-P OUTPUT DROP`` still fails the verify."""
        assert self._run_verify(v4_policy="DROP", v6_mode="ACCEPT") != 0

    def test_both_drop_present_passes(self) -> None:
        assert self._run_verify(v4_policy="DROP", v6_mode="DROP") == 0


# ── docker backend translates network policy to docker run argv ────────────────


def _make_spec(network_policy: LimitedNetworking | UnrestrictedNetworking) -> SandboxSpec:
    return SandboxSpec(
        session_id="sess_01TEST",
        instance_id="inst_TEST",
        workspace=Mount(host_path=Path("/tmp/ws"), sandbox_path="/workspace"),
        extra_mounts=(),
        environment={},
        labels={"aios.managed": "true"},
        network_policy=network_policy,
        host_gateway_alias=None,
        image="aios-sandbox:test",
    )


async def _capture_docker_argv(spec: SandboxSpec) -> list[str]:
    captured: list[list[str]] = []

    async def fake_run_docker(
        argv: list[str], *, timeout_s: float = 30.0, **kwargs: Any
    ) -> tuple[int, bytes, bytes]:
        captured.append(argv)
        return 0, b"container_abc123\n", b""

    with patch("aios.sandbox.backends.docker.run_docker_cli", fake_run_docker):
        await DockerBackend().create(spec)
    return captured[0]


class TestDockerBackendArgs:
    """The DockerBackend translates SandboxSpec.network_policy to the right argv."""

    @pytest.mark.asyncio
    async def test_limited_does_not_add_cap_net_admin(self) -> None:
        """Durable session sandboxes (§5.8): the sandbox holds NO ``NET_ADMIN``
        even under Limited networking — the lockdown is applied from an
        ephemeral operator-image sidecar joined to the netns, so root-in-sandbox
        can neither poison nor flush its own lockdown."""
        argv = await _capture_docker_argv(
            _make_spec(LimitedNetworking(type="limited", allowed_hosts=["example.com"]))
        )
        assert "NET_ADMIN" not in argv

    @pytest.mark.asyncio
    async def test_unrestricted_no_cap_net_admin(self) -> None:
        argv = await _capture_docker_argv(_make_spec(UnrestrictedNetworking()))
        assert "--cap-add" not in argv

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "policy",
        [
            LimitedNetworking(type="limited", allowed_hosts=["example.com"]),
            UnrestrictedNetworking(),
        ],
        ids=["limited", "unrestricted"],
    )
    async def test_security_opt_no_new_privileges(
        self, policy: LimitedNetworking | UnrestrictedNetworking
    ) -> None:
        argv = await _capture_docker_argv(_make_spec(policy))
        assert "--security-opt" in argv
        i = argv.index("--security-opt")
        assert argv[i + 1] == "no-new-privileges"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "policy",
        [
            LimitedNetworking(type="limited", allowed_hosts=["example.com"]),
            UnrestrictedNetworking(),
        ],
        ids=["limited", "unrestricted"],
    )
    async def test_ipc_private(self, policy: LimitedNetworking | UnrestrictedNetworking) -> None:
        argv = await _capture_docker_argv(_make_spec(policy))
        assert "--ipc" in argv
        i = argv.index("--ipc")
        assert argv[i + 1] == "private"


class TestDockerBackendIsAlive:
    """DockerBackend.is_alive maps ``docker inspect`` outcomes to a bool (#691).

    The registry's warm path depends on this method being total: any
    failure mode must resolve to ``True`` (running) or ``False`` (re-
    provision), never an exception that escapes into the tool call.
    These tests patch ``run_docker_cli`` so no real daemon is needed.
    """

    @pytest.mark.asyncio
    async def test_running_true_returns_alive(self) -> None:
        async def fake(argv: list[str]) -> tuple[int, bytes, bytes]:
            # Sanity: we inspect the right container with the State.Running format.
            assert argv[:3] == ["docker", "inspect", "--format"]
            assert make_handle().sandbox_id in argv
            return 0, b"true\n", b""

        with patch("aios.sandbox.backends.docker.run_docker_cli", fake):
            assert await DockerBackend().is_alive(make_handle()) is True

    @pytest.mark.asyncio
    async def test_running_false_string_returns_dead(self) -> None:
        """A stopped (but not yet removed) container inspects as 'false'."""

        async def fake(argv: list[str]) -> tuple[int, bytes, bytes]:
            return 0, b"false\n", b""

        with patch("aios.sandbox.backends.docker.run_docker_cli", fake):
            assert await DockerBackend().is_alive(make_handle()) is False

    @pytest.mark.asyncio
    async def test_nonzero_exit_returns_dead(self) -> None:
        """`--rm` removed the container → inspect exits nonzero ('No such container')."""

        async def fake(argv: list[str]) -> tuple[int, bytes, bytes]:
            return 1, b"", b"Error: No such container: abc123\n"

        with patch("aios.sandbox.backends.docker.run_docker_cli", fake):
            assert await DockerBackend().is_alive(make_handle()) is False

    @pytest.mark.asyncio
    async def test_probe_launch_failure_returns_dead_not_raises(self) -> None:
        """A daemon hiccup / timeout (SandboxBackendError) must not escape."""

        async def fake(argv: list[str]) -> tuple[int, bytes, bytes]:
            raise SandboxBackendError("docker cli timed out after 30.0s")

        with patch("aios.sandbox.backends.docker.run_docker_cli", fake):
            assert await DockerBackend().is_alive(make_handle()) is False

    @pytest.mark.asyncio
    async def test_unexpected_exception_returns_dead_not_raises(self) -> None:
        """Totality contract: ANY non-cancellation error resolves to dead."""

        async def fake(argv: list[str]) -> tuple[int, bytes, bytes]:
            raise RuntimeError("unexpected boom")

        with patch("aios.sandbox.backends.docker.run_docker_cli", fake):
            assert await DockerBackend().is_alive(make_handle()) is False

    @pytest.mark.asyncio
    async def test_cancellation_propagates(self) -> None:
        """CancelledError (worker shutdown) must NOT be swallowed as 'dead'."""

        async def fake(argv: list[str]) -> tuple[int, bytes, bytes]:
            raise asyncio.CancelledError

        with (
            patch("aios.sandbox.backends.docker.run_docker_cli", fake),
            pytest.raises(asyncio.CancelledError),
        ):
            await DockerBackend().is_alive(make_handle())


# ── network lockdown helper applies the right script via the backend ──────────


class TestApplyNetworkLockdown:
    """apply_network_lockdown builds the script and applies + verifies it via
    the operator-image netns sidecar (durable session sandboxes, §5.8)."""

    @staticmethod
    def _sidecar_scripts(backend: FakeBackend) -> list[str]:
        return [c[1]["script"] for c in backend.calls if c[0] == "run_netns_sidecar"]

    @pytest.mark.asyncio
    async def test_applies_and_verifies_via_sidecar(self) -> None:
        backend = FakeBackend()
        handle = make_handle()
        networking = LimitedNetworking(type="limited", allowed_hosts=["api.example.com"])

        await apply_network_lockdown(backend, handle, networking)

        scripts = self._sidecar_scripts(backend)
        # Two sidecar calls: apply, then read-back verify.
        assert len(scripts) == 2
        apply_script, verify_script = scripts
        assert '"$IPT" -P OUTPUT DROP' in apply_script
        assert "resolve_ipv4 api.example.com" in apply_script
        # The sidecar inherits the operator image's (empty) resolv.conf, so the
        # apply script points itself at the netns embedded DNS before getent.
        assert "127.0.0.11" in apply_script
        assert "OUTPUT DROP" in verify_script
        # The sidecar runs the OPERATOR image, never env_config.image.
        sidecar_call = next(c for c in backend.calls if c[0] == "run_netns_sidecar")
        assert sidecar_call[1]["image"].endswith("aios-sandbox:latest")
        assert sidecar_call[1]["target_sandbox_id"] == handle.sandbox_id

    @pytest.mark.asyncio
    async def test_includes_package_registries_when_enabled(self) -> None:
        backend = FakeBackend()
        handle = make_handle()
        networking = LimitedNetworking(
            type="limited",
            allowed_hosts=["api.example.com"],
            allow_package_managers=True,
        )

        await apply_network_lockdown(backend, handle, networking)

        script = self._sidecar_scripts(backend)[0]
        assert "pypi.org" in script
        assert "registry.npmjs.org" in script
        assert "api.example.com" in script

    @pytest.mark.asyncio
    async def test_no_package_registries_when_disabled(self) -> None:
        backend = FakeBackend()
        handle = make_handle()
        networking = LimitedNetworking(
            type="limited",
            allowed_hosts=["api.example.com"],
            allow_package_managers=False,
        )

        await apply_network_lockdown(backend, handle, networking)

        script = self._sidecar_scripts(backend)[0]
        assert "pypi.org" not in script
        assert "api.example.com" in script

    @pytest.mark.asyncio
    async def test_extra_host_ports_threaded_through(self) -> None:
        backend = FakeBackend()
        handle = make_handle()
        networking = LimitedNetworking(type="limited", allowed_hosts=[])

        await apply_network_lockdown(
            backend,
            handle,
            networking,
            extra_host_ports=[("aios-worker", 8765)],
        )

        script = self._sidecar_scripts(backend)[0]
        assert "aios-worker:8765" in script

    @pytest.mark.asyncio
    async def test_dnat_params_threaded_through(self) -> None:
        backend = FakeBackend()
        handle = make_handle()
        networking = LimitedNetworking(type="limited", allowed_hosts=[])

        await apply_network_lockdown(
            backend,
            handle,
            networking,
            extra_host_ports=[("aios-worker", 49152)],
            dnat_hosts=["api.secret.com"],
            dnat_target=("aios-worker", 49152),
            dns_port=53535,
        )

        apply_script, verify_script = self._sidecar_scripts(backend)
        assert (
            f'"$IPT" -t nat -A OUTPUT -d {CREDENTIAL_SENTINEL_IP} -p tcp --dport 443 '
            '-j DNAT --to-destination "$PROXY_IP:49152"'
        ) in apply_script
        assert '--dport 53 -j DNAT --to-destination "$PROXY_IP:53535"' in apply_script
        # No credential host is resolved to build the rules any more (#2042).
        assert "resolve_ipv4 api.secret.com" not in apply_script
        # #984 + #2042: with dnat_hosts present the read-back verify asserts
        # every rule of the chokepoint — a partial install fails closed.
        assert (
            '"$IPT" -t nat -S OUTPUT | grep -q -- '
            f"'-d {CREDENTIAL_SENTINEL_IP}.*--dport 443 -j DNAT'"
        ) in verify_script

    @pytest.mark.asyncio
    async def test_verify_omits_nat_assertion_without_dnat_hosts(self) -> None:
        """Without credential DNAT, the read-back verify only asserts the filter
        DROP policy — it must not reference the nat table (#984)."""
        backend = FakeBackend()
        handle = make_handle()
        networking = LimitedNetworking(type="limited", allowed_hosts=["api.example.com"])

        await apply_network_lockdown(backend, handle, networking)

        _apply, verify_script = self._sidecar_scripts(backend)
        assert "-t nat" not in verify_script
        assert "DNAT" not in verify_script

    @pytest.mark.asyncio
    async def test_apply_and_verify_agree_on_legacy_backend(self) -> None:
        """#1022: gVisor only implements legacy netfilter. The apply script
        and the read-back verify script must select the SAME iptables backend,
        or the verify could read an empty nft table while the legacy table holds
        the DROP policy (or vice versa)."""
        backend = FakeBackend()
        handle = make_handle()
        networking = LimitedNetworking(type="limited", allowed_hosts=["api.example.com"])

        await apply_network_lockdown(backend, handle, networking)

        apply_script, verify_script = self._sidecar_scripts(backend)
        # Both scripts run the same backend-selection preamble.
        assert "command -v iptables-legacy" in apply_script
        assert "command -v iptables-legacy" in verify_script
        # The verify reads the OUTPUT chain via the selected binary, not bare iptables.
        assert '"$IPT" -S OUTPUT' in verify_script
        for line in verify_script.splitlines():
            stripped = line.strip()
            if "command -v iptables-legacy" in stripped:
                continue
            assert not stripped.startswith("iptables "), (
                f"verify uses a bare iptables (nft default): {line!r}"
            )

    def test_verify_script_uses_selected_backend(self) -> None:
        """The verify script itself selects the legacy backend so any caller
        that runs it directly stays consistent with the apply path."""
        script = build_lockdown_verify_script()
        assert "command -v iptables-legacy" in script
        assert '"$IPT" -S OUTPUT' in script
        assert "OUTPUT DROP" in script

    @pytest.mark.asyncio
    async def test_runtime_threaded_to_both_sidecar_calls(self) -> None:
        """The sandbox's container runtime (#1014) reaches BOTH sidecar
        calls (apply and read-back verify) so the lockdown runs under
        the same runtime as the sandbox it locks down."""
        backend = FakeBackend()
        handle = make_handle()
        networking = LimitedNetworking(type="limited", allowed_hosts=["api.example.com"])

        await apply_network_lockdown(backend, handle, networking, runtime="runsc")

        runtimes = [c[1]["runtime"] for c in backend.calls if c[0] == "run_netns_sidecar"]
        assert runtimes == ["runsc", "runsc"]

    @pytest.mark.asyncio
    async def test_runtime_defaults_to_none(self) -> None:
        backend = FakeBackend()
        handle = make_handle()
        networking = LimitedNetworking(type="limited", allowed_hosts=["api.example.com"])

        await apply_network_lockdown(backend, handle, networking)

        runtimes = [c[1]["runtime"] for c in backend.calls if c[0] == "run_netns_sidecar"]
        assert runtimes == [None, None]


# ── lockdown fails closed (security gate, not best-effort) ─────────────────────


class TestApplyNetworkLockdownFailsClosed:
    """A :class:`Limited` sandbox whose lockdown didn't apply (or whose DROP
    policy didn't verify) is a silent unrestricted-networking bypass. The
    lockdown is a security gate, so a nonzero apply, a failed read-back verify,
    or a sidecar infra error must raise rather than log-and-continue, letting
    the registry tear the sandbox down.
    """

    @pytest.mark.asyncio
    async def test_nonzero_apply_raises_sandbox_backend_error(self) -> None:
        backend = FakeBackend()
        backend.sidecar_results = [
            CommandResult(
                exit_code=3,
                stdout="",
                stderr="iptables: command not found",
                timed_out=False,
                truncated=False,
            )
        ]
        handle = make_handle()
        networking = LimitedNetworking(type="limited", allowed_hosts=["api.example.com"])

        with pytest.raises(SandboxBackendError, match="network lockdown failed"):
            await apply_network_lockdown(backend, handle, networking)

    @pytest.mark.asyncio
    async def test_failed_verify_raises(self) -> None:
        """Apply succeeds but the read-back shows the OUTPUT policy is not DROP
        (the lockdown didn't actually land) → fail closed."""
        backend = FakeBackend()
        ok = CommandResult(exit_code=0, stdout="", stderr="", timed_out=False, truncated=False)
        bad = CommandResult(exit_code=1, stdout="", stderr="", timed_out=False, truncated=False)
        backend.sidecar_results = [ok, bad]  # apply ok, verify fails
        handle = make_handle()
        networking = LimitedNetworking(type="limited", allowed_hosts=["api.example.com"])

        with pytest.raises(SandboxBackendError, match="verification failed"):
            await apply_network_lockdown(backend, handle, networking)

    @pytest.mark.asyncio
    async def test_sidecar_error_propagates(self) -> None:
        """An infra failure to even run the lockdown sidecar must not be
        swallowed into a wide-open sandbox — it propagates so the provision
        aborts."""
        backend = FakeBackend()
        backend.run_netns_sidecar = AsyncMock(  # type: ignore[method-assign]
            side_effect=SandboxBackendError("daemon hiccup")
        )
        handle = make_handle()
        networking = LimitedNetworking(type="limited", allowed_hosts=["api.example.com"])

        with pytest.raises(SandboxBackendError, match="daemon hiccup"):
            await apply_network_lockdown(backend, handle, networking)

    @pytest.mark.asyncio
    async def test_zero_exit_does_not_raise(self) -> None:
        """The happy path (apply + verify both exit 0) is unchanged — no exception."""
        backend = FakeBackend()  # default sidecar returns exit 0 for both calls
        handle = make_handle()
        networking = LimitedNetworking(type="limited", allowed_hosts=["api.example.com"])

        await apply_network_lockdown(backend, handle, networking)  # must not raise


# ── DNAT-only apply for Unrestricted credentialed sandboxes (#1153) ────────────


class TestApplySecretEgressDnat:
    """``apply_secret_egress_dnat`` installs the credential-host → proxy DNAT in
    an OPEN-egress sandbox: same operator-image netns sidecar + fail-closed
    posture as the Limited path, but DNAT-only (no filter DROP) and verified
    with ``assert_drop=False``."""

    @staticmethod
    def _sidecar_scripts(backend: FakeBackend) -> list[str]:
        return [c[1]["script"] for c in backend.calls if c[0] == "run_netns_sidecar"]

    @pytest.mark.asyncio
    async def test_applies_dnat_only_no_drop_then_verifies(self) -> None:
        backend = FakeBackend()
        handle = make_handle()

        await apply_secret_egress_dnat(
            backend,
            handle,
            dnat_hosts=["api.secret.com"],
            dnat_target=("aios-worker", 49152),
            dns_port=53535,
        )

        apply_script, verify_script = self._sidecar_scripts(backend)
        # Chokepoint installed, but general egress stays open (no DROP, no
        # per-host ACCEPT). Interception is keyed on the sentinel, and no
        # credential host is resolved to build it (#2042).
        assert (
            f"-d {CREDENTIAL_SENTINEL_IP} -p tcp --dport 443 "
            '-j DNAT --to-destination "$PROXY_IP:49152"'
        ) in apply_script
        assert '--dport 53 -j DNAT --to-destination "$PROXY_IP:53535"' in apply_script
        assert "resolve_ipv4 api.secret.com" not in apply_script
        assert "-P OUTPUT DROP" not in apply_script
        # The verify asserts the chokepoint landed but NOT a DROP policy.
        assert (
            '"$IPT" -t nat -S OUTPUT | grep -q -- '
            f"'-d {CREDENTIAL_SENTINEL_IP}.*--dport 443 -j DNAT'"
        ) in verify_script
        assert "OUTPUT DROP" not in verify_script

    @pytest.mark.asyncio
    async def test_resolv_preamble_prepended_to_apply(self) -> None:
        # The apply script points the netns-joining sidecar at the embedded
        # resolver before any getent runs (same preamble as the Limited path).
        backend = FakeBackend()
        handle = make_handle()

        await apply_secret_egress_dnat(
            backend,
            handle,
            dnat_hosts=["api.secret.com"],
            dnat_target=("aios-worker", 49152),
            dns_port=53535,
        )

        apply_script = self._sidecar_scripts(backend)[0]
        assert "nameserver 127.0.0.11" in apply_script

    @pytest.mark.asyncio
    async def test_runtime_threaded_to_both_sidecar_calls(self) -> None:
        backend = FakeBackend()
        handle = make_handle()

        await apply_secret_egress_dnat(
            backend,
            handle,
            dnat_hosts=["api.secret.com"],
            dnat_target=("aios-worker", 49152),
            dns_port=53535,
            runtime="runsc",
        )

        runtimes = [c[1]["runtime"] for c in backend.calls if c[0] == "run_netns_sidecar"]
        assert runtimes == ["runsc", "runsc"]

    @pytest.mark.asyncio
    async def test_nonzero_apply_raises_sandbox_backend_error(self) -> None:
        backend = FakeBackend()
        backend.sidecar_results = [
            CommandResult(
                exit_code=3,
                stdout="",
                stderr="iptables: command not found",
                timed_out=False,
                truncated=False,
            )
        ]
        handle = make_handle()

        with pytest.raises(SandboxBackendError, match="secret-egress DNAT failed"):
            await apply_secret_egress_dnat(
                backend,
                handle,
                dnat_hosts=["api.secret.com"],
                dnat_target=("aios-worker", 49152),
                dns_port=53535,
            )

    @pytest.mark.asyncio
    async def test_failed_verify_raises(self) -> None:
        # Apply succeeds but the read-back shows no DNAT rule landed (e.g. a
        # zero-IP credential host, #984) → fail closed.
        backend = FakeBackend()
        ok = CommandResult(exit_code=0, stdout="", stderr="", timed_out=False, truncated=False)
        bad = CommandResult(exit_code=1, stdout="", stderr="", timed_out=False, truncated=False)
        backend.sidecar_results = [ok, bad]
        handle = make_handle()

        with pytest.raises(SandboxBackendError, match="secret-egress DNAT verification failed"):
            await apply_secret_egress_dnat(
                backend,
                handle,
                dnat_hosts=["api.secret.com"],
                dnat_target=("aios-worker", 49152),
                dns_port=53535,
            )

    @pytest.mark.asyncio
    async def test_sidecar_error_propagates(self) -> None:
        backend = FakeBackend()
        backend.run_netns_sidecar = AsyncMock(  # type: ignore[method-assign]
            side_effect=SandboxBackendError("daemon hiccup")
        )
        handle = make_handle()

        with pytest.raises(SandboxBackendError, match="daemon hiccup"):
            await apply_secret_egress_dnat(
                backend,
                handle,
                dnat_hosts=["api.secret.com"],
                dnat_target=("aios-worker", 49152),
                dns_port=53535,
            )

    @pytest.mark.asyncio
    async def test_zero_exit_does_not_raise(self) -> None:
        backend = FakeBackend()  # default sidecar returns exit 0 for both calls
        handle = make_handle()

        await apply_secret_egress_dnat(
            backend,
            handle,
            dnat_hosts=["api.secret.com"],
            dnat_target=("aios-worker", 49152),
            dns_port=53535,
        )  # must not raise


# ── behavioural egress verdict: what happens to a packet, not what the rule
#    text says (finding 4 / eumemic/aios#2042) ──────────────────────────────────


class TestCredentialHostEgressVerdict:
    """Evaluate the GENERATED ruleset against a packet the sandbox would send.

    Every other test in this file asserts *rule syntax* — that some string is
    present in the script. That is exactly why the Unrestricted fail-open in
    finding 4 survived a review cycle: rules were emitted, the syntax
    assertions passed, and nobody asked what happens to a packet addressed to
    an IP the sampler never returned. This class asks that question.

    Method, post-#2042, is end-to-end in the one way that matters: the sandbox
    reaches a credential host **by name**, so the harness RESOLVES THE NAME THE
    WAY THE SANDBOX WOULD — through the real
    :class:`~aios.sandbox.credential_dns.CredentialDnsResolver` that the netns
    ``:53`` DNAT points at — and only then replays the resulting packet against
    the recorded ruleset the way netfilter would: nat OUTPUT first (a DNAT match
    rewrites the destination), then filter OUTPUT, first-match-wins, falling
    through to the chain policy. Verdicts:

    * ``proxied``  — rewritten to the secret-egress proxy (safe: the swap fires),
    * ``blocked``  — REJECTed or DROPped (safe: nothing leaves),
    * ``direct``   — leaves the netns toward the real upstream (UNSAFE for a
      credential host: this is the literal placeholder on the wire).

    ``SAMPLED_IP`` is an address the in-sandbox resolver returned when the
    rules were generated; ``UNSAMPLED_IP`` is a live pool member it never
    returned (a rotating DNS pool serves only a subset per query, so this is
    the ordinary case, not an exotic one). The whole point of #2042 is that
    this distinction no longer exists downstream of resolution: NEITHER address
    is what the name resolves to inside the sandbox any more.
    """

    HOST = "api.secret.com"
    SAMPLED_IP = "140.82.112.5"
    UNSAMPLED_IP = "140.82.113.22"
    PROXY_IP = "10.0.0.9"
    PROXY_PORT = 49152
    DNS_PORT = 53535

    # ── harness ───────────────────────────────────────────────────────────────

    def _record_rules(self, script: str) -> list[tuple[str, list[str]]]:
        """Run the generated script with recording shims; return (table, argv)."""
        bindir = tempfile.mkdtemp()
        log = os.path.join(bindir, "rules.log")
        # getent ahostsv4 <host> answers with the SAMPLED address only — the
        # subset the resolver happened to return at rule-generation time. Post
        # #2042 the credential host is not looked up at all; the shim still
        # answers so any regression that reintroduces sampling is visible.
        getent = (
            "#!/usr/bin/env bash\n"
            f'if [ "$2" = "{self.HOST}" ]; then echo "{self.SAMPLED_IP} STREAM {self.HOST}"; fi\n'
            f'if [ "$2" = "aios-worker" ]; then echo "{self.PROXY_IP} STREAM aios-worker"; fi\n'
            "exit 0\n"
        )
        # iptables shim: append the argv of every mutating call to the log.
        # -S/-C/-D calls are answered so the script's guards behave (nothing is
        # installed, so -C "rule exists?" is false and -D is a no-op).
        ipt = (
            "#!/usr/bin/env bash\n"
            f'printf "%s\\n" "$*" >> {log}\n'
            'for a in "$@"; do\n'
            '  case "$a" in -S) exit 0;; -C) exit 1;; -D) exit 1;; esac\n'
            "done\n"
            "exit 0\n"
        )
        for name, body in (
            ("getent", getent),
            ("ip", "#!/usr/bin/env bash\nexit 0\n"),
            ("iptables-legacy", ipt),
            ("iptables", ipt),
            ("ip6tables-legacy", ipt),
            ("ip6tables", ipt),
        ):
            p = os.path.join(bindir, name)
            with open(p, "w") as f:
                f.write(body)
            os.chmod(p, 0o755)
        env = dict(os.environ)
        env["PATH"] = bindir + os.pathsep + env["PATH"]
        proc = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
        assert proc.returncode == 0, f"apply script failed: {proc.stderr}"
        recorded: list[tuple[str, list[str]]] = []
        if os.path.exists(log):
            with open(log) as f:
                for line in f:
                    argv = line.split()
                    is_nat = "-t" in argv and argv[argv.index("-t") + 1] == "nat"
                    recorded.append(("nat" if is_nat else "filter", argv))
        return recorded

    def _resolve_in_sandbox(self, host: str, *, pool: Sequence[str]) -> str | None:
        """Resolve ``host`` the way a process INSIDE the sandbox would.

        All sandbox ``:53`` is DNATed to the session's
        :class:`CredentialDnsResolver`, so this drives the real resolver with a
        real wire-format query. ``pool`` is what the upstream would answer for a
        NON-credential name — including addresses no sampler ever returned; if
        the resolver ever forwarded a credential name, the sandbox would learn
        one of these and the fix would be a fiction.

        Returns the dotted-quad the sandbox gets back, or ``None`` for NODATA.
        """
        query = (
            struct.pack("!HHHHHH", 0x1234, 0x0100, 1, 0, 0, 0)
            + b"".join(bytes([len(x)]) + x.encode() for x in host.split("."))
            + b"\x00"
            + struct.pack("!HH", 1, 1)  # A / IN
        )

        async def _drive() -> bytes:
            resolver = CredentialDnsResolver([self.HOST], upstream=None)

            # Stub the forward path: a name the resolver does NOT own is
            # answered from the live rotating pool.
            async def _forward(q: bytes) -> bytes:
                return (
                    struct.pack("!HHHHHH", 0x1234, 0x8180, 1, 1, 0, 0)
                    + q[12:]
                    + struct.pack("!HHHIH", 0xC00C, 1, 1, 60, 4)
                    + socket.inet_aton(pool[0])
                )

            resolver._forward = _forward  # type: ignore[assignment]
            return await resolver.answer(query)

        response = asyncio.run(_drive())
        (ancount,) = struct.unpack_from("!H", response, 6)
        if ancount == 0:
            return None
        rdata_start = len(response) - 4
        return socket.inet_ntoa(response[rdata_start:])

    def _verdict(self, rules: list[tuple[str, list[str]]], dest_ip: str, dport: int = 443) -> str:
        """Replay the recorded ruleset against one packet, netfilter-style."""

        def _chain(table: str) -> list[list[str]]:
            """OUTPUT rules for ``table`` in traversal order (-I prepends)."""
            chain: list[list[str]] = []
            for t, argv in rules:
                if t != table or "OUTPUT" not in argv:
                    continue
                if "-A" in argv:
                    chain.append(argv)
                elif "-I" in argv:
                    chain.insert(0, argv)
            return chain

        def _matches(argv: list[str], ip: str, port: int) -> bool:
            if "-d" in argv and argv[argv.index("-d") + 1] != ip:
                return False
            if "--dport" in argv and argv[argv.index("--dport") + 1] != str(port):
                return False
            # Rules that gate on something this model doesn't carry (loopback
            # interface, conntrack state, a different protocol) can't match a
            # fresh outbound TCP packet to a remote address.
            if "-o" in argv or "conntrack" in argv:
                return False
            return not ("-p" in argv and argv[argv.index("-p") + 1] != "tcp")

        # nat OUTPUT: a DNAT match rewrites the destination.
        for argv in _chain("nat"):
            if _matches(argv, dest_ip, dport) and "DNAT" in argv:
                return "proxied"
        # filter OUTPUT: first match wins, else the chain policy.
        for argv in _chain("filter"):
            if not _matches(argv, dest_ip, dport):
                continue
            if "REJECT" in argv or "DROP" in argv:
                return "blocked"
            if "ACCEPT" in argv:
                return "direct"
        policy = "ACCEPT"
        for _t, argv in rules:
            if argv[:3] == ["-P", "OUTPUT", "DROP"]:
                policy = "DROP"
        return "direct" if policy == "ACCEPT" else "blocked"

    def _limited_rules(self) -> list[tuple[str, list[str]]]:
        return self._record_rules(
            build_iptables_script(
                allowed_hosts={self.HOST},
                dnat_hosts=[self.HOST],
                dnat_target=("aios-worker", self.PROXY_PORT),
                dns_port=self.DNS_PORT,
            )
        )

    def _unrestricted_rules(self) -> list[tuple[str, list[str]]]:
        return self._record_rules(
            build_secret_egress_dnat_script(
                dnat_hosts=[self.HOST],
                dnat_target=("aios-worker", self.PROXY_PORT),
                dns_port=self.DNS_PORT,
            )
        )

    # ── resolution: the name no longer yields ANY real address ────────────────

    def test_credential_host_resolves_only_to_the_sentinel(self) -> None:
        """Inside the sandbox the credential name resolves to the sentinel — no
        matter what the live pool holds, sampled or not (#2042)."""
        got = self._resolve_in_sandbox(self.HOST, pool=[self.UNSAMPLED_IP, self.SAMPLED_IP])
        assert got == CREDENTIAL_SENTINEL_IP

    def test_non_credential_host_resolution_is_untouched(self) -> None:
        """Ordinary names still resolve normally — the resolver forwards them
        verbatim, so this is interception, not a general DNS blackhole."""
        got = self._resolve_in_sandbox("plain.example.com", pool=[self.UNSAMPLED_IP])
        assert got == self.UNSAMPLED_IP

    def test_aaaa_and_https_records_cannot_smuggle_a_real_address(self) -> None:
        """``AAAA``/``HTTPS`` for a credential host are NODATA.

        An ``AAAA``, or the ``ipv4hint`` in an ``HTTPS``/``SVCB`` record, would
        hand the sandbox a real pool address behind the resolver's back — an
        IP-keyed hole reopened through a different record type.
        """

        async def _ask(qtype: int) -> bytes:
            resolver = CredentialDnsResolver([self.HOST], upstream=None)
            query = (
                struct.pack("!HHHHHH", 0x1234, 0x0100, 1, 0, 0, 0)
                + b"".join(bytes([len(x)]) + x.encode() for x in self.HOST.split("."))
                + b"\x00"
                + struct.pack("!HH", qtype, 1)
            )
            return await resolver.answer(query)

        for qtype in (28, 65, 64):  # AAAA, HTTPS, SVCB
            response = asyncio.run(_ask(qtype))
            (ancount,) = struct.unpack_from("!H", response, 6)
            assert ancount == 0, f"qtype {qtype} returned an answer record"
            assert struct.unpack_from("!H", response, 2)[0] & 0x000F == 0  # NOERROR

    def test_query_name_case_cannot_evade_the_match(self) -> None:
        """DNS 0x20 case randomization must not slip a credential host past."""
        got = self._resolve_in_sandbox("API.SeCrEt.CoM", pool=[self.UNSAMPLED_IP])
        assert got == CREDENTIAL_SENTINEL_IP

    # ── the verdict: what actually happens to the packet ──────────────────────

    def test_limited_credential_host_is_proxied(self) -> None:
        rules = self._limited_rules()
        dest = self._resolve_in_sandbox(self.HOST, pool=[self.UNSAMPLED_IP])
        assert dest is not None
        assert self._verdict(rules, dest) == "proxied"

    def test_unrestricted_credential_host_is_proxied(self) -> None:
        rules = self._unrestricted_rules()
        dest = self._resolve_in_sandbox(self.HOST, pool=[self.UNSAMPLED_IP])
        assert dest is not None
        assert self._verdict(rules, dest) == "proxied"

    # ── the unsampled address: the whole point of #2042 ───────────────────────

    def test_unrestricted_unsampled_address_is_proxied(self) -> None:
        """FLIPPED by #2042 — this used to assert the BUG on purpose.

        Before: under Unrestricted the filter policy stayed ACCEPT and both the
        DNAT and the fence were generated only for LEARNED addresses, so an
        address that never appeared in a DNS sample matched nothing and left
        via the default-ACCEPT policy carrying the literal placeholder. The
        old assertion was ``verdict == "direct"``, pinned as a deliberately
        failing-safety expectation and named as this issue's acceptance signal.

        After: the sandbox reaches the credential host by NAME, that name
        resolves ONLY to the sentinel, and the sentinel's only route out is the
        DNAT to the secret-egress proxy. Here the upstream pool contains ONLY
        the address no sampler ever returned — the case that used to fail open —
        and the verdict is ``proxied``.
        """
        rules = self._unrestricted_rules()
        dest = self._resolve_in_sandbox(self.HOST, pool=[self.UNSAMPLED_IP])
        assert dest == CREDENTIAL_SENTINEL_IP, (
            "the sandbox learned a real pool address for a credential host"
        )
        assert self._verdict(rules, dest) == "proxied", (
            "an address no sampler ever saw must still be proxied (#2042)"
        )
        # The destination-side floor is independent of the DNS path: a raw,
        # cached, DoH, or /etc/hosts address is intercepted too.
        assert self._verdict(rules, self.UNSAMPLED_IP) == "proxied"
        assert self._verdict(rules, self.SAMPLED_IP) == "proxied"

    def test_limited_unsampled_address_is_proxied_not_merely_dropped(self) -> None:
        """Limited was already safe (its terminal ``-P OUTPUT DROP`` caught the
        unsampled address), but safe-by-denial meant credential requests to an
        unsampled address simply failed. Name-based interception upgrades that
        from ``blocked`` to ``proxied`` — and #2042 must not regress Limited."""
        rules = self._limited_rules()
        dest = self._resolve_in_sandbox(self.HOST, pool=[self.UNSAMPLED_IP])
        assert dest is not None
        assert self._verdict(rules, dest) == "proxied"

    def test_no_credential_rule_is_keyed_on_a_resolved_address(self) -> None:
        """The property behind #2042, inverted.

        The old assertion was that credential-host rule coverage equalled the
        SAMPLED SET — which is why no IP-keyed rule could ever close the hole.
        Now NO credential rule references a resolved address at all: the only
        destination-keyed credential rules name our own sentinel constant, so
        coverage is a property of the ruleset rather than of what DNS returned.
        """
        for rules in (self._unrestricted_rules(), self._limited_rules()):
            credential_dests = {
                argv[argv.index("-d") + 1]
                for table, argv in rules
                if table == "nat" and "DNAT" in argv and "-d" in argv
            }
            assert credential_dests == {CREDENTIAL_SENTINEL_IP}
            assert self.SAMPLED_IP not in credential_dests
            assert self.UNSAMPLED_IP not in credential_dests

    def test_reserved_sentinel_and_filter_reject_fail_closed(self) -> None:
        """The TEST-NET sentinel cannot identify a real credential endpoint.

        A missing/malformed DNAT reaches the surviving filter REJECT rather
        than a real upstream. Verify by dropping the nat table entirely.
        """
        assert CREDENTIAL_SENTINEL_IP.startswith("192.0.2.")
        filter_only = [(t, argv) for t, argv in self._unrestricted_rules() if t != "nat"]
        assert self._verdict(filter_only, CREDENTIAL_SENTINEL_IP) == "blocked"
        # Non-443 sentinel traffic (e.g. :80) is refused in both modes.
        for rules in (self._unrestricted_rules(), self._limited_rules()):
            assert self._verdict(rules, CREDENTIAL_SENTINEL_IP, dport=80) == "blocked"

    def test_dns_interception_precedes_any_in_netns_resolver(self) -> None:
        """The :53 DNAT is INSERTED at the top of nat OUTPUT.

        Appending it would leave Docker's embedded resolver (127.0.0.11, whose
        own DNAT Docker installs in this chain) free to answer a credential
        name first with a real pool address.
        """
        for rules in (self._unrestricted_rules(), self._limited_rules()):
            dns_rules = [
                argv
                for table, argv in rules
                if table == "nat" and "--dport" in argv and argv[argv.index("--dport") + 1] == "53"
            ]
            assert dns_rules, "no DNS interception rule emitted"
            for argv in dns_rules:
                assert "-I" in argv, f"DNS interception appended, not inserted: {argv}"
