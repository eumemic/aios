"""Unit tests for networking model validation and sandbox lockdown logic."""

from __future__ import annotations

import asyncio
import os
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

    def test_dnat_rule_emitted_per_credential_host_on_443(self) -> None:
        script = build_iptables_script(
            allowed_hosts=set(),
            dnat_hosts=["api.secret.com", "data.secret.com"],
            dnat_target=("aios-worker", 49152),
        )
        assert '"$IPT" -t nat -A OUTPUT' in script
        assert '--dport 443 -j DNAT --to-destination "$PROXY_IP:49152"' in script
        assert "resolve_ipv4 api.secret.com" in script
        assert "resolve_ipv4 data.secret.com" in script
        assert "PROXY_IP=$(resolve_ipv4 aios-worker" in script

    def test_dnat_not_redirect(self) -> None:
        script = build_iptables_script(
            allowed_hosts=set(),
            dnat_hosts=["api.secret.com", "data.secret.com"],
            dnat_target=("aios-worker", 49152),
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
        )
        assert '"$IPT" -A OUTPUT -d "$ip" -p tcp --dport 443 -j ACCEPT' in script
        assert "resolve_ipv4 plain.example.com" in script
        # Only the credential host is DNAT'd; the plain allowed host is not.
        assert script.count("-j DNAT --to-destination") == 1

    def test_dnat_only_on_443_not_80(self) -> None:
        script = build_iptables_script(
            allowed_hosts=set(),
            dnat_hosts=["api.secret.com", "data.secret.com"],
            dnat_target=("aios-worker", 49152),
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
        )
        assert '"$IP6T" -P OUTPUT DROP' in script

    def test_dnat_only_script_has_no_v6_drop(self) -> None:
        """The Unrestricted DNAT-only path leaves general egress open, so it
        must NOT install a v6 DROP (which would deny v6 egress in an otherwise
        open box)."""
        script = build_secret_egress_dnat_script(
            dnat_hosts=["api.secret.com"], dnat_target=("aios-worker", 49152)
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
            ),
            "dnat_only": build_secret_egress_dnat_script(
                dnat_hosts=["api.secret.com"], dnat_target=("aios-worker", 49152)
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

    def test_dnat_loop_and_proxy_alias_use_helper(self) -> None:
        script = build_iptables_script(
            allowed_hosts=set(),
            dnat_hosts=["api.secret.com"],
            dnat_target=("aios-worker", 49152),
        )
        assert "PROXY_IP=$(resolve_ipv4 aios-worker | head -n1)" in script
        assert "for ip in $(resolve_ipv4 api.secret.com); do" in script

    def test_dnat_only_script_defines_and_uses_helper(self) -> None:
        script = build_secret_egress_dnat_script(
            dnat_hosts=["api.secret.com"], dnat_target=("aios-worker", 49152)
        )
        assert "resolve_ipv4()" in script
        assert "getent ahostsv4" in script
        assert "for ip in $(resolve_ipv4 api.secret.com); do" in script

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

    def test_emits_nat_dnat_rule(self) -> None:
        script = build_secret_egress_dnat_script(
            dnat_hosts=["api.secret.com", "data.secret.com"],
            dnat_target=("aios-worker", 49152),
        )
        assert '"$IPT" -t nat -A OUTPUT' in script
        assert '--dport 443 -j DNAT --to-destination "$PROXY_IP:49152"' in script
        assert "resolve_ipv4 api.secret.com" in script
        assert "resolve_ipv4 data.secret.com" in script
        assert "PROXY_IP=$(resolve_ipv4 aios-worker" in script

    def test_no_drop_policy(self) -> None:
        # The whole point: general egress stays open under Unrestricted.
        script = build_secret_egress_dnat_script(
            dnat_hosts=["api.secret.com"], dnat_target=("aios-worker", 49152)
        )
        assert "-P OUTPUT DROP" not in script

    def test_no_filter_accept_rules(self) -> None:
        # No per-host filter ACCEPTs and no loopback/DNS/established ACCEPTs —
        # the filter policy is left at its default ACCEPT (no lockdown).
        script = build_secret_egress_dnat_script(
            dnat_hosts=["api.secret.com"], dnat_target=("aios-worker", 49152)
        )
        assert "-A OUTPUT -o lo -j ACCEPT" not in script
        assert "--dport 53 -j ACCEPT" not in script
        assert "ESTABLISHED,RELATED -j ACCEPT" not in script
        assert "--dport 443 -j ACCEPT" not in script

    def test_flushes_nat_output_only_not_filter(self) -> None:
        script = build_secret_egress_dnat_script(
            dnat_hosts=["api.secret.com"], dnat_target=("aios-worker", 49152)
        )
        assert '"$IPT" -t nat -F OUTPUT' in script
        # The filter OUTPUT chain is deliberately NOT flushed (would disturb a
        # mode the operator left open).
        assert '"$IPT" -F OUTPUT' not in script

    def test_uses_selected_backend_no_bare_iptables(self) -> None:
        script = build_secret_egress_dnat_script(
            dnat_hosts=["api.secret.com"], dnat_target=("aios-worker", 49152)
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
            dnat_hosts=["api.secret.com"], dnat_target=("aios-worker", 49152)
        )
        lockdown = build_iptables_script(
            allowed_hosts=set(),
            dnat_hosts=["api.secret.com"],
            dnat_target=("aios-worker", 49152),
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

    def test_asserts_nat_dnat_coverage_when_dnat_hosts_present(self) -> None:
        """#984: a credential host whose getent returns zero IPs emits no DNAT
        rule and no error — apply exits 0 and a filter-only verify passes,
        silently running the session without DNAT. When dnat_hosts is non-empty
        the verify must ALSO assert the nat table carries a DNAT OUTPUT rule, so
        the zero-IP omission fails the verify (and thus the provision) instead of
        passing silently."""
        script = build_lockdown_verify_script(dnat_hosts=["api.secret.com"])
        assert "\"$IPT\" -t nat -S OUTPUT | grep -q -- '-j DNAT'" in script
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
        assert "\"$IPT\" -t nat -S OUTPUT | grep -q -- '-j DNAT'" in script

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
        )

        apply_script, verify_script = self._sidecar_scripts(backend)
        assert '"$IPT" -t nat -A OUTPUT' in apply_script
        assert '--dport 443 -j DNAT --to-destination "$PROXY_IP:49152"' in apply_script
        assert "resolve_ipv4 api.secret.com" in apply_script
        # #984: with dnat_hosts present, the read-back verify also asserts the
        # nat table carries a DNAT rule — a zero-IP host fails closed.
        assert "\"$IPT\" -t nat -S OUTPUT | grep -q -- '-j DNAT'" in verify_script

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
        )

        apply_script, verify_script = self._sidecar_scripts(backend)
        # DNAT installed, but general egress stays open (no DROP, no per-host ACCEPT).
        assert '--dport 443 -j DNAT --to-destination "$PROXY_IP:49152"' in apply_script
        assert "resolve_ipv4 api.secret.com" in apply_script
        assert "-P OUTPUT DROP" not in apply_script
        # The verify asserts nat DNAT coverage but NOT a DROP policy.
        assert "\"$IPT\" -t nat -S OUTPUT | grep -q -- '-j DNAT'" in verify_script
        assert "OUTPUT DROP" not in verify_script

    @pytest.mark.asyncio
    async def test_resolv_preamble_prepended_to_apply(self) -> None:
        # The apply script points the netns-joining sidecar at the embedded
        # resolver before any getent runs (same preamble as the Limited path).
        backend = FakeBackend()
        handle = make_handle()

        await apply_secret_egress_dnat(
            backend, handle, dnat_hosts=["api.secret.com"], dnat_target=("aios-worker", 49152)
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
                backend, handle, dnat_hosts=["api.secret.com"], dnat_target=("aios-worker", 49152)
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
                backend, handle, dnat_hosts=["api.secret.com"], dnat_target=("aios-worker", 49152)
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
                backend, handle, dnat_hosts=["api.secret.com"], dnat_target=("aios-worker", 49152)
            )

    @pytest.mark.asyncio
    async def test_zero_exit_does_not_raise(self) -> None:
        backend = FakeBackend()  # default sidecar returns exit 0 for both calls
        handle = make_handle()

        await apply_secret_egress_dnat(
            backend, handle, dnat_hosts=["api.secret.com"], dnat_target=("aios-worker", 49152)
        )  # must not raise
