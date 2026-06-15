"""Unit tests for networking model validation and sandbox lockdown logic."""

from __future__ import annotations

import asyncio
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
    Limited,
    Mount,
    SandboxBackendError,
    SandboxSpec,
    Unrestricted,
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
            allow_mcp_servers=False,
        )
        assert n.type == "limited"
        assert n.allowed_hosts == ["api.example.com", "cdn.example.com"]
        assert n.allow_package_managers is True
        assert n.allow_mcp_servers is False

    def test_defaults(self) -> None:
        n = LimitedNetworking(type="limited")
        assert n.allowed_hosts == []
        assert n.allow_package_managers is False
        assert n.allow_mcp_servers is False

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
        assert "getent ahosts api.example.com" in script
        assert "getent ahosts cdn.example.com" in script

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

    def test_no_bare_iptables_invocations(self) -> None:
        """Every netfilter command goes through the ``$IPT`` selector so apply
        and verify agree on the backend; a bare ``iptables ...`` invocation
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
                f"bare iptables invocation would use the nft default: {line!r}"
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
        assert "getent ahosts api.secret.com" in script
        assert "getent ahosts data.secret.com" in script
        assert "PROXY_IP=$(getent ahosts aios-worker" in script

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
        assert "getent ahosts plain.example.com" in script
        # Only the credential host is DNAT'd; the plain allowed host is not.
        assert script.count("-j DNAT --to-destination") == 1

    def test_dnat_only_on_443_not_80(self) -> None:
        script = build_iptables_script(
            allowed_hosts=set(),
            dnat_hosts=["api.secret.com", "data.secret.com"],
            dnat_target=("aios-worker", 49152),
        )
        assert "--dport 80 -j DNAT" not in script


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
        assert "getent ahosts api.secret.com" in script
        assert "getent ahosts data.secret.com" in script
        assert "PROXY_IP=$(getent ahosts aios-worker" in script

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
                f"bare iptables invocation would use the nft default: {line!r}"
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

    def test_assert_drop_false_omits_drop_assertion(self) -> None:
        # The DNAT-only Unrestricted path (#1153) leaves the filter policy at
        # ACCEPT, so the verify must NOT assert a DROP policy (it would always
        # fail) — but still asserts nat DNAT coverage.
        script = build_lockdown_verify_script(dnat_hosts=["api.secret.com"], assert_drop=False)
        assert "OUTPUT DROP" not in script
        assert "\"$IPT\" -t nat -S OUTPUT | grep -q -- '-j DNAT'" in script

    def test_assert_drop_true_is_default(self) -> None:
        # Backward-compat: the Limited callers pass no assert_drop and must keep
        # getting the DROP assertion.
        assert "OUTPUT DROP" in build_lockdown_verify_script(dnat_hosts=["api.secret.com"])


# ── docker backend translates network policy to docker run argv ────────────────


def _make_spec(network_policy: Limited | Unrestricted) -> SandboxSpec:
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
            _make_spec(Limited(allowed_hosts=frozenset({"example.com"})))
        )
        assert "NET_ADMIN" not in argv

    @pytest.mark.asyncio
    async def test_unrestricted_no_cap_net_admin(self) -> None:
        argv = await _capture_docker_argv(_make_spec(Unrestricted()))
        assert "--cap-add" not in argv

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "policy",
        [
            Limited(allowed_hosts=frozenset({"example.com"})),
            Unrestricted(),
        ],
        ids=["limited", "unrestricted"],
    )
    async def test_security_opt_no_new_privileges(self, policy: Limited | Unrestricted) -> None:
        argv = await _capture_docker_argv(_make_spec(policy))
        assert "--security-opt" in argv
        i = argv.index("--security-opt")
        assert argv[i + 1] == "no-new-privileges"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "policy",
        [
            Limited(allowed_hosts=frozenset({"example.com"})),
            Unrestricted(),
        ],
        ids=["limited", "unrestricted"],
    )
    async def test_ipc_private(self, policy: Limited | Unrestricted) -> None:
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
        # Two sidecar invocations: apply, then read-back verify.
        assert len(scripts) == 2
        apply_script, verify_script = scripts
        assert '"$IPT" -P OUTPUT DROP' in apply_script
        assert "getent ahosts api.example.com" in apply_script
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
        assert "getent ahosts api.secret.com" in apply_script
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
        invocations (apply and read-back verify) so the lockdown runs under
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
        assert "getent ahosts api.secret.com" in apply_script
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
