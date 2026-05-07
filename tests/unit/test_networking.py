"""Unit tests for networking model validation and sandbox lockdown logic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from aios.models.environments import (
    EnvironmentConfig,
    LimitedNetworking,
    UnrestrictedNetworking,
)
from aios.sandbox.backends.base import (
    Limited,
    Mount,
    SandboxSpec,
    Unrestricted,
)
from aios.sandbox.backends.docker import DockerBackend
from aios.sandbox.setup import (
    PACKAGE_REGISTRY_HOSTS,
    apply_network_lockdown,
    build_iptables_script,
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
        assert "iptables -P OUTPUT DROP" in script

    def test_includes_each_allowed_host(self) -> None:
        script = build_iptables_script(allowed_hosts={"api.example.com", "cdn.example.com"})
        assert "getent ahosts api.example.com" in script
        assert "getent ahosts cdn.example.com" in script

    def test_loopback_and_dns_always_allowed(self) -> None:
        script = build_iptables_script(allowed_hosts=set())
        assert "iptables -A OUTPUT -o lo -j ACCEPT" in script
        assert "iptables -A OUTPUT -p udp --dport 53 -j ACCEPT" in script
        assert "iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT" in script

    def test_extra_host_ports_added(self) -> None:
        script = build_iptables_script(
            allowed_hosts=set(),
            extra_host_ports=[("host.docker.internal", 8765)],
        )
        assert "host.docker.internal:8765" in script
        assert "--dport 8765 -j ACCEPT" in script


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
        argv: list[str], *, timeout_s: float = 30.0
    ) -> tuple[int, bytes, bytes]:
        captured.append(argv)
        return 0, b"container_abc123\n", b""

    with patch("aios.sandbox.backends.docker._run_docker", fake_run_docker):
        await DockerBackend().create(spec)
    return captured[0]


class TestDockerBackendArgs:
    """The DockerBackend translates SandboxSpec.network_policy to the right argv."""

    @pytest.mark.asyncio
    async def test_limited_adds_cap_net_admin(self) -> None:
        argv = await _capture_docker_argv(
            _make_spec(Limited(allowed_hosts=frozenset({"example.com"})))
        )
        assert "--cap-add" in argv
        cap_idx = argv.index("--cap-add")
        assert argv[cap_idx + 1] == "NET_ADMIN"

    @pytest.mark.asyncio
    async def test_unrestricted_no_cap_net_admin(self) -> None:
        argv = await _capture_docker_argv(_make_spec(Unrestricted()))
        assert "--cap-add" not in argv


# ── network lockdown helper applies the right script via the backend ──────────


class TestApplyNetworkLockdown:
    """apply_network_lockdown builds the script and runs it via backend.exec."""

    @pytest.mark.asyncio
    async def test_calls_backend_exec_with_script(self) -> None:
        backend = FakeBackend()
        handle = make_handle()
        networking = LimitedNetworking(type="limited", allowed_hosts=["api.example.com"])

        await apply_network_lockdown(backend, handle, networking)

        exec_calls = [c for c in backend.calls if c[0] == "exec"]
        assert len(exec_calls) == 1
        script = exec_calls[0][1]["command"]
        assert "iptables -P OUTPUT DROP" in script
        assert "getent ahosts api.example.com" in script

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

        script = backend.calls[0][1]["command"]
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

        script = backend.calls[0][1]["command"]
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
            extra_host_ports=[("host.docker.internal", 8765)],
        )

        script = backend.calls[0][1]["command"]
        assert "host.docker.internal:8765" in script

    def test_package_registry_hosts_constant_is_populated(self) -> None:
        # Sanity: the constant exists and contains representative hosts.
        assert "pypi.org" in PACKAGE_REGISTRY_HOSTS
        assert "registry.npmjs.org" in PACKAGE_REGISTRY_HOSTS
