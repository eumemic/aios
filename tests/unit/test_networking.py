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
            extra_host_ports=[("aios-worker", 8765)],
        )
        assert "aios-worker:8765" in script
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
            extra_host_ports=[("aios-worker", 8765)],
        )

        script = backend.calls[0][1]["command"]
        assert "aios-worker:8765" in script


# ── lockdown fails closed (security gate, not best-effort) ─────────────────────


class TestApplyNetworkLockdownFailsClosed:
    """A :class:`Limited` sandbox whose iptables lockdown didn't apply is a
    silent unrestricted-networking bypass — especially dangerous combined
    with the per-environment image override (#724), where a tenant-supplied
    image might lack ``iptables``/``getent``. The lockdown is a security
    gate, so a nonzero exit (or a backend exec error) must raise rather than
    log-and-continue, letting the registry tear the sandbox down.
    """

    @pytest.mark.asyncio
    async def test_nonzero_exit_raises_sandbox_backend_error(self) -> None:
        backend = FakeBackend()
        backend.next_result = CommandResult(
            exit_code=3,
            stdout="",
            stderr="iptables: command not found",
            timed_out=False,
            truncated=False,
        )
        handle = make_handle()
        networking = LimitedNetworking(type="limited", allowed_hosts=["api.example.com"])

        with pytest.raises(SandboxBackendError, match="network lockdown failed"):
            await apply_network_lockdown(backend, handle, networking)

    @pytest.mark.asyncio
    async def test_backend_exec_error_propagates(self) -> None:
        """An infra failure to even run the lockdown script must not be
        swallowed into a wide-open sandbox — it propagates so the provision
        aborts."""
        backend = FakeBackend()
        backend.exec = AsyncMock(  # type: ignore[method-assign]
            side_effect=SandboxBackendError("daemon hiccup")
        )
        handle = make_handle()
        networking = LimitedNetworking(type="limited", allowed_hosts=["api.example.com"])

        with pytest.raises(SandboxBackendError, match="daemon hiccup"):
            await apply_network_lockdown(backend, handle, networking)

    @pytest.mark.asyncio
    async def test_zero_exit_does_not_raise(self) -> None:
        """The happy path (exit 0) is unchanged — no exception."""
        backend = FakeBackend()  # default exec returns exit 0
        handle = make_handle()
        networking = LimitedNetworking(type="limited", allowed_hosts=["api.example.com"])

        await apply_network_lockdown(backend, handle, networking)  # must not raise
