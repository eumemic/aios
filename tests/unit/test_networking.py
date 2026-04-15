"""Unit tests for networking model validation and provisioner logic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from aios.models.environments import (
    EnvironmentConfig,
    LimitedNetworking,
    UnrestrictedNetworking,
)
from aios.sandbox.container import CommandResult, ContainerHandle
from aios.sandbox.provisioner import PACKAGE_REGISTRY_HOSTS, build_iptables_script

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

    def test_unrestricted(self) -> None:
        config = EnvironmentConfig.model_validate({"networking": {"type": "unrestricted"}})
        assert isinstance(config.networking, UnrestrictedNetworking)

    def test_limited(self) -> None:
        config = EnvironmentConfig.model_validate(
            {"networking": {"type": "limited", "allowed_hosts": ["api.example.com"]}}
        )
        assert isinstance(config.networking, LimitedNetworking)
        assert config.networking.allowed_hosts == ["api.example.com"]

    def test_null_networking_is_none(self) -> None:
        config = EnvironmentConfig.model_validate({"networking": None})
        assert config.networking is None

    def test_empty_dict_networking_normalized_to_none(self) -> None:
        """Backward compat: existing DB rows may have networking: {}."""
        config = EnvironmentConfig.model_validate({"networking": {}})
        assert config.networking is None

    def test_missing_networking_key(self) -> None:
        """Backward compat: existing DB rows may have no networking key."""
        config = EnvironmentConfig.model_validate({"packages": {"pip": ["pandas"]}})
        assert config.networking is None

    def test_rejects_bogus_type(self) -> None:
        with pytest.raises(ValueError):
            EnvironmentConfig.model_validate({"networking": {"type": "bogus"}})


# ── build_iptables_script ─────────────────────────────────────────────────────


class TestBuildIptablesScript:
    def test_empty_hosts(self) -> None:
        script = build_iptables_script(set())
        assert "iptables -F OUTPUT" in script
        assert "iptables -P OUTPUT DROP" in script
        # No host-specific rules
        assert "getent" not in script

    def test_single_host(self) -> None:
        script = build_iptables_script({"api.example.com"})
        assert "getent ahosts api.example.com" in script
        assert "--dport 80" in script
        assert "--dport 443" in script

    def test_multiple_hosts_sorted(self) -> None:
        script = build_iptables_script({"z.example.com", "a.example.com"})
        z_pos = script.index("z.example.com")
        a_pos = script.index("a.example.com")
        assert a_pos < z_pos, "hosts should be sorted alphabetically"

    def test_allows_dns(self) -> None:
        script = build_iptables_script({"example.com"})
        assert "--dport 53" in script

    def test_allows_loopback(self) -> None:
        script = build_iptables_script({"example.com"})
        assert "-o lo -j ACCEPT" in script

    def test_allows_established(self) -> None:
        script = build_iptables_script({"example.com"})
        assert "ESTABLISHED,RELATED" in script

    def test_package_registry_hosts_integration(self) -> None:
        """Verify PACKAGE_REGISTRY_HOSTS can be passed directly."""
        script = build_iptables_script(PACKAGE_REGISTRY_HOSTS)
        assert "pypi.org" in script
        assert "registry.npmjs.org" in script


# ── provisioner docker args ───────────────────────────────────────────────────


def _make_handle(session_id: str = "sess_01TEST") -> ContainerHandle:
    handle = ContainerHandle(
        session_id=session_id,
        container_id="abc123",
        workspace_path=Path("/tmp/ws"),
    )
    handle.run_command = AsyncMock(  # type: ignore[method-assign]
        return_value=CommandResult(
            exit_code=0, stdout="", stderr="", timed_out=False, truncated=False
        )
    )
    return handle


class TestProvisionerDockerArgs:
    """Test that provision_for_session passes the right docker args."""

    @pytest.mark.asyncio
    async def test_limited_adds_cap_net_admin(self) -> None:
        limited_config = EnvironmentConfig(
            networking=LimitedNetworking(type="limited", allowed_hosts=["example.com"])
        )
        captured_argv: list[list[str]] = []

        async def fake_run_docker(argv: list[str]) -> tuple[int, bytes, bytes]:
            captured_argv.append(argv)
            # Return a container id for docker run
            return 0, b"container_abc123\n", b""

        with (
            patch(
                "aios.sandbox.provisioner._load_environment_config",
                AsyncMock(return_value=limited_config),
            ),
            patch("aios.sandbox.provisioner._run_docker", fake_run_docker),
            patch(
                "aios.sandbox.provisioner._load_session_provisioning",
                AsyncMock(return_value=("/tmp/ws", {})),
            ),
            patch("aios.sandbox.volumes.ensure_workspace_path", return_value=Path("/tmp/ws")),
            patch(
                "aios.sandbox.provisioner._install_packages",
                AsyncMock(),
            ) as mock_install,
            patch(
                "aios.sandbox.provisioner._apply_network_lockdown",
                AsyncMock(),
            ) as mock_lockdown,
        ):
            from aios.sandbox.provisioner import provision_for_session

            await provision_for_session("sess_01TEST")

        # docker run argv should contain --cap-add NET_ADMIN
        run_argv = captured_argv[0]
        assert "--cap-add" in run_argv
        cap_idx = run_argv.index("--cap-add")
        assert run_argv[cap_idx + 1] == "NET_ADMIN"

        # install should be called before lockdown
        mock_install.assert_called_once()
        mock_lockdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_unrestricted_no_cap_net_admin(self) -> None:
        unrestricted_config = EnvironmentConfig(networking=UnrestrictedNetworking())
        captured_argv: list[list[str]] = []

        async def fake_run_docker(argv: list[str]) -> tuple[int, bytes, bytes]:
            captured_argv.append(argv)
            return 0, b"container_abc123\n", b""

        with (
            patch(
                "aios.sandbox.provisioner._load_environment_config",
                AsyncMock(return_value=unrestricted_config),
            ),
            patch("aios.sandbox.provisioner._run_docker", fake_run_docker),
            patch(
                "aios.sandbox.provisioner._load_session_provisioning",
                AsyncMock(return_value=("/tmp/ws", {})),
            ),
            patch("aios.sandbox.volumes.ensure_workspace_path", return_value=Path("/tmp/ws")),
            patch("aios.sandbox.provisioner._install_packages", AsyncMock()),
            patch(
                "aios.sandbox.provisioner._apply_network_lockdown",
                AsyncMock(),
            ) as mock_lockdown,
        ):
            from aios.sandbox.provisioner import provision_for_session

            await provision_for_session("sess_01TEST")

        run_argv = captured_argv[0]
        assert "--cap-add" not in run_argv
        mock_lockdown.assert_not_called()

    @pytest.mark.asyncio
    async def test_none_config_no_lockdown(self) -> None:
        """No environment config at all — should work like unrestricted."""
        captured_argv: list[list[str]] = []

        async def fake_run_docker(argv: list[str]) -> tuple[int, bytes, bytes]:
            captured_argv.append(argv)
            return 0, b"container_abc123\n", b""

        with (
            patch(
                "aios.sandbox.provisioner._load_environment_config",
                AsyncMock(return_value=None),
            ),
            patch("aios.sandbox.provisioner._run_docker", fake_run_docker),
            patch(
                "aios.sandbox.provisioner._load_session_provisioning",
                AsyncMock(return_value=("/tmp/ws", {})),
            ),
            patch("aios.sandbox.volumes.ensure_workspace_path", return_value=Path("/tmp/ws")),
            patch("aios.sandbox.provisioner._install_packages", AsyncMock()),
            patch(
                "aios.sandbox.provisioner._apply_network_lockdown",
                AsyncMock(),
            ) as mock_lockdown,
        ):
            from aios.sandbox.provisioner import provision_for_session

            await provision_for_session("sess_01TEST")

        run_argv = captured_argv[0]
        assert "--cap-add" not in run_argv
        mock_lockdown.assert_not_called()


class TestApplyNetworkLockdown:
    """Test _apply_network_lockdown builds and executes the right script."""

    @pytest.mark.asyncio
    async def test_calls_run_command_with_script(self) -> None:
        from aios.sandbox.provisioner import _apply_network_lockdown

        handle = _make_handle()
        networking = LimitedNetworking(type="limited", allowed_hosts=["api.example.com"])

        await _apply_network_lockdown(handle, networking, "sess_01TEST")

        handle.run_command.assert_called_once()  # type: ignore[union-attr]
        script = handle.run_command.call_args[0][0]  # type: ignore[union-attr]
        assert "iptables -P OUTPUT DROP" in script
        assert "getent ahosts api.example.com" in script

    @pytest.mark.asyncio
    async def test_includes_package_registries_when_enabled(self) -> None:
        from aios.sandbox.provisioner import _apply_network_lockdown

        handle = _make_handle()
        networking = LimitedNetworking(
            type="limited",
            allowed_hosts=["api.example.com"],
            allow_package_managers=True,
        )

        await _apply_network_lockdown(handle, networking, "sess_01TEST")

        script = handle.run_command.call_args[0][0]  # type: ignore[union-attr]
        assert "pypi.org" in script
        assert "registry.npmjs.org" in script
        assert "api.example.com" in script

    @pytest.mark.asyncio
    async def test_no_package_registries_when_disabled(self) -> None:
        from aios.sandbox.provisioner import _apply_network_lockdown

        handle = _make_handle()
        networking = LimitedNetworking(
            type="limited",
            allowed_hosts=["api.example.com"],
            allow_package_managers=False,
        )

        await _apply_network_lockdown(handle, networking, "sess_01TEST")

        script = handle.run_command.call_args[0][0]  # type: ignore[union-attr]
        assert "pypi.org" not in script
        assert "api.example.com" in script
