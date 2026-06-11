"""Unit tests for the ``sandbox_provision_*`` span pair (issue #78)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.sandbox.backends.base import (
    Mount,
    SandboxSpec,
    Unrestricted,
)
from aios.sandbox.registry import SandboxRegistry
from aios.sandbox.spec import ProvisioningPlan
from tests.helpers.sandbox import FakeBackend, make_handle


def _make_plan() -> ProvisioningPlan:
    spec = SandboxSpec(
        session_id="sess_01TEST",
        instance_id="inst_TEST",
        workspace=Mount(host_path=Path("/tmp/w"), sandbox_path="/workspace"),
        extra_mounts=(),
        environment={},
        labels={},
        network_policy=Unrestricted(),
        host_gateway_alias=None,
        image="aios-sandbox:test",
    )
    return ProvisioningPlan(
        spec=spec,
        env_config=None,
        memory_echoes=[],
        github_echoes=[],
        git_proxy=None,
    )


class TestSandboxProvisionSpan:
    async def test_cold_start_emits_span_pair(self) -> None:
        backend = FakeBackend(next_handle_id="abc123def456abc123def456")
        registry = SandboxRegistry(backend=backend)
        pool = MagicMock()
        span_start = SimpleNamespace(id="ev_span_start")
        append_event = AsyncMock(return_value=span_start)

        with (
            patch(
                "aios.sandbox.registry.build_spec_from_session",
                AsyncMock(return_value=_make_plan()),
            ),
            patch("aios.sandbox.registry.ensure_workspace_runtime_dirs", AsyncMock()),
            patch("aios.sandbox.registry.install_egress_ca", AsyncMock()),
            patch("aios.sandbox.registry.install_packages", AsyncMock()),
            patch("aios.sandbox.registry.apply_network_lockdown", AsyncMock()),
            patch("aios.services.sessions.append_event", append_event),
        ):
            await registry.get_or_provision("sess_01TEST", pool=pool)

        assert append_event.await_count == 2
        start_data = append_event.await_args_list[0].args[3]
        end_data = append_event.await_args_list[1].args[3]
        assert start_data == {"event": "sandbox_provision_start"}
        assert end_data["event"] == "sandbox_provision_end"
        assert end_data["sandbox_provision_start_id"] == "ev_span_start"
        assert end_data["is_error"] is False
        assert end_data["container_id"] == "abc123def456"  # 12-char short id

    async def test_warm_hit_emits_no_span(self) -> None:
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        cached = make_handle(sandbox_id="warmcache_id", spec_version=0)
        registry._handles["sess_01TEST"] = cached
        pool = MagicMock()
        append_event = AsyncMock()

        # The warm path now also probes ``sessions.spec_version`` (#713);
        # report no drift so the cached handle is returned without a span.
        with (
            patch("aios.services.sessions.append_event", append_event),
            patch(
                "aios.sandbox.registry.queries.unscoped_get_session_spec_version",
                AsyncMock(return_value=0),
            ),
        ):
            result = await registry.get_or_provision("sess_01TEST", pool=pool)

        assert result is cached
        append_event.assert_not_awaited()

    async def test_no_pool_no_span(self) -> None:
        """When pool is not passed (e.g. worker startup paths), no span emission."""
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        append_event = AsyncMock()

        with (
            patch(
                "aios.sandbox.registry.build_spec_from_session",
                AsyncMock(return_value=_make_plan()),
            ),
            patch("aios.sandbox.registry.ensure_workspace_runtime_dirs", AsyncMock()),
            patch("aios.sandbox.registry.install_egress_ca", AsyncMock()),
            patch("aios.sandbox.registry.install_packages", AsyncMock()),
            patch("aios.sandbox.registry.apply_network_lockdown", AsyncMock()),
            patch("aios.services.sessions.append_event", append_event),
        ):
            await registry.get_or_provision("sess_01TEST")

        append_event.assert_not_awaited()

    async def test_provision_failure_emits_error_end_span(self) -> None:
        backend = FakeBackend()
        registry = SandboxRegistry(backend=backend)
        pool = MagicMock()
        span_start = SimpleNamespace(id="ev_span_start")
        append_event = AsyncMock(return_value=span_start)

        class ProvisionError(Exception):
            pass

        with (
            patch(
                "aios.sandbox.registry.build_spec_from_session",
                AsyncMock(side_effect=ProvisionError("docker exploded")),
            ),
            patch("aios.services.sessions.append_event", append_event),
            pytest.raises(ProvisionError),
        ):
            await registry.get_or_provision("sess_01TEST", pool=pool)

        assert append_event.await_count == 2
        end_data = append_event.await_args_list[1].args[3]
        assert end_data["event"] == "sandbox_provision_end"
        assert end_data["is_error"] is True
        assert "container_id" not in end_data
