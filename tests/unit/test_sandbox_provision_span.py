"""Unit tests for the ``sandbox_provision_*`` span pair (issue #78)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.sandbox.registry import SandboxRegistry


@pytest.fixture
def fake_handle() -> SimpleNamespace:
    return SimpleNamespace(
        session_id="sess_01TEST",
        container_id="abc123def456abc123def456",
        workspace_path="/tmp/w",
    )


class TestSandboxProvisionSpan:
    async def test_cold_start_emits_span_pair(self, fake_handle: SimpleNamespace) -> None:
        registry = SandboxRegistry()
        pool = MagicMock()
        span_start = SimpleNamespace(id="ev_span_start")
        append_event = AsyncMock(return_value=span_start)

        with (
            patch(
                "aios.sandbox.registry.provision_for_session",
                AsyncMock(return_value=fake_handle),
            ),
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

    async def test_warm_hit_emits_no_span(self, fake_handle: SimpleNamespace) -> None:
        registry = SandboxRegistry()
        registry._handles["sess_01TEST"] = fake_handle  # type: ignore[assignment]
        pool = MagicMock()
        append_event = AsyncMock()

        with patch("aios.services.sessions.append_event", append_event):
            result = await registry.get_or_provision("sess_01TEST", pool=pool)

        assert result is fake_handle
        append_event.assert_not_awaited()

    async def test_no_pool_no_span(self, fake_handle: SimpleNamespace) -> None:
        """When pool is not passed (e.g. worker startup paths), no span emission."""
        registry = SandboxRegistry()
        append_event = AsyncMock()

        with (
            patch(
                "aios.sandbox.registry.provision_for_session",
                AsyncMock(return_value=fake_handle),
            ),
            patch("aios.services.sessions.append_event", append_event),
        ):
            await registry.get_or_provision("sess_01TEST")

        append_event.assert_not_awaited()

    async def test_provision_failure_emits_error_end_span(self) -> None:
        registry = SandboxRegistry()
        pool = MagicMock()
        span_start = SimpleNamespace(id="ev_span_start")
        append_event = AsyncMock(return_value=span_start)

        class ProvisionError(Exception):
            pass

        with (
            patch(
                "aios.sandbox.registry.provision_for_session",
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
