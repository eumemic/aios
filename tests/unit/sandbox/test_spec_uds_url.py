"""Spec-builder selection between TCP and Unix-socket broker transports.

When ``settings.mcp_broker_socket_path`` is set, ``build_spec_from_session``
must point sandboxes at ``unix:///var/run/aios/mcp-broker.sock`` and
attach a bind mount of the host socket file at that path. When unset,
the sandbox keeps using the ephemeral TCP URL — no socket Mount.

The plan-assembly function ``_assemble_plan`` is the single point that
decides MCP_BROKER_URL and the socket Mount; these tests pin both
selection branches there.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from aios.sandbox.spec import MCP_BROKER_SOCKET_SANDBOX_PATH, _assemble_plan


def _call_assemble(
    *,
    mcp_broker_url: str,
    mcp_socket_host_path: Path | None,
) -> object:
    # ``_assemble_plan`` imports its volume helpers function-locally, so
    # we patch them at the ``aios.sandbox.volumes`` source module.
    with (
        patch(
            "aios.sandbox.volumes.ensure_session_attachments_dir",
            return_value=Path("/tmp/a"),
        ),
        patch(
            "aios.sandbox.volumes.ensure_session_uploads_dir",
            return_value=Path("/tmp/u"),
        ),
    ):
        return _assemble_plan(
            session_id="sess_01TEST",
            instance_id="inst_TEST",
            image="aios-sandbox:test",
            workspace_path=Path("/tmp/w"),
            env_config=None,
            session_env={},
            memory_echoes=[],
            github_echoes=[],
            git_proxy=None,
            mcp_broker_url=mcp_broker_url,
            mcp_broker_secret="secret123",
            mcp_socket_host_path=mcp_socket_host_path,
        )


class TestSpecMcpBrokerUrlSelection:
    def test_uses_unix_url_when_socket_path_set(self) -> None:
        sock_path = Path("/var/run/aios/mcp-broker.sock")
        plan = _call_assemble(
            mcp_broker_url=f"unix://{MCP_BROKER_SOCKET_SANDBOX_PATH}",
            mcp_socket_host_path=sock_path,
        )
        env = plan.spec.environment
        assert env["MCP_BROKER_URL"] == f"unix://{MCP_BROKER_SOCKET_SANDBOX_PATH}"
        # And a socket Mount is appended.
        sock_mounts = [
            m for m in plan.spec.extra_mounts if m.sandbox_path == MCP_BROKER_SOCKET_SANDBOX_PATH
        ]
        assert len(sock_mounts) == 1
        m = sock_mounts[0]
        assert m.host_path == sock_path
        assert m.read_only is False

    def test_uses_http_url_when_socket_path_none(self) -> None:
        """No socket Mount is attached when broker is TCP-only."""
        plan = _call_assemble(
            mcp_broker_url="http://aios-worker:54321",
            mcp_socket_host_path=None,
        )
        env = plan.spec.environment
        assert env["MCP_BROKER_URL"].startswith("http://aios-worker:")
        sock_mounts = [
            m for m in plan.spec.extra_mounts if m.sandbox_path == MCP_BROKER_SOCKET_SANDBOX_PATH
        ]
        assert sock_mounts == []
