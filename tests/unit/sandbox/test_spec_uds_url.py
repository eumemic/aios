"""Spec-builder selection between TCP and Unix-socket broker transports.

When ``settings.tool_broker_socket_path`` is set, ``build_spec_from_session``
must point sandboxes at ``unix:///var/run/aios/tool-broker.sock`` and
attach a bind mount of the host socket file at that path. When unset,
the sandbox keeps using the ephemeral TCP URL — no socket Mount.

The plan-assembly function ``_assemble_plan`` is the single point that
decides TOOL_BROKER_URL and the socket Mount; these tests pin both
selection branches there.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from aios.sandbox.spec import (
    TOOL_BROKER_SECRET_SANDBOX_PATH,
    TOOL_BROKER_SOCKET_SANDBOX_PATH,
    _assemble_plan,
)


def _call_assemble(
    *,
    tool_broker_url: str,
    tool_socket_host_path: Path | None,
    tool_broker_secret: str = "secret123",
    session_id: str = "sess_01TEST",
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
            session_id=session_id,
            instance_id="inst_TEST",
            image="aios-sandbox:test",
            workspace_path=Path("/tmp/w"),
            env_config=None,
            session_env={},
            memory_echoes=[],
            github_echoes=[],
            git_proxy=None,
            tool_broker_url=tool_broker_url,
            tool_broker_secret=tool_broker_secret,
            tool_socket_host_path=tool_socket_host_path,
        )


class TestSpecToolBrokerUrlSelection:
    def test_uses_unix_url_when_socket_path_set(self, tmp_path: Path) -> None:
        # The path doesn't need to exist for the spec-assembly test, but
        # its parent directory must — the secret file is written next to
        # the socket file. ``tmp_path`` satisfies that for free.
        sock_path = tmp_path / "tool-broker.sock"
        plan = _call_assemble(
            tool_broker_url=f"unix://{TOOL_BROKER_SOCKET_SANDBOX_PATH}",
            tool_socket_host_path=sock_path,
        )
        env = plan.spec.environment
        assert env["TOOL_BROKER_URL"] == f"unix://{TOOL_BROKER_SOCKET_SANDBOX_PATH}"
        # And a socket Mount is appended.
        sock_mounts = [
            m for m in plan.spec.extra_mounts if m.sandbox_path == TOOL_BROKER_SOCKET_SANDBOX_PATH
        ]
        assert len(sock_mounts) == 1
        m = sock_mounts[0]
        assert m.host_path == sock_path
        assert m.read_only is False

    def test_uses_http_url_when_socket_path_none(self) -> None:
        """No socket Mount is attached when broker is TCP-only."""
        plan = _call_assemble(
            tool_broker_url="http://aios-worker:54321",
            tool_socket_host_path=None,
        )
        env = plan.spec.environment
        assert env["TOOL_BROKER_URL"].startswith("http://aios-worker:")
        sock_mounts = [
            m for m in plan.spec.extra_mounts if m.sandbox_path == TOOL_BROKER_SOCKET_SANDBOX_PATH
        ]
        assert sock_mounts == []


class TestSpecToolBrokerSecretBindMount:
    """The per-session ``TOOL_BROKER_SECRET`` is also written to the host
    next to the broker socket and bind-mounted read-only into the
    sandbox, so the in-sandbox ``bin/tool`` CLI can fall back to reading
    it from disk when env injection lapses (issue #698)."""

    def test_writes_secret_file_when_uds_set(self, tmp_path: Path) -> None:
        sock_path = tmp_path / "broker.sock"
        _call_assemble(
            tool_broker_url=f"unix://{TOOL_BROKER_SOCKET_SANDBOX_PATH}",
            tool_socket_host_path=sock_path,
            tool_broker_secret="thePerSessionSecret",
            session_id="sess_01TEST",
        )
        secret_file = tmp_path / "sess_01TEST.secret"
        assert secret_file.exists()
        assert secret_file.read_text() == "thePerSessionSecret"
        # 0o600 (not 0o644): the bind-mounted secret file is read by root
        # inside the container; no need to grant group/world read on host.
        # Set via ``os.open`` mode bits so create + perms are one syscall
        # (no umask-derived window where the file is briefly more open).
        assert (secret_file.stat().st_mode & 0o777) == 0o600

    def test_bind_mounts_secret_file_when_uds_set(self, tmp_path: Path) -> None:
        sock_path = tmp_path / "broker.sock"
        plan = _call_assemble(
            tool_broker_url=f"unix://{TOOL_BROKER_SOCKET_SANDBOX_PATH}",
            tool_socket_host_path=sock_path,
            session_id="sess_01TEST",
        )
        secret_mounts = [
            m for m in plan.spec.extra_mounts if m.sandbox_path == TOOL_BROKER_SECRET_SANDBOX_PATH
        ]
        assert len(secret_mounts) == 1
        m = secret_mounts[0]
        assert m.host_path == tmp_path / "sess_01TEST.secret"
        assert m.read_only is True

    def test_no_secret_file_when_uds_unset(self, tmp_path: Path) -> None:
        plan = _call_assemble(
            tool_broker_url="http://aios-worker:54321",
            tool_socket_host_path=None,
            session_id="sess_01TEST",
        )
        # No secret file should be written anywhere (tmp_path is pristine).
        assert list(tmp_path.glob("*.secret")) == []
        secret_mounts = [
            m for m in plan.spec.extra_mounts if m.sandbox_path == TOOL_BROKER_SECRET_SANDBOX_PATH
        ]
        assert secret_mounts == []
