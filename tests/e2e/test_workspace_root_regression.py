"""E2E regression for issue #2064 — workspace-root drift detection and
canonical account-scoped path acceptance.

The original issue: the standing CEO session has a canonical workspace path
``/srv/aios/workspaces/<account_id>/<session_id>`` that is correctly inside
the account subdirectory, but the worker's effective ``AIOS_WORKSPACE_ROOT``
diverged from the API's, causing ``validate_workspace_path`` to reject every
filesystem tool call with ``ForbiddenError``.

This module exercises two axes of the fix through the real Docker-backed
harness:

1. **Aligned root — canonical path accepted + real bash/read/write/scratch
   roundtrip.**  The ``docker_harness`` creates a session whose default
   ``workspace_volume_path`` is the canonical
   ``<workspace_root>/<account_id>/<session_id>`` shape (the same shape
   the production API returns).  A real Docker sandbox is provisioned
   via ``build_spec_from_session`` / the registry, and the test verifies
   bash execution, ``write`` tool, ``read`` tool, and scratch-file
   lifecycle all succeed end-to-end through the bind-mounted workspace.

2. **Divergent root — fails before readiness with expected diagnostic.**
   ``validate_workspace_root_against_sessions`` is called against a real
   Postgres with a session row whose ``workspace_volume_path`` was written
   under the canonical root, but the process's ``AIOS_WORKSPACE_ROOT`` is
   then changed to a different directory.  The validation must raise
   ``RuntimeError`` with the full diagnostic (service, workspace_root,
   account_root, raw_path, resolved_path, account_id, session_id) before
   the process could reach readiness.

Residual not covered here: the test does NOT exercise a genuinely
split-process (two separate containers with different volume mounts)
deployment.  That scenario requires production-shaped Docker Compose /
Kubernetes config that CI does not provision.  The invariant is covered
to the boundary: same Postgres, same ``validate_workspace_root_against_sessions``
call, same diagnostic shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aios.config import get_settings
from aios.harness import runtime
from aios.sandbox.workspace_root_startup import validate_workspace_root_against_sessions
from aios.tools.bash import bash_handler
from aios.tools.read import read_handler
from aios.tools.write import write_handler
from tests.conftest import needs_docker
from tests.e2e.harness import Harness

pytestmark = pytest.mark.docker


# ── Axis 1: aligned root, canonical path, real sandbox roundtrip ─────────


@needs_docker
class TestCanonicalPathAcceptedWithRealSandbox:
    """The exact issue shape: a canonical ``<workspace_root>/<account_id>/
    <session_id>`` path must be accepted by ``validate_workspace_path``,
    ``build_spec_from_session``, real registry provisioning, and the
    actual bash/read/write/scratch tool handlers inside a Docker sandbox.
    """

    async def test_bash_roundtrip_canonical_workspace(self, docker_harness: Harness) -> None:
        """bash tool writes and reads a scratch file through the real
        bind-mounted workspace at the canonical account-scoped path."""
        session = await docker_harness.start("scratch roundtrip", tools=["bash"])

        # Verify the sandbox handle's workspace_path has the canonical
        # account-scoped shape.  The Session model does not expose the
        # raw ``workspace_volume_path`` column, so read it from the
        # provisioned SandboxHandle which stores the resolved host path.
        sandbox = runtime.require_sandbox_registry()
        handle = await sandbox.get_or_provision(session.id, pool=docker_harness._pool)
        settings = get_settings()
        account_root = settings.workspace_root / "acc_test_stub"
        assert handle.workspace_path.is_relative_to(account_root), (
            f"handle.workspace_path {handle.workspace_path} is not under "
            f"the expected account root {account_root}"
        )

        # Write a scratch file via bash, read it back, then delete it
        result = await bash_handler(
            session.id,
            {
                "command": (
                    'echo "aios-2064-canary" > /workspace/.scratch-probe '
                    "&& cat /workspace/.scratch-probe "
                    "&& rm /workspace/.scratch-probe "
                    "&& echo PROBE_OK"
                )
            },
        )
        assert result["exit_code"] == 0, f"bash scratch roundtrip failed: {result}"
        assert "aios-2064-canary" in result["stdout"], result
        assert "PROBE_OK" in result["stdout"], result

    async def test_write_read_tool_roundtrip_canonical_workspace(
        self, docker_harness: Harness
    ) -> None:
        """The write and read tools succeed for a canonical account-scoped
        workspace path through real Docker provisioning."""
        session = await docker_harness.start(
            "write-read roundtrip", tools=["bash", "read", "write"]
        )

        # Write via the write tool (returns dict[str, Any])
        write_result = await write_handler(
            session.id,
            {"path": "/workspace/probe-2064.txt", "content": "workspace-root-regression-probe"},
        )
        assert "error" not in str(write_result).lower(), write_result

        # Read via the read tool (returns dict | ToolResult)
        read_result = await read_handler(session.id, {"path": "/workspace/probe-2064.txt"})
        # Normalise to string for content assertion
        if isinstance(read_result, dict):
            content = str(read_result.get("content", read_result))
        else:
            content = str(read_result.content)
        assert "workspace-root-regression-probe" in content, (
            f"read tool did not return written content: {content!r}"
        )

        # Verify the host-side bind-mount source reflects the write
        sandbox = runtime.require_sandbox_registry()
        handle = await sandbox.get_or_provision(session.id, pool=docker_harness._pool)
        host_file = handle.workspace_path / "probe-2064.txt"
        assert host_file.exists(), f"host-side file not found at {host_file}"
        assert host_file.read_text().strip() == "workspace-root-regression-probe"

    async def test_startup_validation_passes_aligned_root(self, docker_harness: Harness) -> None:
        """``validate_workspace_root_against_sessions`` passes when the
        session's canonical workspace path is under the process's
        configured ``AIOS_WORKSPACE_ROOT``.  Exercises the real DB path
        (not mocked)."""
        # Create a session so there's at least one live row
        _session = await docker_harness.start("validation probe", tools=["bash"])

        # Run the startup validation against the real pool — must pass
        await validate_workspace_root_against_sessions(docker_harness._pool, service="test")
        # If we get here without RuntimeError, the canonical path was accepted


# ── Axis 2: divergent root fails before readiness ────────────────────────


@needs_docker
class TestDivergentRootFailsBeforeReadiness:
    """When ``AIOS_WORKSPACE_ROOT`` diverges from the root under which
    session rows were created, ``validate_workspace_root_against_sessions``
    must raise ``RuntimeError`` with the full diagnostic BEFORE the process
    reaches readiness — i.e. this is a startup-gate, not a per-call check.
    """

    async def test_mismatched_root_raises_with_diagnostic(
        self,
        docker_harness: Harness,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A session created under the real workspace root is rejected
        when the process's configured root is changed to a different
        directory, with the full diagnostic in the error message."""
        # Create a session under the real (aligned) root
        _session = await docker_harness.start("divergence probe", tools=["bash"])
        original_root = get_settings().workspace_root

        # Now simulate the API/worker drift: change workspace_root to a
        # different directory (as would happen if the API and worker had
        # different AIOS_WORKSPACE_ROOT or different volume mounts)
        divergent_root = tmp_path / "divergent_workspaces"
        divergent_root.mkdir()
        monkeypatch.setattr(get_settings(), "workspace_root", divergent_root)

        try:
            with pytest.raises(RuntimeError) as exc_info:
                await validate_workspace_root_against_sessions(
                    docker_harness._pool, service="worker"
                )

            message = str(exc_info.value)
            # The diagnostic must contain all the fields the issue requires
            assert "workspace-root startup validation failed" in message, message
            assert "service='worker'" in message, message
            assert str(divergent_root) in message, (
                f"diagnostic should contain the divergent root {divergent_root!r}: {message}"
            )
            assert "account_id=" in message, message
            assert "session_id=" in message, message
            assert "raw_path=" in message, message
            assert "resolved_path=" in message, message
            assert "workspace_root=" in message, message
            assert "account_root=" in message, message
        finally:
            # Restore so harness teardown (which releases sandboxes) works
            monkeypatch.setattr(get_settings(), "workspace_root", original_root)
