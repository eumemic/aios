"""Drift pin: ``RESERVED_SANDBOX_ENV_KEYS`` must equal the set of env keys the
harness injects into every sandbox.

``models/vaults.py`` rejects an ``environment_variable`` credential whose
``secret_name`` collides with a harness-injected key (a collision would either
hijack a load-bearing variable like ``PATH`` or be silently shadowed by the
merge order). That blocklist is hardcoded in the model layer because importing
``sandbox.setup`` there would cycle via ``aios.config``. This test builds a
real provisioning plan with no user-supplied env and asserts the blocklist is
exactly the harness-injected key set — so adding a key to ``merged_env`` (or to
``WORKSPACE_RUNTIME_ENV``) fails here until the blocklist follows.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from aios.models.vaults import RESERVED_SANDBOX_ENV_KEYS
from aios.sandbox.spec import _assemble_plan


def test_reserved_keys_match_injected_sandbox_env() -> None:
    with (
        patch("aios.sandbox.volumes.ensure_session_attachments_dir", return_value=Path("/tmp/a")),
        patch("aios.sandbox.volumes.ensure_session_uploads_dir", return_value=Path("/tmp/u")),
    ):
        plan = _assemble_plan(
            session_id="sess_01TEST",
            instance_id="inst_TEST",
            image="aios-sandbox:test",
            workspace_path=Path("/tmp/w"),
            env_config=None,
            session_env={},
            memory_echoes=[],
            github_echoes=[],
            git_proxy=None,
            tool_broker_url="http://aios-worker:54321",
            tool_broker_secret="secret123",
            tool_socket_host_path=None,
        )
    # With no env_config / session_env, every key in the merged env is
    # harness-injected; the reserved blocklist must cover exactly that set.
    assert set(plan.spec.environment) == RESERVED_SANDBOX_ENV_KEYS
