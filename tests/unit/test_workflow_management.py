"""Unit tests for the agent-acting workflow builtins (slice 3).

These stub the worker pool + services so they need no live Postgres. They cover the
two identity-load-bearing invariants that don't depend on the DB:

* **F1** — a trusted id smuggled into the arguments is rejected by the tool schema
  (``additionalProperties: false``) before the handler ever runs.
* **S2** — a 4xx service refusal (or a pydantic semantic error the JSON schema can't
  encode) becomes a model-visible ``ToolBail``, never a raised exception (which would
  evict the sandbox); a genuine 5xx still propagates.

Plus the ``await_run`` wiring (it sources ``db_url`` from settings) and the
script-trimming of returned dicts. Full attenuation/depth behavior is covered by the
integration tests.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

import aios.tools  # noqa: F401 — registers the builtins
from aios.errors import CryptoDecryptError, ForbiddenError
from aios.models.workflows import WfRunWaitResponse, Workflow
from aios.tools import workflow_management as wm
from aios.tools.invoke import ToolBail, invoke_builtin

_DT = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _stub_runtime(monkeypatch: Any) -> None:
    """Make ``require_pool`` + ``load_session_account_id`` cheap (no DB)."""
    monkeypatch.setattr("aios.harness.runtime.require_pool", lambda: object())
    monkeypatch.setattr(
        "aios.services.sessions.load_session_account_id", AsyncMock(return_value="acc_x")
    )


def _workflow(**over: Any) -> Workflow:
    base: dict[str, Any] = dict(
        id="wf_1",
        account_id="acc_x",
        name="w",
        version=1,
        script="SECRET",
        created_at=_DT,
        updated_at=_DT,
    )
    base.update(over)
    return Workflow(**base)


class TestSchemaRejectsInjectedTrustedIds:
    """F1: the trusted ids are never schema fields, so injection is rejected up front."""

    async def test_create_workflow_creator_session_id_rejected(self) -> None:
        with pytest.raises(ToolBail):
            await invoke_builtin(
                "ses_1",
                "create_workflow",
                {"name": "w", "script": "s", "creator_session_id": "ses_victim"},
            )

    async def test_update_workflow_actor_session_id_rejected(self) -> None:
        with pytest.raises(ToolBail):
            await invoke_builtin(
                "ses_1",
                "update_workflow",
                {"workflow_id": "wf_1", "version": 1, "actor_session_id": "ses_victim"},
            )

    async def test_create_run_environment_id_rejected(self) -> None:
        # F2: environment_id is deliberately NOT a field — the run inherits the caller's.
        with pytest.raises(ToolBail):
            await invoke_builtin(
                "ses_1", "create_run", {"workflow_id": "wf_1", "environment_id": "env_other"}
            )

    async def test_create_run_parent_run_id_rejected(self) -> None:
        with pytest.raises(ToolBail):
            await invoke_builtin(
                "ses_1", "create_run", {"workflow_id": "wf_1", "parent_run_id": "wfr_x"}
            )


class TestErrorConversion:
    async def test_pydantic_semantic_error_becomes_toolbail(self) -> None:
        # An mcp server named after a builtin passes the JSON schema but fails the
        # McpServerSpec custom validator — must surface as ToolBail, not a raw exception.
        with pytest.raises(ToolBail, match="invalid arguments"):
            await wm.create_workflow_handler(
                "ses_1",
                {"name": "w", "script": "s", "mcp_servers": [{"name": "bash", "url": "https://x"}]},
            )

    async def test_forbidden_becomes_toolbail(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "aios.services.workflows.create_workflow",
            AsyncMock(
                side_effect=ForbiddenError("nope", detail={"ungranted": {"tools": ["bash"]}})
            ),
        )
        with pytest.raises(ToolBail, match="nope"):
            await wm.create_workflow_handler("ses_1", {"name": "w", "script": "s"})

    async def test_internal_5xx_propagates(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "aios.services.workflows.create_workflow",
            AsyncMock(side_effect=CryptoDecryptError("boom")),
        )
        with pytest.raises(CryptoDecryptError):
            await wm.create_workflow_handler("ses_1", {"name": "w", "script": "s"})


class TestReturnShape:
    async def test_create_workflow_excludes_script(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "aios.services.workflows.create_workflow",
            AsyncMock(return_value=_workflow(version=1)),
        )
        out = await wm.create_workflow_handler("ses_1", {"name": "w", "script": "SECRET"})
        assert out["name"] == "w" and out["version"] == 1
        assert "script" not in out  # heavy snapshot field trimmed

    async def test_await_run_passes_settings_db_url(self, monkeypatch: Any) -> None:
        mock_await = AsyncMock(
            return_value=WfRunWaitResponse(run_status="completed", done=True, output=5)
        )
        monkeypatch.setattr("aios.services.workflows.await_run", mock_await)
        monkeypatch.setattr(
            "aios.tools.workflow_management.get_settings",
            lambda: SimpleNamespace(db_url="postgres://test-db"),
        )
        out = await wm.await_run_handler("ses_1", {"run_id": "wfr_1", "timeout_seconds": 7})
        assert out["done"] is True and out["output"] == 5
        # db_url is sourced from settings (positional arg 2), not model input.
        pos = mock_await.call_args.args
        assert pos[1] == "postgres://test-db"
        assert mock_await.call_args.kwargs["timeout_seconds"] == 7
