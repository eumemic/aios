"""Unit tests for the agent-acting workflow builtins (slice 3).

These stub the worker pool + services so they need no live Postgres. They cover the
two identity-load-bearing invariants that don't depend on the DB:

* **F1** — a trusted id smuggled into the arguments is rejected by the tool schema
  (``additionalProperties: false``) before the handler ever runs.
* **Error handling** — a bad-argument pydantic error the JSON schema can't encode (e.g.
  an ``McpServerSpec`` name rule) bails locally as a ``ToolBail``; a service ``AiosError``
  propagates unconverted, leaving the dispatch layer (``_classify_tool_error``) to decide
  eviction vs. a clean model-visible result (see ``test_tool_dispatch``).

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
from aios.errors import CryptoDecryptError, ForbiddenError, NotFoundError
from aios.models.workflows import (
    WfRun,
    WfRunEvent,
    WfRunWaitResponse,
    Workflow,
    WorkflowCreate,
    WorkflowUpdate,
)
from aios.tools import workflow_management as wm
from aios.tools.invoke import ToolBail, invoke_builtin
from aios.tools.registry import registry

_DT = datetime(2026, 1, 1, tzinfo=UTC)
_SCRIPT_CONTRACT_TOKENS = ("async def main", "agent", "tool", "gate", "parallel", "pipeline", "log")


def _assert_script_contract_present(text: str | None) -> None:
    assert text
    for token in _SCRIPT_CONTRACT_TOKENS:
        assert token in text


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


def _run(**over: Any) -> WfRun:
    base: dict[str, Any] = dict(
        id="wfr_1",
        workflow_id="wf_1",
        account_id="acc_x",
        environment_id="env_x",
        script="SECRET",
        script_sha="sha",
        host_semantics_epoch=1,
        status="running",
        last_event_seq=0,
        budget_usd=None,
        created_at=_DT,
        updated_at=_DT,
    )
    base.update(over)
    return WfRun(**base)


def _event(**over: Any) -> WfRunEvent:
    base: dict[str, Any] = dict(
        id="wfe_1",
        run_id="wfr_1",
        seq=1,
        type="annotation",
        call_key="k0",
        payload={"kind": "log", "text": "hello"},
        created_at=_DT,
    )
    base.update(over)
    return WfRunEvent(**base)


class TestWorkflowScriptContractDiscovery:
    def test_create_schema_script_description_contains_contract(self) -> None:
        description = WorkflowCreate.model_json_schema()["properties"]["script"].get("description")
        _assert_script_contract_present(description)

    def test_update_schema_script_description_contains_contract(self) -> None:
        description = WorkflowUpdate.model_json_schema()["properties"]["script"].get("description")
        _assert_script_contract_present(description)

    def test_registered_create_workflow_description_contains_contract(self) -> None:
        _assert_script_contract_present(registry.get("create_workflow").description)

    def test_registered_update_workflow_description_contains_contract(self) -> None:
        _assert_script_contract_present(registry.get("update_workflow").description)


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

    async def test_cancel_run_canceller_session_id_rejected(self) -> None:
        with pytest.raises(ToolBail):
            await invoke_builtin(
                "ses_1", "cancel_run", {"run_id": "wfr_1", "canceller_session_id": "ses_victim"}
            )

    async def test_archive_workflow_account_id_rejected(self) -> None:
        with pytest.raises(ToolBail):
            await invoke_builtin(
                "ses_1", "archive_workflow", {"workflow_id": "wf_1", "account_id": "acc_victim"}
            )

    async def test_unarchive_workflow_account_id_rejected(self) -> None:
        with pytest.raises(ToolBail):
            await invoke_builtin(
                "ses_1",
                "unarchive_workflow",
                {"workflow_id": "wf_1", "account_id": "acc_victim"},
            )

    async def test_get_run_account_id_rejected(self) -> None:
        with pytest.raises(ToolBail):
            await invoke_builtin(
                "ses_1", "get_run", {"run_id": "wfr_1", "account_id": "acc_victim"}
            )

    async def test_list_runs_launcher_session_id_rejected(self) -> None:
        with pytest.raises(ToolBail):
            await invoke_builtin("ses_1", "list_runs", {"launcher_session_id": "ses_victim"})

    async def test_list_runs_account_id_rejected(self) -> None:
        with pytest.raises(ToolBail):
            await invoke_builtin("ses_1", "list_runs", {"account_id": "acc_victim"})

    async def test_list_run_events_account_id_rejected(self) -> None:
        with pytest.raises(ToolBail):
            await invoke_builtin(
                "ses_1", "list_run_events", {"run_id": "wfr_1", "account_id": "acc_victim"}
            )

    async def test_get_workflow_account_id_rejected(self) -> None:
        with pytest.raises(ToolBail):
            await invoke_builtin(
                "ses_1", "get_workflow", {"workflow_id": "wf_1", "account_id": "acc_victim"}
            )

    async def test_list_runs_rejects_invalid_status(self) -> None:
        # status is a WfRunStatus enum, not free-form str: 'in_progress' is not a valid
        # run status, so it bails at validation (rather than silently matching zero rows).
        with pytest.raises(ToolBail):
            await invoke_builtin("ses_1", "list_runs", {"status": "in_progress"})

    async def test_list_runs_account_wide_as_id_rejected(self) -> None:
        # The widen control is a bool; smuggling the sibling launcher_session_id key
        # alongside account_wide=True trips extra="forbid" before the handler runs.
        with pytest.raises(ToolBail):
            await invoke_builtin(
                "ses_1",
                "list_runs",
                {"account_wide": True, "launcher_session_id": "ses_victim"},
            )

    async def test_resume_gate_resumer_session_id_rejected(self) -> None:
        with pytest.raises(ToolBail):
            await invoke_builtin(
                "ses_1",
                "resume_gate",
                {
                    "run_id": "wfr_1",
                    "gate_nonce": "nonce",
                    "result": "ok",
                    "resumer_session_id": "ses_victim",
                },
            )


class TestErrorPropagation:
    async def test_pydantic_semantic_error_becomes_toolbail(self) -> None:
        # An mcp server named after a builtin passes the JSON schema but fails the
        # McpServerSpec custom validator — _parse surfaces it as ToolBail (the only
        # error the handler converts locally; service errors propagate to the dispatch
        # layer — see test_tool_dispatch).
        with pytest.raises(ToolBail, match="invalid arguments"):
            await wm.create_workflow_handler(
                "ses_1",
                {"name": "w", "script": "s", "mcp_servers": [{"name": "bash", "url": "https://x"}]},
            )

    async def test_service_aios_error_propagates_unconverted(self, monkeypatch: Any) -> None:
        # The handler no longer converts service errors itself — it lets the AiosError
        # propagate so the dispatch layer (_classify_tool_error) decides eviction vs a
        # clean model-visible result. A 4xx ForbiddenError reaches the caller as-is here.
        monkeypatch.setattr(
            "aios.services.workflows.create_workflow",
            AsyncMock(
                side_effect=ForbiddenError("nope", detail={"ungranted": {"tools": ["bash"]}})
            ),
        )
        with pytest.raises(ForbiddenError, match="nope"):
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

    async def test_archive_workflow_calls_service_and_returns_archived_at(
        self, monkeypatch: Any
    ) -> None:
        archived = datetime(2026, 1, 2, tzinfo=UTC)
        mock_archive = AsyncMock(return_value=_workflow(archived_at=archived))
        monkeypatch.setattr("aios.services.workflows.archive_workflow", mock_archive)
        out = await wm.archive_workflow_handler("ses_1", {"workflow_id": "wf_1"})
        assert out["id"] == "wf_1"
        assert out["archived_at"] == archived.isoformat().replace("+00:00", "Z")
        assert "script" not in out
        assert mock_archive.call_args.args[1] == "wf_1"
        assert mock_archive.call_args.kwargs["account_id"] == "acc_x"

    async def test_unarchive_workflow_calls_service_and_returns_live_row(
        self, monkeypatch: Any
    ) -> None:
        mock_unarchive = AsyncMock(return_value=_workflow(archived_at=None))
        monkeypatch.setattr("aios.services.workflows.unarchive_workflow", mock_unarchive)
        out = await wm.unarchive_workflow_handler("ses_1", {"workflow_id": "wf_1"})
        assert out["id"] == "wf_1"
        assert out["archived_at"] is None
        assert mock_unarchive.call_args.args[1] == "wf_1"
        assert mock_unarchive.call_args.kwargs["account_id"] == "acc_x"

    async def test_create_run_threads_budget_usd(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "aios.services.sessions.get_session_basic",
            AsyncMock(return_value=SimpleNamespace(environment_id="env_x", parent_run_id=None)),
        )
        mock_create = AsyncMock(return_value=_run(budget_usd=1.5))
        monkeypatch.setattr("aios.services.workflows.create_run", mock_create)
        out = await wm.create_run_handler("ses_1", {"workflow_id": "wf_1", "budget_usd": 1.5})
        assert out["budget_usd"] == 1.5
        assert mock_create.call_args.kwargs["budget_usd"] == 1.5

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

    async def test_resume_gate_passes_launcher_session_id_and_trims_run(
        self, monkeypatch: Any
    ) -> None:
        mock_resume = AsyncMock(return_value=_run(status="suspended"))
        monkeypatch.setattr("aios.services.workflows.resume_gate_by_nonce", mock_resume)
        out = await wm.resume_gate_handler(
            "ses_1", {"run_id": "wfr_1", "gate_nonce": "nonce", "result": {"ok": True}}
        )
        assert out["id"] == "wfr_1" and out["status"] == "suspended"
        assert "script" not in out
        assert mock_resume.call_args.kwargs == {
            "run_id": "wfr_1",
            "account_id": "acc_x",
            "gate_nonce": "nonce",
            "result": {"ok": True},
            "resumer_session_id": "ses_1",
        }


class TestReadHandlers:
    """The five read-only builtins: full vs. lean return shapes + launcher scoping."""

    async def test_get_workflow_returns_full_script_and_version(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "aios.services.workflows.get_workflow",
            AsyncMock(return_value=_workflow(version=3, script="SECRET")),
        )
        out = await wm.get_workflow_handler("ses_1", {"workflow_id": "wf_1"})
        assert out["script"] == "SECRET"
        assert out["version"] == 3
        assert out["id"] == "wf_1"

    async def test_list_workflows_excludes_script_and_heavy_fields(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "aios.services.workflows.list_workflows",
            AsyncMock(return_value=[_workflow(name="w", version=2, script="SECRET")]),
        )
        out = await wm.list_workflows_handler("ses_1", {})
        row = out["workflows"][0]
        assert row["name"] == "w"
        assert row["version"] == 2
        for heavy in (
            "script",
            "input_schema",
            "output_schema",
            "tools",
            "mcp_servers",
            "http_servers",
        ):
            assert heavy not in row
        assert "description" in row
        assert "created_at" in row

    async def test_list_runs_default_filters_by_launcher(self, monkeypatch: Any) -> None:
        mock_list = AsyncMock(return_value=[_run()])
        monkeypatch.setattr("aios.services.workflows.list_runs", mock_list)
        await wm.list_runs_handler("ses_1", {})
        assert mock_list.call_args.kwargs["launcher_session_id"] == "ses_1"

    async def test_list_runs_account_wide_drops_launcher_filter(self, monkeypatch: Any) -> None:
        mock_list = AsyncMock(return_value=[_run()])
        monkeypatch.setattr("aios.services.workflows.list_runs", mock_list)
        await wm.list_runs_handler("ses_1", {"account_wide": True})
        assert mock_list.call_args.kwargs["launcher_session_id"] is None

    async def test_list_runs_strips_heavy_run_fields(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "aios.services.workflows.list_runs",
            AsyncMock(return_value=[_run(script="SECRET")]),
        )
        out = await wm.list_runs_handler("ses_1", {})
        row = out["runs"][0]
        for heavy in ("script", "script_sha", "tools", "mcp_servers", "http_servers"):
            assert heavy not in row
        assert row["status"] == "running"

    async def test_list_runs_passes_through_workflow_id_and_status(self, monkeypatch: Any) -> None:
        mock_list = AsyncMock(return_value=[_run()])
        monkeypatch.setattr("aios.services.workflows.list_runs", mock_list)
        await wm.list_runs_handler("ses_1", {"workflow_id": "wf_9", "status": "completed"})
        assert mock_list.call_args.kwargs["workflow_id"] == "wf_9"
        assert mock_list.call_args.kwargs["status"] == "completed"

    async def test_list_runs_passes_through_parent_run_id(self, monkeypatch: Any) -> None:
        mock_list = AsyncMock(return_value=[_run()])
        monkeypatch.setattr("aios.services.workflows.list_runs", mock_list)
        await wm.list_runs_handler("ses_1", {"parent_run_id": "wfr_parent"})
        assert mock_list.call_args.kwargs["parent_run_id"] == "wfr_parent"

    async def test_get_run_returns_full_including_script(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "aios.services.workflows.get_run",
            AsyncMock(return_value=_run(script="SECRET")),
        )
        out = await wm.get_run_handler("ses_1", {"run_id": "wfr_1"})
        assert out["script"] == "SECRET"
        assert out["script_sha"] == "sha"
        assert out["status"] == "running"

    async def test_list_run_events_pages_and_preserves_annotation_order(
        self, monkeypatch: Any
    ) -> None:
        # Pre-flight get_run must resolve (a real run exists) so the journal read proceeds.
        monkeypatch.setattr("aios.services.workflows.get_run", AsyncMock(return_value=_run()))
        mock_ev = AsyncMock(
            return_value=[
                _event(seq=2, payload={"kind": "phase", "text": "a"}),
                _event(seq=3, payload={"kind": "log", "text": "b"}),
            ]
        )
        monkeypatch.setattr("aios.services.workflows.list_run_events", mock_ev)
        out = await wm.list_run_events_handler(
            "ses_1", {"run_id": "wfr_1", "after_seq": 1, "limit": 50}
        )
        assert [e["seq"] for e in out["events"]] == [2, 3]
        assert out["events"][0]["payload"]["kind"] == "phase"
        assert mock_ev.call_args.kwargs["after_seq"] == 1
        assert mock_ev.call_args.kwargs["limit"] == 50

    async def test_list_run_events_raises_for_unknown_run(self, monkeypatch: Any) -> None:
        # A nonexistent / cross-account run_id must surface as a real NotFoundError the
        # model can act on — not an empty event list indistinguishable from "no new events".
        monkeypatch.setattr(
            "aios.services.workflows.get_run",
            AsyncMock(side_effect=NotFoundError("workflow run not found")),
        )
        with pytest.raises(NotFoundError):
            await wm.list_run_events_handler("ses_1", {"run_id": "wfr_missing"})
