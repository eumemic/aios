from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aios.api.deps import get_account_id, get_pool
from aios.api.routers.workflows import runs_router
from aios.errors import ValidationError, install_exception_handlers
from aios.models.workflows import WfRunCreate
from aios.tools.invoke_session import _CallWorkflowArgs
from aios.workflows import service
from aios.workflows.determinism import content_hash
from aios.workflows.wf_script_host import agent


def test_http_run_defaults_to_fresh_workspace() -> None:
    run = WfRunCreate(workflow_id="wf_1", environment_id="env_1")
    assert run.workspace == "fresh"


def test_session_workflow_call_defaults_shared_and_allows_fresh() -> None:
    assert _CallWorkflowArgs(workflow_id="wf_1").workspace == "shared"
    assert _CallWorkflowArgs(workflow_id="wf_1", workspace="fresh").workspace == "fresh"


def test_agent_workspace_mode_enters_call_identity() -> None:
    shared = agent("work")
    fresh = agent("work", workspace="fresh")
    assert shared._spec["workspace"] == "shared"
    assert fresh._spec["workspace"] == "fresh"
    assert content_hash(shared._capability_id, shared._spec) != content_hash(
        fresh._capability_id, fresh._spec
    )


class _ExplodingPool:
    def acquire(self) -> None:
        raise AssertionError("shared-without-launcher must fail before database access")


def test_http_shared_workspace_rejected_422_before_service_or_db() -> None:
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(runs_router)
    app.dependency_overrides[get_pool] = lambda: _ExplodingPool()
    app.dependency_overrides[get_account_id] = lambda: "acc_test"

    response = TestClient(app).post(
        "/v1/runs",
        json={"workflow_id": "wf_1", "environment_id": "env_1", "workspace": "shared"},
    )

    assert response.status_code == 422
    assert "requires a launcher session" in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_service_shared_without_launcher_rejected_before_database() -> None:
    with pytest.raises(ValidationError, match="requires a launcher session"):
        await service.create_run(
            _ExplodingPool(),
            account_id="acc_test",
            workflow_id="wf_1",
            environment_id="env_1",
            workspace="shared",
        )
