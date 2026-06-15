"""Unit tests for the #1124 trusted-recursion DOWN-counter rule.

Pure, DB-free coverage of ``next_trusted_depth`` — the single decrement-and-refuse
rule shared by every trusted-invocation hop (run→run, run→session, session→session,
api→session) — plus the shared budget constant and the ``WfRun.depth`` carrier. The
end-to-end edge/refusal behaviour is covered against real rows in
``tests/integration/test_wf_run_vaults.py`` (run→run chain + edgeless-root seed),
``tests/integration/test_trigger_event_fires.py`` (cascade), and
``tests/integration/test_request_opened_edge.py`` (session→session A↔B cycle).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

import aios.tools  # noqa: F401 — load the full module graph first (avoids the

# service-imported-standalone-first circular import that predates #1124)
from aios.models.workflows import WfRun
from aios.workflows.service import (
    WORKFLOW_RUN_MAX_DEPTH,
    WorkflowRunDepthExceededError,
    next_trusted_depth,
)


def test_budget_constant_is_ten() -> None:
    """The single shared budget seed for edgeless roots."""
    assert WORKFLOW_RUN_MAX_DEPTH == 10


def test_decrement_by_one_at_each_hop() -> None:
    """``child = parent - 1`` — the trusted scalar decrements once per hop."""
    assert next_trusted_depth(WORKFLOW_RUN_MAX_DEPTH) == WORKFLOW_RUN_MAX_DEPTH - 1
    assert next_trusted_depth(5) == 4
    assert next_trusted_depth(1) == 0  # a depth-0 child is a valid terminal leaf


def test_full_chain_bottoms_out_at_zero() -> None:
    """Seeding at the budget and decrementing yields exactly budget+1 valid depths
    (budget..0); the next hop refuses. This is the cycle bound BY CONSTRUCTION."""
    depths = [WORKFLOW_RUN_MAX_DEPTH]
    while True:
        try:
            depths.append(next_trusted_depth(depths[-1]))
        except WorkflowRunDepthExceededError:
            break
    assert depths == list(range(WORKFLOW_RUN_MAX_DEPTH, -1, -1))  # 10, 9, ..., 1, 0
    assert depths[-1] == 0
    assert len(depths) == WORKFLOW_RUN_MAX_DEPTH + 1


def test_refuse_before_write_at_floor() -> None:
    """A parent with no budget left (depth 0) refuses BEFORE producing a child — the
    child would be below the floor. The refusal is the shared 409 depth error."""
    with pytest.raises(WorkflowRunDepthExceededError) as exc:
        next_trusted_depth(0)
    assert exc.value.status_code == 409
    assert exc.value.error_type == "workflow_run_depth_exceeded"


def test_wf_run_carries_depth_scalar() -> None:
    """Depth rides the run's own row (the trusted scalar), not a forgeable counter."""
    run = WfRun(
        id="run_x",
        workflow_id="wf_x",
        account_id="acc_x",
        environment_id="env_x",
        depth=7,
        script="async def main(i):\n    return i\n",
        script_sha="x" * 64,
        host_semantics_epoch=1,
        status="pending",
        last_event_seq=0,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert run.depth == 7
