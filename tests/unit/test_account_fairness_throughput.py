"""A fairness bounce (no step ran) is not wake throughput (#2593 F2, dead-man #2541)."""

from __future__ import annotations

from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.harness import runtime
from aios.harness.account_fairness import AccountStepGate
from aios.harness.throughput import WAKE_COMPLETIONS
from tests.unit.conftest import fake_pool_yielding_conn


async def test_bounced_wake_is_not_counted_as_completed_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aios.harness.tasks import wake_session

    gate = AccountStepGate(1, stale_after_seconds=30)
    monkeypatch.setattr(runtime, "_account_step_gate", gate)
    assert gate.try_admit("acc_A", "ses_hung")  # A's only slot held by a hung step
    body = AsyncMock()
    monkeypatch.setattr(runtime, "pool", fake_pool_yielding_conn(MagicMock()))
    before = WAKE_COMPLETIONS.total
    with (
        mock.patch(
            "aios.services.sessions.load_live_session_account_id",
            new=AsyncMock(return_value="acc_A"),
        ),
        mock.patch("aios.harness.loop._run_session_step_body", new=body),
        mock.patch("aios.harness.loop.defer_wake", new=AsyncMock()),
    ):
        for _ in range(5):
            await wake_session.func("ses_surplus", cause="message")
    body.assert_not_awaited()  # no step ran: pure bounces
    assert WAKE_COMPLETIONS.total == before, (
        f"{WAKE_COMPLETIONS.total - before} bounces counted as throughput"
    )
