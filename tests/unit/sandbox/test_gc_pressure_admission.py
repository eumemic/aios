"""GC pressure is applied only to provisioning that can worsen that pressure."""

import importlib
from unittest.mock import Mock

import pytest

from aios.harness.worker import _consume_snapshot_pressure
from aios.sandbox.registry import GcPressureResult, SandboxRegistry
from tests.helpers.sandbox import FakeBackend


def test_account_pressure_is_scoped_and_alarm_clears(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = SandboxRegistry(backend=FakeBackend())
    alarm = Mock()
    monkeypatch.setattr("aios.sandbox.registry.log.error", alarm)
    registry.set_provisioning_pressure(GcPressureResult(pressured_accounts=frozenset({"acct_hot"})))

    with pytest.raises(RuntimeError, match="snapshot capacity pressure"):
        registry._admit_capacity_provision("sess_hot", account_id="acct_hot")
    registry._admit_capacity_provision("sess_other", account_id="acct_other")
    registry._admit_capacity_provision("wfr_run", account_id=None, durable=False)
    alarm.assert_called_once()

    registry.set_provisioning_pressure(GcPressureResult())
    registry._admit_capacity_provision("sess_hot", account_id="acct_hot")


def test_host_pool_pressure_remains_global() -> None:
    registry = SandboxRegistry(backend=FakeBackend())
    registry.set_provisioning_pressure(GcPressureResult(pool_used_bytes=11, pool_budget_bytes=10))

    with pytest.raises(RuntimeError, match="snapshot capacity pressure"):
        registry._admit_capacity_provision("sess_any", account_id="acct_other")
    with pytest.raises(RuntimeError, match="snapshot capacity pressure"):
        registry._admit_capacity_provision("wfr_run", account_id=None, durable=False)


def test_worker_gc_callback_drives_and_clears_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = SandboxRegistry(backend=FakeBackend())
    worker_alarm = Mock()
    worker_module = importlib.import_module("aios.harness.worker")
    logger = Mock()
    logger.error = worker_alarm
    monkeypatch.setattr(worker_module, "get_logger", Mock(return_value=logger))

    _consume_snapshot_pressure(
        registry, GcPressureResult(pressured_accounts=frozenset({"acct_hot"}))
    )
    with pytest.raises(RuntimeError):
        registry._admit_capacity_provision("sess_hot", account_id="acct_hot")
    worker_alarm.assert_called_once_with(
        "worker.sandbox_capacity_pressure_alarm",
        pool_used_bytes=0,
        pool_budget_bytes=None,
        pressured_accounts=["acct_hot"],
    )

    _consume_snapshot_pressure(registry, GcPressureResult())
    registry._admit_capacity_provision("sess_hot", account_id="acct_hot")
