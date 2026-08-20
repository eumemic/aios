"""Production persistence-boundary test for the fleet egress audit."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.db.pool import create_pool
from aios.harness import fleet_egress_audit as audit
from aios.models.environments import UnrestrictedNetworking
from aios.sandbox.backends.base import Mount, SandboxSpec
from aios.sandbox.registry import SandboxRegistry
from aios.sandbox.setup import EgressProvisionResult, HostSkip
from aios.sandbox.spec import ProvisioningPlan
from tests.helpers.sandbox import FakeBackend
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.docker


def _plan(session_id: str) -> ProvisioningPlan:
    return ProvisioningPlan(
        spec=SandboxSpec(
            session_id=session_id,
            instance_id=f"inst_{session_id}",
            workspace=Mount(host_path=Path("/tmp/w"), sandbox_path="/workspace"),
            extra_mounts=(),
            environment={},
            labels={},
            network_policy=UnrestrictedNetworking(),
            host_gateway_alias=None,
            image="aios-sandbox:test",
        ),
        env_config=None,
        memory_echoes=[],
        github_echoes=[],
        git_proxy=None,
        env_var_credentials=(),
    )


async def _provision_with_outcome(
    pool: object,
    session_id: str,
    account_id: str,
    outcome: EgressProvisionResult | BaseException,
) -> None:
    """Run the production lifecycle producer while faking only its sandbox I/O."""
    registry = SandboxRegistry(backend=FakeBackend())
    apply = (
        AsyncMock(side_effect=outcome)
        if isinstance(outcome, BaseException)
        else AsyncMock(return_value=outcome)
    )
    with (
        patch(
            "aios.sandbox.registry.build_spec_from_session",
            AsyncMock(return_value=_plan(session_id)),
        ),
        patch("aios.sandbox.registry.install_egress_ca", AsyncMock()),
        patch("aios.sandbox.registry.install_packages", AsyncMock()),
        patch.object(registry, "_apply_egress_rules", apply),
    ):
        if isinstance(outcome, BaseException):
            with pytest.raises(type(outcome), match=str(outcome)):
                await registry.get_or_provision(session_id, pool=pool, account_id=account_id)
        else:
            await registry.get_or_provision(session_id, pool=pool, account_id=account_id)


async def test_production_egress_events_persist_and_auditor_alerts_each_adverse_outcome(
    migrated_db_url: str,
    _reset_db_state: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross the real writer -> events table -> fleet-reader boundary."""
    pool = await create_pool(migrated_db_url, min_size=1, max_size=4)
    account_id = "acc_fleet_egress"
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO accounts (id, display_name) VALUES ($1, $2)",
                account_id,
                "fleet egress test",
            )

        sessions = []
        for prefix in ("healthy", "skipped", "failed"):
            _, _, session = await seed_agent_env_session(
                pool, account_id=account_id, prefix=f"fleet-{prefix}"
            )
            sessions.append(session)
        healthy, skipped, failed = sessions

        await _provision_with_outcome(pool, healthy.id, account_id, EgressProvisionResult())
        await _provision_with_outcome(
            pool,
            skipped.id,
            account_id,
            EgressProvisionResult(
                hosts_installed=("api.example.com",),
                hosts_skipped=(HostSkip(host="missing.example.com", reason="no IPv4 address"),),
            ),
        )
        await _provision_with_outcome(
            pool,
            failed.id,
            account_id,
            RuntimeError("sidecar unavailable"),
        )

        warning = MagicMock()
        monkeypatch.setattr(audit.log, "warning", warning)
        result = await audit.run_fleet_egress_audit(pool)

        assert result.events_examined == 3
        assert result.healthy_events_observed == 1
        assert [(finding.session_id, finding.event) for finding in result.findings] == [
            (skipped.id, "egress_provisioned"),
            (failed.id, "egress_provision_failed"),
        ]
        assert warning.call_count == 2
    finally:
        await pool.close()
