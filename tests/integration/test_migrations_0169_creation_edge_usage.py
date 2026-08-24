"""Migration 0169 backfills only durable creation evidence.

``archive_when_idle`` is a lifetime choice, not ownership provenance. A public
API client can create a self-archiving root and another session can invoke it
later. The resulting first ``request_opened`` event must not transfer the
target's historical or future subtree spend to that caller.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import asyncpg
import psycopg
import pytest

from tests.conftest import _docker_available, needs_docker
from tests.integration.test_migrations import _alembic_url, _run_alembic

_SEED_SQL = r"""
INSERT INTO accounts (id, parent_account_id, can_mint_children, display_name)
VALUES ('acc_0169', NULL, TRUE, '0169 migration');
INSERT INTO environments (id, name, account_id)
VALUES ('env_0169', 'env-0169', 'acc_0169');
INSERT INTO agents (id, name, model, account_id)
VALUES ('agent_0169', 'agent-0169', 'test/model', 'acc_0169');

INSERT INTO sessions (
    id, agent_id, environment_id, workspace_volume_path, account_id,
    archive_when_idle, last_event_seq, created_by_type, created_by_ref
)
VALUES
    ('ses_caller_0169', 'agent_0169', 'env_0169', '/tmp/caller-0169', 'acc_0169',
     FALSE, 0, 'api_actor', 'key_caller_0169'),
    -- Public API creation: lifetime is ephemeral, ownership is still root.
    ('ses_api_target_0169', 'agent_0169', 'env_0169', '/tmp/api-target-0169', 'acc_0169',
     TRUE, 1, 'api_actor', 'key_target_0169'),
    -- Explicit resource provenance is creation-specific and safe to recover.
    ('ses_provenance_target_0169', 'agent_0169', 'env_0169',
     '/tmp/provenance-target-0169', 'acc_0169', TRUE, 1,
     'session_actor', 'ses_caller_0169');

INSERT INTO events (id, session_id, seq, kind, data, account_id)
VALUES
    ('evt_api_later_0169', 'ses_api_target_0169', 1, 'lifecycle',
     '{"event":"request_opened","request_id":"req_api_later_0169",'
     '"caller":{"kind":"session","id":"ses_caller_0169","awaited":true}}'::jsonb,
     'acc_0169'),
    ('evt_provenance_0169', 'ses_provenance_target_0169', 1, 'lifecycle',
     '{"event":"request_opened","request_id":"req_provenance_0169",'
     '"caller":{"kind":"session","id":"ses_caller_0169","awaited":true}}'::jsonb,
     'acc_0169');
"""


@pytest.fixture
def postgres() -> Iterator[object]:
    if not _docker_available():
        pytest.skip("Docker not available")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


async def _execute(db_url: str, sql: str) -> None:
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(sql)
    finally:
        await conn.close()


async def _creator(db_url: str, session_id: str) -> str | None:
    conn = await asyncpg.connect(db_url)
    try:
        return cast(
            "str | None",
            await conn.fetchval(
                "SELECT creator_session_id FROM sessions WHERE id = $1", session_id
            ),
        )
    finally:
        await conn.close()


async def _seed_workflow_spend(db_url: str, *, spent_microusd: int, run_cost_microusd: int) -> None:
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(
            "INSERT INTO accounts "
            "(id, parent_account_id, can_mint_children, display_name, spent_microusd) "
            "VALUES ('acc_spend_0169', NULL, TRUE, '0169 spend', $1)",
            spent_microusd,
        )
        await conn.execute(
            "INSERT INTO environments (id, name, account_id) "
            "VALUES ('env_spend_0169', 'env-spend-0169', 'acc_spend_0169')"
        )
        await conn.execute(
            "INSERT INTO wf_runs "
            "(id, workflow_id, account_id, environment_id, script, script_sha, "
            " host_semantics_epoch, call_llm_cost_microusd) "
            "VALUES ('run_spend_0169', NULL, 'acc_spend_0169', 'env_spend_0169', "
            " 'async def main(i): return i', 'sha-spend-0169', 0, $1)",
            run_cost_microusd,
        )
    finally:
        await conn.close()


async def _account_spend(db_url: str) -> int:
    conn = await asyncpg.connect(db_url)
    try:
        return int(
            await conn.fetchval("SELECT spent_microusd FROM accounts WHERE id = 'acc_spend_0169'")
        )
    finally:
        await conn.close()


async def _seed_workflow_spend_watermark(db_url: str, accounted_microusd: int) -> None:
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(
            "INSERT INTO workflow_spend_accounting_watermarks "
            "(account_id, accounted_run_cost_microusd) VALUES ($1, $2)",
            "acc_spend_0169",
            accounted_microusd,
        )
    finally:
        await conn.close()


async def _workflow_spend_watermark(db_url: str) -> tuple[int, int, int]:
    conn = await asyncpg.connect(db_url)
    try:
        row = await conn.fetchrow(
            "SELECT accounted_run_cost_microusd, last_observed_run_cost_microusd, "
            "last_applied_delta_microusd FROM workflow_spend_accounting_watermarks "
            "WHERE account_id = 'acc_spend_0169'"
        )
        assert row is not None
        return (
            int(row["accounted_run_cost_microusd"]),
            int(row["last_observed_run_cost_microusd"]),
            int(row["last_applied_delta_microusd"]),
        )
    finally:
        await conn.close()


@needs_docker
@pytest.mark.integration
def test_backfill_does_not_infer_creation_from_archive_when_idle(postgres: object) -> None:
    db_url = _alembic_url(postgres)

    up = _run_alembic(["upgrade", "0166"], db_url)
    assert up.returncode == 0, f"upgrade to 0166 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_execute(db_url, _SEED_SQL))

    up = _run_alembic(["upgrade", "0169"], db_url)
    assert up.returncode == 0, f"upgrade to 0169 failed:\n{up.stderr}\n{up.stdout}"

    # The invocation-only edge remains unowned even though the target opted
    # into self-archival. Only explicit creation provenance is backfilled.
    assert asyncio.run(_creator(db_url, "ses_api_target_0169")) is None
    assert asyncio.run(_creator(db_url, "ses_provenance_target_0169")) == "ses_caller_0169"


@needs_docker
@pytest.mark.integration
@pytest.mark.parametrize(
    ("spent_microusd", "accounted_microusd"),
    [(0, 0), (100, 100), (50, 50)],
)
def test_historical_workflow_spend_is_reconciled_exactly_once(
    postgres: object, *, spent_microusd: int, accounted_microusd: int
) -> None:
    db_url = _alembic_url(postgres)
    up = _run_alembic(["upgrade", "0166"], db_url)
    assert up.returncode == 0, f"upgrade to 0166 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(
        _seed_workflow_spend(
            db_url,
            spent_microusd=spent_microusd,
            run_cost_microusd=100,
        )
    )
    up = _run_alembic(["upgrade", "0168"], db_url)
    assert up.returncode == 0, f"upgrade to 0168 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_seed_workflow_spend_watermark(db_url, accounted_microusd))

    up = _run_alembic(["upgrade", "0169"], db_url)
    assert up.returncode == 0, f"upgrade to 0169 failed:\n{up.stderr}\n{up.stdout}"
    assert asyncio.run(_account_spend(db_url)) == 100
    assert asyncio.run(_workflow_spend_watermark(db_url)) == (
        100,
        100,
        100 - accounted_microusd,
    )

    # A legacy writer knows only the run meter. The database trigger installed
    # at cutover still projects its post-migration delta exactly once.
    asyncio.run(
        _execute(
            db_url,
            "UPDATE wf_runs SET call_llm_cost_microusd = "
            "call_llm_cost_microusd + 40 WHERE id = 'run_spend_0169'",
        )
    )
    assert asyncio.run(_account_spend(db_url)) == 140
    assert asyncio.run(_workflow_spend_watermark(db_url)) == (140, 140, 40)


@needs_docker
@pytest.mark.integration
def test_old_writer_commit_across_migration_snapshot_is_not_lost(postgres: object) -> None:
    db_url = _alembic_url(postgres)
    up = _run_alembic(["upgrade", "0166"], db_url)
    assert up.returncode == 0, f"upgrade to 0166 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_seed_workflow_spend(db_url, spent_microusd=0, run_cost_microusd=100))
    up = _run_alembic(["upgrade", "0168"], db_url)
    assert up.returncode == 0, f"upgrade to 0168 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_seed_workflow_spend_watermark(db_url, 0))

    # Hold an old-writer update open. The migration must wait for that writer,
    # then reconcile the committed 140 total rather than its earlier 100 view.
    with psycopg.connect(db_url) as writer, ThreadPoolExecutor(max_workers=1) as executor:
        writer.execute(
            "UPDATE wf_runs SET call_llm_cost_microusd = "
            "call_llm_cost_microusd + 40 WHERE id = 'run_spend_0169'"
        )
        future = executor.submit(_run_alembic, ["upgrade", "0169"], db_url)
        assert not future.done()
        writer.commit()
        up = future.result(timeout=30)

    assert up.returncode == 0, f"upgrade to 0169 failed:\n{up.stderr}\n{up.stdout}"
    assert asyncio.run(_account_spend(db_url)) == 140
    assert asyncio.run(_workflow_spend_watermark(db_url)) == (140, 140, 140)


@needs_docker
@pytest.mark.integration
def test_missing_workflow_spend_watermark_fails_closed(postgres: object) -> None:
    db_url = _alembic_url(postgres)
    up = _run_alembic(["upgrade", "0166"], db_url)
    assert up.returncode == 0, f"upgrade to 0166 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_seed_workflow_spend(db_url, spent_microusd=50, run_cost_microusd=100))
    up = _run_alembic(["upgrade", "0168"], db_url)
    assert up.returncode == 0, f"upgrade to 0168 failed:\n{up.stderr}\n{up.stdout}"

    up = _run_alembic(["upgrade", "0169"], db_url)
    assert up.returncode != 0
    assert "missing workflow spend accounting watermark" in up.stderr
    assert asyncio.run(_account_spend(db_url)) == 50


@needs_docker
@pytest.mark.integration
def test_workflow_spend_watermark_above_retained_cost_fails_closed(postgres: object) -> None:
    db_url = _alembic_url(postgres)
    up = _run_alembic(["upgrade", "0166"], db_url)
    assert up.returncode == 0, f"upgrade to 0166 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_seed_workflow_spend(db_url, spent_microusd=101, run_cost_microusd=100))
    up = _run_alembic(["upgrade", "0168"], db_url)
    assert up.returncode == 0, f"upgrade to 0168 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_seed_workflow_spend_watermark(db_url, 101))

    up = _run_alembic(["upgrade", "0169"], db_url)
    assert up.returncode != 0
    assert "workflow spend accounting watermark exceeds retained run cost" in up.stderr
    assert asyncio.run(_account_spend(db_url)) == 101


@needs_docker
@pytest.mark.integration
def test_coincidental_aggregate_equality_does_not_fake_provenance(postgres: object) -> None:
    db_url = _alembic_url(postgres)
    up = _run_alembic(["upgrade", "0166"], db_url)
    assert up.returncode == 0, f"upgrade to 0166 failed:\n{up.stderr}\n{up.stdout}"
    # The existing 100 is unrelated spend. Its equality with the retained run
    # meter proves nothing about whether that run was charged.
    asyncio.run(_seed_workflow_spend(db_url, spent_microusd=100, run_cost_microusd=100))
    up = _run_alembic(["upgrade", "0168"], db_url)
    assert up.returncode == 0, f"upgrade to 0168 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_seed_workflow_spend_watermark(db_url, 0))

    up = _run_alembic(["upgrade", "0169"], db_url)
    assert up.returncode == 0, f"upgrade to 0169 failed:\n{up.stderr}\n{up.stdout}"
    assert asyncio.run(_account_spend(db_url)) == 200
    assert asyncio.run(_workflow_spend_watermark(db_url)) == (100, 100, 100)


@needs_docker
@pytest.mark.integration
def test_historical_tokens_are_incomplete_and_downgrade_is_lossless(
    postgres: object,
) -> None:
    db_url = _alembic_url(postgres)
    up = _run_alembic(["upgrade", "0166"], db_url)
    assert up.returncode == 0, f"upgrade to 0166 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_seed_workflow_spend(db_url, spent_microusd=0, run_cost_microusd=100))
    asyncio.run(
        _execute(
            db_url,
            "INSERT INTO wf_run_events (id, run_id, seq, type, call_key, payload) VALUES "
            "('evt_started_0169', 'run_spend_0169', 1, 'call_started', 'call_0169', "
            ' \'{"capability":"call_llm"}\'::jsonb), '
            "('evt_result_0169', 'run_spend_0169', 2, 'call_result', 'call_0169', "
            ' \'{"result":{"usage":{"input_tokens":"17"}}}\'::jsonb)',
        )
    )
    up = _run_alembic(["upgrade", "0168"], db_url)
    assert up.returncode == 0, f"upgrade to 0168 failed:\n{up.stderr}\n{up.stdout}"
    asyncio.run(_seed_workflow_spend_watermark(db_url, 0))
    up = _run_alembic(["upgrade", "0169"], db_url)
    assert up.returncode == 0, f"upgrade to 0169 failed:\n{up.stderr}\n{up.stdout}"

    async def _usage_state() -> tuple[tuple[int, int, bool], int, object]:
        conn = await asyncpg.connect(db_url)
        try:
            row = await conn.fetchrow(
                "SELECT call_llm_input_tokens, call_llm_output_tokens, "
                "call_llm_tokens_complete "
                "FROM wf_runs WHERE id = 'run_spend_0169'"
            )
            assert row is not None
            ledger_cost = await conn.fetchval(
                "SELECT COALESCE(SUM(cost_microusd), 0) "
                "FROM inference_usage_ledger WHERE run_id = 'run_spend_0169'"
            )
            coverage = await conn.fetchval(
                "SELECT usage_ledger_started_at FROM accounts WHERE id = 'acc_spend_0169'"
            )
            return (
                (
                    int(row["call_llm_input_tokens"]),
                    int(row["call_llm_output_tokens"]),
                    bool(row["call_llm_tokens_complete"]),
                ),
                int(ledger_cost),
                coverage,
            )
        finally:
            await conn.close()

    initial = asyncio.run(_usage_state())
    assert initial[0] == (0, 0, False)
    asyncio.run(
        _execute(
            db_url,
            "UPDATE wf_runs SET call_llm_input_tokens = 17, "
            "call_llm_output_tokens = 9 WHERE id = 'run_spend_0169'; "
            "INSERT INTO inference_usage_ledger "
            "(account_id, run_id, input_tokens, output_tokens, cost_microusd) "
            "VALUES ('acc_spend_0169', 'run_spend_0169', 17, 9, 23)",
        )
    )
    before_downgrade = asyncio.run(_usage_state())
    assert before_downgrade[:2] == ((17, 9, False), 23)

    down = _run_alembic(["downgrade", "0166"], db_url)
    assert down.returncode == 0, f"downgrade to 0166 failed:\n{down.stderr}\n{down.stdout}"

    async def _archived_state() -> tuple[tuple[int, int, bool], int, int, int]:
        conn = await asyncpg.connect(db_url)
        try:
            row = await conn.fetchrow(
                "SELECT call_llm_input_tokens, call_llm_output_tokens, "
                "call_llm_tokens_complete FROM _aios_0169_wf_run_usage_archive "
                "WHERE run_id = 'run_spend_0169'"
            )
            assert row is not None
            ledger_cost = await conn.fetchval(
                "SELECT SUM(cost_microusd) "
                "FROM _aios_0169_inference_usage_ledger_archive "
                "WHERE run_id = 'run_spend_0169'"
            )
            spend = await conn.fetchval(
                "SELECT spent_microusd FROM accounts WHERE id = 'acc_spend_0169'"
            )
            accounted = await conn.fetchval(
                "SELECT accounted_run_cost_microusd "
                "FROM _aios_0168_workflow_spend_watermarks_archive "
                "WHERE account_id = 'acc_spend_0169'"
            )
            return (
                (
                    int(row["call_llm_input_tokens"]),
                    int(row["call_llm_output_tokens"]),
                    bool(row["call_llm_tokens_complete"]),
                ),
                int(ledger_cost),
                int(spend),
                int(accounted),
            )
        finally:
            await conn.close()

    assert asyncio.run(_archived_state()) == ((17, 9, False), 23, 100, 100)

    up = _run_alembic(["upgrade", "0169"], db_url)
    assert up.returncode == 0, f"re-upgrade to 0169 failed:\n{up.stderr}\n{up.stdout}"
    restored = asyncio.run(_usage_state())
    assert restored[:2] == ((17, 9, False), 23)
    assert restored[2] == before_downgrade[2]
    assert asyncio.run(_account_spend(db_url)) == 100
    assert asyncio.run(_workflow_spend_watermark(db_url)) == (100, 100, 0)
