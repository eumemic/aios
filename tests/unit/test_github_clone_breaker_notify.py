"""``rotate_token`` fires the github-clone-breaker clear NOTIFY (#1720).

A token rotation runs in the API process, but the live
``GithubCloneBreaker`` state for that resource lives in the worker's
in-memory breaker. This test pins that ``rotate_token`` emits a
``pg_notify`` on ``aios_github_clone_breaker_clear`` with the resource id
after the UPDATE commits, so a fixed credential re-probes on the next
provision instead of serving out a stale cooldown.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.db import queries
from aios.db.listen import GITHUB_CLONE_BREAKER_CLEAR_CHANNEL
from aios.models.github_repositories import GithubRepositoryResourceEcho
from aios.services import github_repositories as github_repo_service

pytestmark = pytest.mark.asyncio


def _fake_pool_with_conn() -> Any:
    conn = MagicMock()
    txn_cm = MagicMock()
    txn_cm.__aenter__ = AsyncMock(return_value=None)
    txn_cm.__aexit__ = AsyncMock(return_value=None)
    conn.transaction = MagicMock(return_value=txn_cm)

    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=None)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=acquire_cm)
    pool.execute = AsyncMock()
    return pool


def _echo(resource_id: str = "ghr_1") -> GithubRepositoryResourceEcho:
    now = datetime.now(UTC)
    return GithubRepositoryResourceEcho(
        id=resource_id,
        url="https://github.com/acme/foo.git",
        mount_path="/workspace/foo",
        created_at=now,
        updated_at=now,
    )


async def test_rotate_token_fires_breaker_clear_notify() -> None:
    pool = _fake_pool_with_conn()
    crypto_box = MagicMock()
    crypto_box.derive_account_subkey = MagicMock(return_value=MagicMock(encrypt=MagicMock()))

    with patch.object(
        queries, "update_session_github_repo_blob", AsyncMock(return_value=_echo("ghr_1"))
    ):
        result = await github_repo_service.rotate_token(
            pool,
            crypto_box,
            account_id="acc_1",
            session_id="sess_1",
            resource_id="ghr_1",
            new_token="new-token",
        )

    assert result.id == "ghr_1"
    pool.execute.assert_awaited_once()
    args = pool.execute.await_args.args
    assert args[0] == "SELECT pg_notify($1, $2)"
    assert args[1] == GITHUB_CLONE_BREAKER_CLEAR_CHANNEL
    assert args[2] == "ghr_1"


async def test_rotate_token_survives_failed_breaker_clear_notify() -> None:
    """Item 5b (#1720 seat-gate): the rotation UPDATE already committed, so a
    failed advisory NOTIFY must NOT 500 — rotate_token swallows it and returns
    the rotated echo. The breaker's own cooldown/half-open re-probe still
    recovers the fixed credential."""
    pool = _fake_pool_with_conn()
    pool.execute = AsyncMock(side_effect=RuntimeError("pg_notify connection reset"))
    crypto_box = MagicMock()
    crypto_box.derive_account_subkey = MagicMock(return_value=MagicMock(encrypt=MagicMock()))

    with patch.object(
        queries, "update_session_github_repo_blob", AsyncMock(return_value=_echo("ghr_1"))
    ):
        result = await github_repo_service.rotate_token(
            pool,
            crypto_box,
            account_id="acc_1",
            session_id="sess_1",
            resource_id="ghr_1",
            new_token="new-token",
        )

    # The rotation succeeded despite the NOTIFY failure.
    assert result.id == "ghr_1"
    pool.execute.assert_awaited_once()
