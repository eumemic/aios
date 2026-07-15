from unittest.mock import AsyncMock

import pytest

from aios.db.queries import memory_stores
from aios.models.github_repositories import GithubRepositoryResourceEcho


@pytest.mark.asyncio
async def test_get_session_github_repo_delegates_to_with_blob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    echo = AsyncMock(spec=GithubRepositoryResourceEcho)
    get_with_blob = AsyncMock(return_value=(echo, AsyncMock()))
    monkeypatch.setattr(memory_stores, "get_session_github_repo_with_blob", get_with_blob)
    conn = AsyncMock()

    result = await memory_stores.get_session_github_repo(
        conn, "session-id", "resource-id", account_id="account-id"
    )

    assert result is echo
    get_with_blob.assert_awaited_once_with(
        conn, "session-id", "resource-id", account_id="account-id"
    )
    conn.fetchrow.assert_not_awaited()
