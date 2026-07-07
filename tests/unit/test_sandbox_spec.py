"""Unit coverage for sandbox spec assembly cleanup paths.

``_materialize_github_clones`` starts a :class:`GitProxy` at the top of
the function (uvicorn server + httpx client + bound TCP port +
in-memory token map). It then iterates the session's github echoes
calling ``ensure_session_working_tree`` for each. The per-repo
``except`` catches only :class:`GithubCloneError` — any other exception
(``OSError`` from a full disk during ``mkdir``/``rmtree``;
``asyncio.CancelledError`` on step interrupt; asyncpg failure during
the failure-event append) escapes without ``proxy.stop()``, leaking
the proxy for the worker's lifetime.

The outer ``try/except BaseException`` in ``build_spec_from_session``
wraps ``_assemble_plan`` (``spec.py:307-337``), not
``_materialize_github_clones`` — it cannot catch a proxy already
orphaned upstream of the plan-assembly call.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.models.github_repositories import GithubRepositoryResourceEcho
from aios.sandbox.spec import _materialize_github_clones


def _make_echo(repo_id: str = "ghr_test") -> GithubRepositoryResourceEcho:
    now = datetime.now(tz=UTC)
    return GithubRepositoryResourceEcho(
        id=repo_id,
        url="https://github.com/acme/foo.git",
        mount_path="/workspace/foo",
        created_at=now,
        updated_at=now,
    )


def _make_pool_with_conn() -> MagicMock:
    """Build a pool whose ``acquire()`` yields a stub connection."""
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=MagicMock())
    acquire_cm.__aexit__ = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=acquire_cm)
    return pool


async def test_session_clone_timeout_does_not_abort_sibling_clones() -> None:
    """A per-session ``git clone --reference`` timeout (now bounded by
    ``Settings.github_clone_session_timeout_seconds`` — issue #697)
    surfaces as a ``GithubCloneError``. The per-repo ``except
    GithubCloneError`` branch must drop the failed echo and continue
    cloning siblings, logging one ``github_clone_failed`` lifecycle
    event. Without this branch, a single timed-out clone would skip
    every later attached repo on the session.
    """
    mock_proxy = MagicMock()
    mock_proxy.start = AsyncMock()
    mock_proxy.stop = AsyncMock()
    mock_proxy.proxy_url = MagicMock(return_value="http://stub/proxy")

    from aios.sandbox.github_clone import GithubCloneError

    failed_echo = _make_echo("ghr_failed")
    ok_echo = _make_echo("ghr_ok")

    append_event_mock = AsyncMock()

    with (
        patch("aios.sandbox.spec.GitProxy", return_value=mock_proxy),
        patch(
            "aios.sandbox.spec.queries.list_session_github_repo_echoes",
            AsyncMock(return_value=[failed_echo, ok_echo]),
        ),
        patch(
            "aios.services.github_repositories.get_session_token",
            AsyncMock(return_value="ghp_TEST"),
        ),
        patch(
            "aios.sandbox.spec.ensure_cache_clone",
            AsyncMock(return_value=Path("/tmp/cache")),
        ),
        patch(
            "aios.sandbox.spec.ensure_session_working_tree",
            AsyncMock(
                side_effect=[
                    GithubCloneError("git clone --reference timed out after 30s"),
                    None,
                ]
            ),
        ),
        patch(
            "aios.sandbox.spec.runtime.require_pool",
            return_value=_make_pool_with_conn(),
        ),
        patch(
            "aios.sandbox.spec.runtime.require_crypto_box",
            return_value=MagicMock(),
        ),
        patch(
            "aios.sandbox.spec.sessions_service.append_event",
            append_event_mock,
        ),
    ):
        materialized, attempted, returned_proxy = await _materialize_github_clones(
            "sess_x", account_id="acct_x"
        )

    assert [e.id for e in materialized] == [ok_echo.id]
    # The failed clone is dropped from ``materialized`` (no bind mount) but
    # STAYS in ``attempted`` — that's the set the drift key is stamped from,
    # so a permanently-failing clone can't churn provision↔release (#1725).
    assert {e.id for e in attempted} == {failed_echo.id, ok_echo.id}
    assert returned_proxy is mock_proxy
    append_event_mock.assert_awaited_once()
    call_args = append_event_mock.await_args
    assert call_args is not None
    payload = call_args.args[3] if len(call_args.args) >= 4 else call_args.kwargs["payload"]
    assert payload["event"] == "github_clone_failed"
    assert payload["resource_id"] == failed_echo.id
    # Sibling clone still ran and succeeded; proxy stays alive for the caller.
    mock_proxy.stop.assert_not_called()


async def test_materialize_github_clones_stops_proxy_on_non_clone_error() -> None:
    """If ``ensure_session_working_tree`` raises a non-``GithubCloneError``
    (``OSError`` from a full disk during mkdir/rmtree;
    ``CancelledError`` on step interrupt), the :class:`GitProxy` started
    at the top of ``_materialize_github_clones`` must be stopped before
    the exception propagates. Otherwise the uvicorn server, httpx
    client, bound TCP port, and in-memory token map leak for the
    worker's lifetime — and every subsequent provision for this session
    would orphan another one.
    """
    mock_proxy = MagicMock()
    mock_proxy.start = AsyncMock()
    mock_proxy.stop = AsyncMock()
    mock_proxy.proxy_url = MagicMock(return_value="http://stub/proxy")

    with (
        patch("aios.sandbox.spec.GitProxy", return_value=mock_proxy),
        patch(
            "aios.sandbox.spec.queries.list_session_github_repo_echoes",
            AsyncMock(return_value=[_make_echo()]),
        ),
        patch(
            "aios.services.github_repositories.get_session_token",
            AsyncMock(return_value="ghp_TEST"),
        ),
        patch(
            "aios.sandbox.spec.ensure_cache_clone",
            AsyncMock(return_value=Path("/tmp/cache")),
        ),
        patch(
            "aios.sandbox.spec.ensure_session_working_tree",
            AsyncMock(side_effect=OSError("disk full")),
        ),
        patch(
            "aios.sandbox.spec.runtime.require_pool",
            return_value=_make_pool_with_conn(),
        ),
        patch(
            "aios.sandbox.spec.runtime.require_crypto_box",
            return_value=MagicMock(),
        ),
        pytest.raises(OSError),
    ):
        await _materialize_github_clones("sess_x", account_id="acct_x")

    mock_proxy.start.assert_awaited_once()
    mock_proxy.stop.assert_awaited_once()
