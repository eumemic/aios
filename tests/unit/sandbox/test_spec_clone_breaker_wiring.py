"""``_materialize_github_clones`` wiring to ``runtime.github_clone_breaker`` (#1720).

Separate from ``tests/unit/test_sandbox_spec.py`` (which pins the proxy
cleanup/sibling-continuation contract with no breaker present) — these pin
that repeated clone failures for the SAME resource id open the breaker,
skip the clone subprocess on subsequent attempts, and emit exactly one
``github_clone_degraded`` event on the DOWN transition.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from aios.models.github_repositories import GithubRepositoryResourceEcho
from aios.sandbox.github_clone import GithubCloneError
from aios.sandbox.github_clone_breaker import _BREAKER_FAILURE_THRESHOLD, GithubCloneBreaker
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
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=MagicMock())
    acquire_cm.__aexit__ = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=acquire_cm)
    return pool


async def _run_once(
    *, breaker: GithubCloneBreaker, echo: GithubRepositoryResourceEcho, append_event_mock: AsyncMock
) -> tuple[list[GithubRepositoryResourceEcho], list[GithubRepositoryResourceEcho]]:
    mock_proxy = MagicMock()
    mock_proxy.start = AsyncMock()
    mock_proxy.stop = AsyncMock()
    mock_proxy.proxy_url = MagicMock(return_value="http://stub/proxy")

    with (
        patch("aios.sandbox.spec.GitProxy", return_value=mock_proxy),
        patch(
            "aios.sandbox.spec.queries.list_session_github_repo_echoes",
            AsyncMock(return_value=[echo]),
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
            AsyncMock(side_effect=GithubCloneError("boom")),
        ),
        patch(
            "aios.sandbox.spec.runtime.require_pool",
            return_value=_make_pool_with_conn(),
        ),
        patch(
            "aios.sandbox.spec.runtime.require_crypto_box",
            return_value=MagicMock(),
        ),
        patch("aios.sandbox.spec.runtime.github_clone_breaker", breaker),
        patch(
            "aios.sandbox.spec.sessions_service.append_event",
            append_event_mock,
        ),
    ):
        materialized, attempted, _proxy = await _materialize_github_clones(
            "sess_x", account_id="acct_x"
        )
    return materialized, attempted


async def test_breaker_opens_after_threshold_and_emits_degraded_event() -> None:
    breaker = GithubCloneBreaker()
    echo = _make_echo("ghr_dead")
    append_event_mock = AsyncMock()

    for _ in range(_BREAKER_FAILURE_THRESHOLD):
        await _run_once(breaker=breaker, echo=echo, append_event_mock=append_event_mock)

    assert breaker.is_open("ghr_dead") is True
    events = [c.args[3] for c in append_event_mock.await_args_list]
    degraded_events = [e for e in events if e["event"] == "github_clone_degraded"]
    assert len(degraded_events) == 1
    assert degraded_events[0]["resource_id"] == "ghr_dead"


async def test_open_breaker_skips_clone_subprocess() -> None:
    breaker = GithubCloneBreaker()
    echo = _make_echo("ghr_dead2")
    append_event_mock = AsyncMock()

    for _ in range(_BREAKER_FAILURE_THRESHOLD):
        await _run_once(breaker=breaker, echo=echo, append_event_mock=append_event_mock)
    assert breaker.is_open("ghr_dead2") is True

    ensure_working_tree_mock = AsyncMock(side_effect=GithubCloneError("boom"))
    mock_proxy = MagicMock()
    mock_proxy.start = AsyncMock()
    mock_proxy.stop = AsyncMock()
    mock_proxy.proxy_url = MagicMock(return_value="http://stub/proxy")

    with (
        patch("aios.sandbox.spec.GitProxy", return_value=mock_proxy),
        patch(
            "aios.sandbox.spec.queries.list_session_github_repo_echoes",
            AsyncMock(return_value=[echo]),
        ),
        patch(
            "aios.services.github_repositories.get_session_token",
            AsyncMock(return_value="ghp_TEST"),
        ),
        patch(
            "aios.sandbox.spec.ensure_cache_clone",
            AsyncMock(return_value=Path("/tmp/cache")),
        ),
        patch("aios.sandbox.spec.ensure_session_working_tree", ensure_working_tree_mock),
        patch(
            "aios.sandbox.spec.runtime.require_pool",
            return_value=_make_pool_with_conn(),
        ),
        patch(
            "aios.sandbox.spec.runtime.require_crypto_box",
            return_value=MagicMock(),
        ),
        patch("aios.sandbox.spec.runtime.github_clone_breaker", breaker),
        patch(
            "aios.sandbox.spec.sessions_service.append_event",
            AsyncMock(),
        ),
    ):
        materialized, attempted, _proxy = await _materialize_github_clones(
            "sess_x", account_id="acct_x"
        )

    ensure_working_tree_mock.assert_not_awaited()
    assert materialized == []
    assert {e.id for e in attempted} == {"ghr_dead2"}


async def test_open_breaker_does_not_append_clone_failed_event_per_turn() -> None:
    """Item 3 (#1720): while the breaker is OPEN, the per-turn skip must NOT
    append a ``github_clone_failed`` event (the 2,100-rows/day churn). The
    prelude line + the one-shot ``github_clone_degraded`` edge carry the signal.
    """
    breaker = GithubCloneBreaker()
    echo = _make_echo("ghr_open")

    # Drive it OPEN (records the degraded DOWN edge on the K-th failure).
    open_append = AsyncMock()
    for _ in range(_BREAKER_FAILURE_THRESHOLD):
        await _run_once(breaker=breaker, echo=echo, append_event_mock=open_append)
    assert breaker.is_open("ghr_open") is True

    # Now a further turn hits the skip path (is_open True) — fresh mock.
    skip_append = AsyncMock()
    mock_proxy = MagicMock()
    mock_proxy.start = AsyncMock()
    mock_proxy.stop = AsyncMock()
    mock_proxy.proxy_url = MagicMock(return_value="http://stub/proxy")
    with (
        patch("aios.sandbox.spec.GitProxy", return_value=mock_proxy),
        patch(
            "aios.sandbox.spec.queries.list_session_github_repo_echoes",
            AsyncMock(return_value=[echo]),
        ),
        patch(
            "aios.services.github_repositories.get_session_token",
            AsyncMock(return_value="ghp_TEST"),
        ),
        patch("aios.sandbox.spec.ensure_cache_clone", AsyncMock(return_value=Path("/tmp/cache"))),
        patch("aios.sandbox.spec.ensure_session_working_tree", AsyncMock()),
        patch("aios.sandbox.spec.runtime.require_pool", return_value=_make_pool_with_conn()),
        patch("aios.sandbox.spec.runtime.require_crypto_box", return_value=MagicMock()),
        patch("aios.sandbox.spec.runtime.github_clone_breaker", breaker),
        patch("aios.sandbox.spec.sessions_service.append_event", skip_append),
    ):
        await _materialize_github_clones("sess_x", account_id="acct_x")

    events = [c.args[3]["event"] for c in skip_append.await_args_list]
    assert "github_clone_failed" not in events


async def test_transient_failure_records_auth_false_and_error_on_breaker() -> None:
    """The classification (#1720 item 1) is threaded from the error into the
    breaker: a non-auth ``GithubCloneError`` opens with ``auth_failure=False``
    and carries the (redacted) message as ``last_error``."""
    breaker = GithubCloneBreaker()
    echo = _make_echo("ghr_transient")
    append = AsyncMock()
    for _ in range(_BREAKER_FAILURE_THRESHOLD):
        await _run_once(breaker=breaker, echo=echo, append_event_mock=append)

    degraded = breaker.degraded_repos()[0]
    assert degraded.auth_failure is False  # "boom" is not auth-shaped
    assert "boom" in degraded.last_error


async def test_breaker_recovers_on_successful_clone() -> None:
    breaker = GithubCloneBreaker()
    echo = _make_echo("ghr_recovers")
    append_event_mock = AsyncMock()

    mock_proxy = MagicMock()
    mock_proxy.start = AsyncMock()
    mock_proxy.stop = AsyncMock()
    mock_proxy.proxy_url = MagicMock(return_value="http://stub/proxy")

    with (
        patch("aios.sandbox.spec.GitProxy", return_value=mock_proxy),
        patch(
            "aios.sandbox.spec.queries.list_session_github_repo_echoes",
            AsyncMock(return_value=[echo]),
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
            AsyncMock(return_value=None),
        ),
        patch(
            "aios.sandbox.spec.runtime.require_pool",
            return_value=_make_pool_with_conn(),
        ),
        patch(
            "aios.sandbox.spec.runtime.require_crypto_box",
            return_value=MagicMock(),
        ),
        patch("aios.sandbox.spec.runtime.github_clone_breaker", breaker),
        patch(
            "aios.sandbox.spec.sessions_service.append_event",
            append_event_mock,
        ),
    ):
        materialized, _attempted, _proxy = await _materialize_github_clones(
            "sess_x", account_id="acct_x"
        )

    assert [e.id for e in materialized] == ["ghr_recovers"]
    assert breaker.is_open("ghr_recovers") is False
    append_event_mock.assert_not_awaited()
