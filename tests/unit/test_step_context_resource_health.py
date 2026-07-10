"""``compute_step_prelude`` wires breaker state into the resource-health block (#1720).

Unit-level: no Postgres. Stubs the pool (only ``get_open_obligations`` is
touched) and the ``ToolProvider`` seam, same pattern as
``test_step_context_provider_clamp.py``. The two cases nail the wiring:

* an attached repo whose ``GithubCloneBreaker`` is open renders
  ``repos: <mount_path> AUTH-FAILED since <since>`` in the system prompt;
* an agent-declared MCP server whose ``McpSessionPool`` breaker is open
  renders ``mcp: <name> DEGRADED``.

A healthy session (no open breakers) gets neither line — the system prompt
is unchanged from the pre-#1720 baseline.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import pytest

from aios.harness import runtime
from aios.harness.step_context import compute_step_prelude
from aios.mcp.pool import McpSessionPool
from aios.models.agents import AgentBinding, McpServerSpec, StepSurface, ToolSpec
from aios.models.github_repositories import GithubRepositoryResourceEcho
from aios.sandbox.github_clone_breaker import _BREAKER_FAILURE_THRESHOLD, GithubCloneBreaker

pytestmark = pytest.mark.asyncio

_ACCOUNT = "acc_1720"
_SESSION = "sess_1720"
_REPO_ID = "ghrepo_01HEALTH"
_REPO_URL = "https://github.com/acme/theme.git"
_MOUNT_PATH = "/workspace/theme"
_MCP_URL = "https://mcp.example.com/github"


def _agent(*, mcp_servers: list[McpServerSpec] | None = None) -> StepSurface:
    return StepSurface(
        model="gpt-test",
        system="you are a test agent",
        tools=[ToolSpec(type="bash")],
        skills=[],
        mcp_servers=mcp_servers or [],
        http_servers=[],
        litellm_extra={},
        window_min=1,
        window_max=10,
        binding=AgentBinding(agent_id="agt_1720", version=1),
        preempt_policy="wait",
    )


def _session() -> Any:
    return mock.Mock(id=_SESSION, parent_run_id=None)


def _repo_echo() -> GithubRepositoryResourceEcho:
    now = datetime.now(UTC)
    return GithubRepositoryResourceEcho(
        id=_REPO_ID,
        url=_REPO_URL,
        mount_path=_MOUNT_PATH,
        created_at=now,
        updated_at=now,
    )


class _StubConn:
    async def __aenter__(self) -> _StubConn:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _StubPool:
    def acquire(self) -> _StubConn:
        return _StubConn()


def _stub_tool_provider() -> Any:
    tp = mock.Mock()
    tp.list_tools_for_session = AsyncMock(return_value=[])
    return tp


async def _prelude_system(
    agent: StepSurface,
    *,
    github_repo_echoes: list[GithubRepositoryResourceEcho] | None = None,
) -> str:
    with mock.patch("aios.db.queries.get_open_obligations", new=AsyncMock(return_value=[])):
        prelude = await compute_step_prelude(
            _StubPool(),
            _SESSION,
            account_id=_ACCOUNT,
            session=_session(),
            agent=agent,
            channels=[],
            memory_store_echoes=[],
            github_repo_echoes=github_repo_echoes,
        )
    return prelude.system_prompt


@pytest.fixture(autouse=True)
def _stub_tool_provider_runtime(monkeypatch: Any) -> None:
    monkeypatch.setattr(runtime, "tool_provider", _stub_tool_provider())


async def test_healthy_session_has_no_resource_health_block() -> None:
    agent = _agent()
    system = await _prelude_system(agent, github_repo_echoes=[_repo_echo()])
    assert "Resource health" not in system


async def test_degraded_repo_renders_auth_failed_line(monkeypatch: Any) -> None:
    breaker = GithubCloneBreaker()
    for _ in range(_BREAKER_FAILURE_THRESHOLD):
        breaker.record_failure(
            _REPO_ID,
            _REPO_URL,
            _MOUNT_PATH,
            auth_failure=True,
            last_error="Authentication failed",
        )
    monkeypatch.setattr(runtime, "github_clone_breaker", breaker)

    system = await _prelude_system(_agent(), github_repo_echoes=[_repo_echo()])
    assert "━━━ Resource health ━━━" in system
    assert f"repos: {_MOUNT_PATH} AUTH-FAILED since" in system


async def test_degraded_repo_transient_renders_clone_failing_not_auth(monkeypatch: Any) -> None:
    """A transient (non-auth) breaker-open repo renders CLONE-FAILING with the
    real git cause, never a hardcoded AUTH-FAILED (#1720 seat-gate fix)."""
    breaker = GithubCloneBreaker()
    for _ in range(_BREAKER_FAILURE_THRESHOLD):
        breaker.record_failure(
            _REPO_ID,
            _REPO_URL,
            _MOUNT_PATH,
            auth_failure=False,
            last_error="git clone timed out after 30.0s",
        )
    monkeypatch.setattr(runtime, "github_clone_breaker", breaker)

    system = await _prelude_system(_agent(), github_repo_echoes=[_repo_echo()])
    assert "━━━ Resource health ━━━" in system
    assert f"repos: {_MOUNT_PATH} CLONE-FAILING since" in system
    assert "git clone timed out after 30.0s" in system
    assert "AUTH-FAILED" not in system


async def test_degraded_repo_not_attached_to_this_session_is_not_leaked(monkeypatch: Any) -> None:
    """A different session's degraded repo id doesn't leak into this one's prelude."""
    breaker = GithubCloneBreaker()
    for _ in range(_BREAKER_FAILURE_THRESHOLD):
        breaker.record_failure(
            "ghrepo_other", "https://github.com/acme/other.git", "/workspace/other"
        )
    monkeypatch.setattr(runtime, "github_clone_breaker", breaker)

    system = await _prelude_system(_agent(), github_repo_echoes=[_repo_echo()])
    assert "Resource health" not in system


def _mcp_key(url: str) -> tuple[str, str | None, str]:
    return (url, None, "")


async def test_degraded_mcp_server_renders_degraded_line(monkeypatch: Any) -> None:
    pool = McpSessionPool()
    key = _mcp_key(_MCP_URL)
    for _ in range(3):
        pool._record_connect_failure(key, _MCP_URL)
    monkeypatch.setattr(runtime, "mcp_session_pool", pool)

    agent = _agent(mcp_servers=[McpServerSpec(name="github", url=_MCP_URL)])
    system = await _prelude_system(agent)
    assert "━━━ Resource health ━━━" in system
    assert "mcp: github DEGRADED" in system


async def test_degraded_mcp_server_not_declared_by_agent_is_not_leaked(monkeypatch: Any) -> None:
    pool = McpSessionPool()
    key = _mcp_key(_MCP_URL)
    for _ in range(3):
        pool._record_connect_failure(key, _MCP_URL)
    monkeypatch.setattr(runtime, "mcp_session_pool", pool)

    agent = _agent(mcp_servers=[])
    system = await _prelude_system(agent)
    assert "Resource health" not in system
