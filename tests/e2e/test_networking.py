"""E2E tests for networking enforcement.

Requires Docker: provisions real containers and verifies that iptables
rules actually block/allow outbound traffic.
"""

from __future__ import annotations

from aios.models.environments import EnvironmentConfig, LimitedNetworking, UnrestrictedNetworking
from aios.services import agents as agents_service
from aios.services import environments as environments_service
from aios.services import sessions as sessions_service
from tests.conftest import needs_docker
from tests.e2e.harness import Harness, assistant, bash


async def _start_session_with_networking(
    harness: Harness,
    networking: LimitedNetworking | UnrestrictedNetworking,
) -> str:
    """Create an environment with the given networking config and return a session id."""
    from aios.ids import make_id
    from aios.models.agents import ToolSpec

    env = await environments_service.create_environment(
        harness._pool,
        name=f"net-test-{make_id('env')[-8:]}",
        config=EnvironmentConfig(networking=networking),
    )
    agent = await agents_service.create_agent(
        harness._pool,
        name=f"net-test-{make_id('agent')[-8:]}",
        model="fake/test",
        system="You are a test assistant.",
        tools=[ToolSpec(type="bash")],
        description=None,
        metadata={},
        window_min=50_000,
        window_max=150_000,
    )
    session = await sessions_service.create_session(
        harness._pool,
        agent_id=agent.id,
        environment_id=env.id,
        title="networking-e2e",
        metadata={},
    )
    await sessions_service.append_user_message(harness._pool, session.id, "test")
    return session.id


@needs_docker
class TestNetworkingEnforcement:
    """Verify that iptables lockdown actually blocks/allows traffic."""

    async def test_limited_blocks_unlisted_host(self, docker_harness: Harness) -> None:
        """A limited environment should block curl to a host NOT in allowed_hosts."""
        session_id = await _start_session_with_networking(
            docker_harness,
            LimitedNetworking(type="limited", allowed_hosts=["example.com"]),
        )
        # Script the model to curl a host that is NOT allowed.
        # --connect-timeout 5 so the test doesn't hang.
        docker_harness.script_model(
            [
                assistant(
                    tool_calls=[bash("curl -s --connect-timeout 5 http://httpbin.org/get")],
                ),
                assistant("Done."),
            ]
        )
        await docker_harness.run_until_idle(session_id)

        events = await docker_harness.events(session_id)
        # The curl should fail — connection refused or timed out.
        tool_result = next(
            e for e in events if e.kind == "message" and e.data.get("role") == "tool"
        )
        content = tool_result.data.get("content", "")
        # iptables DROP causes a timeout or connection error, not an HTTP response.
        assert (
            "httpbin" not in content.lower()
            or "timed out" in content.lower()
            or (tool_result.data.get("exit_code", 0) != 0)
        )

    async def test_limited_allows_listed_host(self, docker_harness: Harness) -> None:
        """A limited environment should allow curl to a host in allowed_hosts."""
        session_id = await _start_session_with_networking(
            docker_harness,
            LimitedNetworking(type="limited", allowed_hosts=["example.com"]),
        )
        docker_harness.script_model(
            [
                assistant(
                    tool_calls=[bash("curl -s --connect-timeout 10 https://example.com")],
                ),
                assistant("Done."),
            ]
        )
        await docker_harness.run_until_idle(session_id)

        events = await docker_harness.events(session_id)
        tool_result = next(
            e for e in events if e.kind == "message" and e.data.get("role") == "tool"
        )
        content = tool_result.data.get("content", "")
        # example.com returns a simple HTML page with "Example Domain"
        assert "Example Domain" in content

    async def test_unrestricted_allows_all(self, docker_harness: Harness) -> None:
        """An unrestricted environment should allow curl to any host."""
        session_id = await _start_session_with_networking(
            docker_harness,
            UnrestrictedNetworking(),
        )
        docker_harness.script_model(
            [
                assistant(
                    tool_calls=[bash("curl -s --connect-timeout 10 https://example.com")],
                ),
                assistant("Done."),
            ]
        )
        await docker_harness.run_until_idle(session_id)

        events = await docker_harness.events(session_id)
        tool_result = next(
            e for e in events if e.kind == "message" and e.data.get("role") == "tool"
        )
        content = tool_result.data.get("content", "")
        assert "Example Domain" in content
