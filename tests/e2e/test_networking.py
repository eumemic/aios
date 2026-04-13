"""E2E tests for networking enforcement.

Requires Docker: provisions real containers and verifies that iptables
rules actually block/allow outbound traffic.
"""

from __future__ import annotations

from aios.models.environments import EnvironmentConfig, LimitedNetworking, UnrestrictedNetworking
from tests.conftest import needs_docker
from tests.e2e.harness import Harness, assistant, bash


@needs_docker
class TestNetworkingEnforcement:
    """Verify that iptables lockdown actually blocks/allows traffic."""

    async def test_limited_blocks_unlisted_host(self, docker_harness: Harness) -> None:
        """A limited environment should block curl to a host NOT in allowed_hosts."""
        docker_harness.script_model(
            [
                assistant(
                    tool_calls=[bash("curl -s --connect-timeout 5 http://httpbin.org/get")],
                ),
                assistant("Done."),
            ]
        )
        session = await docker_harness.start(
            "test",
            tools=["bash"],
            environment_config=EnvironmentConfig(
                networking=LimitedNetworking(type="limited", allowed_hosts=["example.com"]),
            ),
        )
        await docker_harness.run_until_idle(session.id)

        events = await docker_harness.events(session.id)
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
        docker_harness.script_model(
            [
                assistant(
                    tool_calls=[bash("curl -s --connect-timeout 10 https://example.com")],
                ),
                assistant("Done."),
            ]
        )
        session = await docker_harness.start(
            "test",
            tools=["bash"],
            environment_config=EnvironmentConfig(
                networking=LimitedNetworking(type="limited", allowed_hosts=["example.com"]),
            ),
        )
        await docker_harness.run_until_idle(session.id)

        events = await docker_harness.events(session.id)
        tool_result = next(
            e for e in events if e.kind == "message" and e.data.get("role") == "tool"
        )
        content = tool_result.data.get("content", "")
        # example.com returns a simple HTML page with "Example Domain"
        assert "Example Domain" in content

    async def test_unrestricted_allows_all(self, docker_harness: Harness) -> None:
        """An unrestricted environment should allow curl to any host."""
        docker_harness.script_model(
            [
                assistant(
                    tool_calls=[bash("curl -s --connect-timeout 10 https://example.com")],
                ),
                assistant("Done."),
            ]
        )
        session = await docker_harness.start(
            "test",
            tools=["bash"],
            environment_config=EnvironmentConfig(
                networking=UnrestrictedNetworking(),
            ),
        )
        await docker_harness.run_until_idle(session.id)

        events = await docker_harness.events(session.id)
        tool_result = next(
            e for e in events if e.kind == "message" and e.data.get("role") == "tool"
        )
        content = tool_result.data.get("content", "")
        assert "Example Domain" in content
