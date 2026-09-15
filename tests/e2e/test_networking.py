"""E2E tests for networking enforcement.

Requires Docker: provisions real containers and verifies that iptables
rules actually block/allow outbound traffic.
"""

from __future__ import annotations

import pytest

from aios.models.environments import EnvironmentConfig, LimitedNetworking, UnrestrictedNetworking
from tests.conftest import needs_docker
from tests.e2e.harness import Harness, assistant, bash

pytestmark = pytest.mark.docker


# Every test here asserts on the EFFECT of the netns-sidecar iptables lockdown, which
# cannot be installed under gVisor/runsc (separate netstacks per container; the nat
# table has never been implemented -- gvisor#170, aios#2310). The lockdown fails closed,
# so the sandbox refuses to provision and no assertion here is meaningful.
#
# NOTE test_limited_blocks_unlisted_host in particular: under runsc it currently
# REPORTS PASS, and that pass is worthless. Its assertion accepts any non-zero exit,
# so "iptables dropped the packet" and "the sandbox never came up" are indistinguishable
# to it. A control that cannot tell those apart is not a control. It is deselected here
# with its siblings rather than left behind as a green that proves nothing.
@needs_docker
class TestNetworkingEnforcement:
    """Verify that iptables lockdown actually blocks/allows traffic."""

    @pytest.mark.netns_sidecar_egress
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

    @pytest.mark.netns_sidecar_egress
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

    # NOT a netns-sidecar failure. An Unrestricted environment with no env-var
    # credentials never invokes the sidecar at all (registry.py _apply_egress_rules
    # falls through to the no-op branch), and the job log confirms this session
    # emitted ZERO lockdown/DNAT failures -- only session_egress_state_invalidated
    # with reason "no_credentials". It fails under runsc with curl exit 6,
    # CURLE_COULDNT_RESOLVE_HOST: a real, unexplained DNS defect under gVisor.
    # Marked with its own honest cause (#2430) rather than being buried under the
    # nat/netstack story, which would have silenced a genuine signal.
    @pytest.mark.runsc_dns_unresolved
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
