"""E2E regression: the full payload sent to the model must fit within
the agent's ``window_max`` — including system prompt and tool-schema
overhead.

Pre-fix, :func:`aios.db.queries.read_windowed_events` used
``cumulative_tokens`` (the per-event sum, with tools and system
excluded) as its budget target.  That undercount meant the harness
layered the system prompt and tool schemas on top at send time and
the actual prompt could exceed ``window_max`` by the entire overhead
amount.  This case pins the full-payload invariant against regression.
"""

from __future__ import annotations

from aios.harness.tokens import approx_tokens
from aios.models.agents import ToolSpec
from aios.services import agents as agents_service
from aios.services import environments as environments_service
from aios.services import sessions as sessions_service
from tests.e2e.harness import Harness, assistant


class TestWindowingOverhead:
    async def test_full_payload_fits_window_max(self, harness: Harness) -> None:
        # Agent with a meaningfully chunky system prompt plus the
        # standard built-in tool set → non-trivial per-step overhead.
        system = (
            "You are a test assistant. Be helpful, harmless, and honest. "
            "Use tools when appropriate. "
        ) * 60

        env = await environments_service.create_environment(harness._pool, name="overhead-env")
        agent = await agents_service.create_agent(
            harness._pool,
            name="overhead-agent",
            model="fake/test",
            system=system,
            tools=[ToolSpec(type=t) for t in ("bash", "read", "write", "edit", "glob", "grep")],
            description=None,
            metadata={},
            # Snug window: overhead consumes most of it, leaving only
            # ~1-2k for kept events.  Windowing is forced to drop.
            window_min=4_000,
            window_max=6_000,
        )
        session = await sessions_service.create_session(
            harness._pool,
            agent_id=agent.id,
            environment_id=env.id,
            title="overhead-test",
            metadata={},
        )
        # Pile enough user messages to force drops.
        for i in range(80):
            await sessions_service.append_user_message(
                harness._pool,
                session.id,
                f"message {i:03d}: " + "word " * 40,
            )

        harness.script_model([assistant("ack")])
        await harness.run_until_idle(session.id)

        # Exactly one model call was issued; its kwargs are the authoritative
        # record of what the provider saw.
        assert len(harness.model_calls) == 1
        call = harness.model_calls[0]
        full_local = approx_tokens(call["messages"], tools=call.get("tools"))

        assert full_local <= agent.window_max, (
            f"full payload {full_local} tokens exceeds window_max "
            f"{agent.window_max} (messages={len(call['messages'])}, "
            f"tools={len(call.get('tools') or [])})"
        )
