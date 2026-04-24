"""E2E regression: the full payload sent to the model must fit within
the agent's ``window_max`` — including system prompt, tool-schema, and
channels-tail-block overhead.

PR #165 established the full-payload invariant but only counted the
system prompt + tool schemas.  The channels tail block
(:func:`~aios.harness.channels.build_channels_tail_block`) is appended
in :func:`~aios.harness.step_context.compose_step_context` AFTER
windowing runs, so its size was leaking out of the budget too.  On JN
(6 Signal bindings) that leak was ~1.5K provider tokens — enough to
push the post-windowing prompt over ``window_max`` by the tail size.
This file pins BOTH contributions (system+tools AND tail block) against
regression.
"""

from __future__ import annotations

import uuid

from aios.harness.tokens import approx_tokens
from aios.models.agents import ToolSpec
from aios.services import agents as agents_service
from aios.services import channels as channels_service
from aios.services import connections as connections_service
from aios.services import environments as environments_service
from aios.services import sessions as sessions_service
from aios.services import vaults as vaults_service
from tests.e2e.harness import Harness, assistant


class TestWindowingOverhead:
    async def test_full_payload_fits_window_max(self, harness: Harness) -> None:
        # Agent with a meaningfully chunky system prompt plus the
        # standard built-in tool set → non-trivial per-step overhead.
        system = (
            "You are a test assistant. Be helpful, harmless, and honest. "
            "Use tools when appropriate. "
        ) * 60

        # Unique model name isolates this test's calibration aggregate
        # from any other e2e case that uses the shared "fake/test" alias
        # — otherwise leftover model_request_end spans from earlier tests
        # can activate a per-fake/test R that perturbs the windowing math.
        model = f"fake/overhead-{uuid.uuid4().hex[:8]}"
        env = await environments_service.create_environment(harness._pool, name="overhead-env")
        agent = await agents_service.create_agent(
            harness._pool,
            name="overhead-agent",
            model=model,
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

    async def test_full_payload_fits_window_max_with_tail_block(self, harness: Harness) -> None:
        """Same invariant, but with channel bindings active so
        :func:`~aios.harness.channels.build_channels_tail_block` emits
        a non-trivial tail block appended after windowing.  Pre-fix the
        tail's size was not subtracted from the window budget, so its
        contents would push the send-time payload past ``window_max``.
        """
        system = (
            "You are a test assistant. Be helpful, harmless, and honest. "
            "Use tools when appropriate. "
        ) * 60
        # Same isolation reason as the sibling test — per-model calibration
        # is shared across e2e cases by default if the model string isn't
        # unique.
        model = f"fake/overhead-tail-{uuid.uuid4().hex[:8]}"
        env = await environments_service.create_environment(harness._pool, name="overhead-tail-env")
        agent = await agents_service.create_agent(
            harness._pool,
            name="overhead-tail-agent",
            model=model,
            system=system,
            tools=[ToolSpec(type=t) for t in ("bash", "read", "write", "edit", "glob", "grep")],
            description=None,
            metadata={},
            # Snug window — the tail block's ~400 local tokens push past
            # the cap if they aren't reserved during windowing.
            window_min=3_500,
            window_max=4_500,
        )
        session = await sessions_service.create_session(
            harness._pool,
            agent_id=agent.id,
            environment_id=env.id,
            title="overhead-tail-test",
            metadata={},
        )

        # Register a connector + connection so bindings resolve.  The MCP
        # URL isn't actually dialed in this test path — bindings are
        # purely a routing-metadata construct for the tail block.
        vault = await vaults_service.create_vault(harness._pool, display_name="tb", metadata={})
        await connections_service.create_connection(
            harness._pool,
            connector="signal",
            account="test",
            mcp_url="https://nope",
            vault_id=vault.id,
            metadata={},
        )
        # Six bindings with Signal-realistic address widths (UUID + base64
        # group id) — the on-wire size of JN's fan-out.  Short stub
        # addresses undercount the tail block by ~3x and hide the bug.
        addresses = [
            f"signal/test/{uuid}/base64groupid-{i:02d}-{'x' * 30}="
            for i, uuid in enumerate(f"{i:08d}-{i:04d}-{i:04d}-{i:04d}-{i:012d}" for i in range(6))
        ]
        for addr in addresses:
            await channels_service.create_binding(
                harness._pool,
                address=addr,
                session_id=session.id,
            )

        # Direct (non-channel) user messages so windowing is forced to
        # drop events.  Appended FIRST so the per-channel messages below
        # land near the tail and survive the drop — otherwise the tail
        # block renders with 0-unread / no-preview on every channel and
        # collapses to a minimal stub.
        for i in range(80):
            await sessions_service.append_user_message(
                harness._pool,
                session.id,
                f"direct message {i:03d}: " + "word " * 40,
            )

        # Per-channel inbound so the tail shows non-zero unread + a preview
        # string per channel — the full-fat shape that appears on a live
        # session.
        for addr in addresses:
            for j in range(3):
                await sessions_service.append_user_message(
                    harness._pool,
                    session.id,
                    f"inbound on {addr} — body number {j} " + "word " * 8,
                    metadata={"channel": addr},
                )

        harness.script_model([assistant("ack")])
        await harness.run_until_idle(session.id)

        assert len(harness.model_calls) == 1
        call = harness.model_calls[0]
        full_local = approx_tokens(call["messages"], tools=call.get("tools"))

        # Sanity: a non-trivial tail block really did render.  Minimum
        # floor: header + 6 listing lines with Signal-width addresses
        # and preview clauses is >= 100 local tokens.
        tail_text = next(
            (
                m["content"]
                for m in call["messages"]
                if isinstance(m.get("content"), str) and "━━━ Channels ━━━" in m["content"]
            ),
            None,
        )
        assert tail_text is not None, "channels tail block not present — preconditions broken"
        tail_local = approx_tokens([{"role": "user", "content": tail_text}])
        assert tail_local >= 100, (
            f"tail block only {tail_local} local tokens — not chunky enough "
            f"to exercise the bug path; test preconditions weak"
        )

        assert full_local <= agent.window_max, (
            f"full payload {full_local} tokens exceeds window_max "
            f"{agent.window_max} (messages={len(call['messages'])}, "
            f"tools={len(call.get('tools') or [])}, tail_local={tail_local})"
        )
