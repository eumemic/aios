"""E2E tests for the unified session sweep.

Tests assert the **correct behavior** of ghost repair and inference
detection. Some of these tests exercise scenarios that were previously
broken (SIGKILL stuck sessions) and should now pass with the sweep.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from aios.services import sessions as sessions_service
from tests.conftest import needs_docker
from tests.e2e.conftest import wait_for_predicate
from tests.e2e.harness import Harness, assistant, tool_call

pytestmark = pytest.mark.docker


async def _read_session_row(harness: Harness, session_id: str) -> dict[str, Any]:
    """Read the maintained scalar columns off ``sessions`` directly (#1746
    test support) — the sweep-derived floor and its counterpart counter
    aren't exposed through the service layer's ``Session`` model."""
    async with harness._pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT open_tool_call_count, open_tool_call_floor_seq FROM sessions WHERE id = $1",
            session_id,
        )
    assert row is not None
    return dict(row)


async def _read_floor(harness: Harness, session_id: str) -> int:
    row = await _read_session_row(harness, session_id)
    return int(row["open_tool_call_floor_seq"])


# ─── ghost recovery ──────────────────────────────────────────────────────────


@needs_docker
class TestGhostRecovery:
    async def test_all_tools_lost_after_sigkill(self, harness: Harness) -> None:
        """SIGKILL before any tool completes — all tool calls lost.

        After ghost repair: synthetic errors appear for both tools.
        After running inference: model sees the errors and responds.
        """
        tool_a_started = asyncio.Event()
        tool_b_started = asyncio.Event()
        tool_a_proceed = asyncio.Event()
        tool_b_proceed = asyncio.Event()

        async def handler_a(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            tool_a_started.set()
            await tool_a_proceed.wait()
            return {"result": "a_done"}

        async def handler_b(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            tool_b_started.set()
            await tool_b_proceed.wait()
            return {"result": "b_done"}

        harness.register_tool("tool_a", handler_a)
        harness.register_tool("tool_b", handler_b)

        call_a = tool_call("tool_a", {}, call_id="call_a")
        call_b = tool_call("tool_b", {}, call_id="call_b")
        harness.script_model(
            [
                assistant(tool_calls=[call_a, call_b]),
                assistant("Both tools failed — I'll try a different approach."),
            ]
        )

        session = await harness.start("run both tools", tools=[])

        # Step 1: model calls both tools.
        await harness.run_step(session.id)
        await asyncio.wait_for(tool_a_started.wait(), timeout=5.0)
        await asyncio.wait_for(tool_b_started.wait(), timeout=5.0)

        # Simulate SIGKILL: cancel tasks without appending results.
        await harness.simulate_sigkill(session.id)

        # Verify: no tool results in the log.
        events = await harness.events(session.id)
        tool_results = [e for e in events if e.kind == "message" and e.data.get("role") == "tool"]
        assert len(tool_results) == 0

        # Ghost repair should detect and fix both.
        repaired = await harness.run_ghost_repair(session.id)
        assert len(repaired) == 2
        repaired_ids = {tcid for _, tcid in repaired}
        assert repaired_ids == {"call_a", "call_b"}

        # Synthetic error results should now be in the log.
        events = await harness.events(session.id)
        tool_results = [e for e in events if e.kind == "message" and e.data.get("role") == "tool"]
        assert len(tool_results) == 2
        for tr in tool_results:
            assert tr.data.get("is_error") is True
            # ``tool_execute_start`` spans committed inside ``_tool_lifecycle``
            # before the handlers reached ``started.set()``, so this hits the
            # "may have completed" branch of the recovery synthesis (#685).
            assert "may have completed" in tr.data.get("content", "")

        # Session should now need inference.
        needs = await harness.sessions_needing_inference(session.id)
        assert session.id in needs

        # Model responds to the errors.
        await harness.run_step(session.id)
        events = await harness.events(session.id)
        last_asst = next(
            e
            for e in reversed(events)
            if e.kind == "message"
            and e.data.get("role") == "assistant"
            and not e.data.get("tool_calls")
        )
        assert "failed" in last_asst.data.get("content", "").lower()

    async def test_started_then_killed_single_tool(self, harness: Harness) -> None:
        """SIGKILL after a single tool's lifecycle began — recovery surfaces
        the "may have completed" branch (#685).

        Distinct from ``test_all_tools_lost_after_sigkill`` (dual-tool batch
        case): this isolates the marker logic on a single dispatched tool so
        a regression that only flipped one of two messages still fails here.
        """
        tool_started = asyncio.Event()
        tool_proceed = asyncio.Event()

        async def handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            tool_started.set()
            await tool_proceed.wait()
            return {"result": "done"}

        harness.register_tool("slow_tool", handler)

        harness.script_model(
            [
                assistant(tool_calls=[tool_call("slow_tool", {}, call_id="call_slow")]),
                assistant("Tool dispatch interrupted — moving on."),
            ]
        )

        session = await harness.start("run the slow tool", tools=[])

        await harness.run_step(session.id)
        await asyncio.wait_for(tool_started.wait(), timeout=5.0)

        await harness.simulate_sigkill(session.id)

        repaired = await harness.run_ghost_repair(session.id)
        assert len(repaired) == 1
        assert repaired[0] == (session.id, "call_slow")

        events = await harness.events(session.id)
        tool_results = [e for e in events if e.kind == "message" and e.data.get("role") == "tool"]
        assert len(tool_results) == 1
        assert tool_results[0].data.get("is_error") is True
        assert "may have completed" in tool_results[0].data.get("content", "")

        needs = await harness.sessions_needing_inference(session.id)
        assert session.id in needs

        # Model must actually react to the synthetic error.  Without this
        # assertion, a regression where ghost-repair fails to bump
        # reacting_to (or where the session status sticks in a state that
        # short-circuits the next step) would still pop the scripted
        # response without ever calling the model.
        await harness.run_step(session.id)
        events = await harness.events(session.id)
        last_asst = next(
            e
            for e in reversed(events)
            if e.kind == "message"
            and e.data.get("role") == "assistant"
            and not e.data.get("tool_calls")
        )
        assert "interrupted" in last_asst.data.get("content", "").lower()

    async def test_cross_session_sweep_may_have_completed(self, harness: Harness) -> None:
        """Cross-session ghost repair (``session_id=None`` — the
        production startup-sweep + 30s periodic-sweep shape) correctly
        classifies a span-present ghost as 'may have completed' (#685).

        Production callers: ``worker_main`` startup sweep and
        ``_periodic_sweep`` both call ``wake_sessions_needing_inference(
        pool, registry)`` without a session_id.  The per-session e2e
        tests cover the scoped query path; this one closes the gap on
        the unscoped path so a refactor that mis-scopes
        ``GHOST_SPAN_START_SQL`` only for cross-session sweeps would
        surface here.
        """
        tool_started = asyncio.Event()
        tool_proceed = asyncio.Event()

        async def handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            tool_started.set()
            await tool_proceed.wait()
            return {"result": "done"}

        harness.register_tool("xs_tool", handler)

        harness.script_model(
            [
                assistant(tool_calls=[tool_call("xs_tool", {}, call_id="call_xs")]),
                assistant("Tool dispatch interrupted."),
            ]
        )

        session = await harness.start("run xs_tool", tools=[])

        await harness.run_step(session.id)
        await asyncio.wait_for(tool_started.wait(), timeout=5.0)

        await harness.simulate_sigkill(session.id)

        # Cross-session sweep: no session_id arg — exercises the production
        # startup-sweep code path that scoped tests never hit.
        repaired = await harness.run_ghost_repair()
        repaired_pairs = {(sid, tcid) for sid, tcid in repaired}
        assert (session.id, "call_xs") in repaired_pairs

        events = await harness.events(session.id)
        tool_results = [e for e in events if e.kind == "message" and e.data.get("role") == "tool"]
        assert len(tool_results) == 1
        assert tool_results[0].data.get("is_error") is True
        assert "may have completed" in tool_results[0].data.get("content", "")

    async def test_crash_before_tool_launch(self, harness: Harness) -> None:
        """Assistant message with tool_calls exists, but tools never dispatched.

        Simulates a crash between appending the assistant message and
        calling launch_tool_calls. Ghost repair should detect and fix it.
        """
        account_id = "acc_test_stub"
        # Create a session and manually append an assistant message with
        # tool_calls, bypassing the step function entirely.
        session = await harness.start("do something", tools=[])

        async def dummy_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {"result": "done"}

        harness.register_tool("my_tool", dummy_handler)

        # Manually append the assistant message with tool_calls.
        call_x = tool_call("my_tool", {}, call_id="call_x")
        await sessions_service.append_event(
            harness._pool,
            session.id,
            "message",
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [call_x],
                "reacting_to": 1,
            },
            account_id=account_id,
        )

        # No tools launched — simulates crash before dispatch.
        # Ghost repair should find call_x.
        repaired = await harness.run_ghost_repair(session.id)
        assert len(repaired) == 1
        assert repaired[0] == (session.id, "call_x")

        # ``launch_tool_calls`` never ran, so no ``tool_execute_start`` span
        # exists — recovery hits the "never started" branch (#685).
        events = await harness.events(session.id)
        tool_results = [e for e in events if e.kind == "message" and e.data.get("role") == "tool"]
        assert len(tool_results) == 1
        assert tool_results[0].data.get("is_error") is True
        assert "did not run" in tool_results[0].data.get("content", "")

        # Session should need inference.
        needs = await harness.sessions_needing_inference(session.id)
        assert session.id in needs

        # Model can now see the error and respond.
        harness.script_model(
            [
                assistant("The tool failed, let me try again."),
            ]
        )
        await harness.run_step(session.id)

    async def test_concurrent_ghost_repair_emits_single_result(self, harness: Harness) -> None:
        """Two concurrent ``find_and_repair_ghosts`` calls must produce
        exactly one synthetic tool_result per ghost — not two.

        Pre-fix the ghost-repair append goes through ``append_event``
        directly (sweep.py:273), bypassing the session row-lock +
        ``find_tool_result_event`` idempotency check that
        ``append_tool_result`` enforces (services/sessions.py:233-282).
        There is a TOCTOU window between the read of ``result_rows``
        (line 200) and the append (line 273) — both sweeps can pass
        the "no result yet" check before either writes, then both
        write. The duplicate violates CLAUDE.md invariant #4
        (tool-always-appends-EXACTLY-one result) and pollutes the
        monotonic-context log.

        In production the two sweeps that race here are the periodic
        all-sessions sweep on the worker (`worker.py`) and the tail
        sweep fired by each tool task's ``_trigger_sweep``
        (`tool_dispatch.py`) — both share the worker event loop and
        interleave at any ``await`` between the read and the append.
        """
        account_id = "acc_test_stub"
        session = await harness.start("do something", tools=[])

        async def dummy_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {"result": "done"}

        harness.register_tool("my_tool", dummy_handler)

        call_x = tool_call("my_tool", {}, call_id="call_x")
        await sessions_service.append_event(
            harness._pool,
            session.id,
            "message",
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [call_x],
                "reacting_to": 1,
            },
            account_id=account_id,
        )

        # Two concurrent repairs. asyncio.gather starts both immediately;
        # they interleave at every ``await`` inside find_and_repair_ghosts
        # (pool.acquire, fetch result_rows, fetch lifecycle_rows, fetch
        # agent_rows, fetch span_rows, load_session_account_id,
        # append_event), so at least one scheduling order will leave both
        # sweeps believing the ghost is unresolved when they reach the
        # append.
        await asyncio.gather(
            harness.run_ghost_repair(session.id),
            harness.run_ghost_repair(session.id),
        )

        events = await harness.events(session.id)
        tool_results_for_call_x = [
            e
            for e in events
            if e.kind == "message"
            and e.data.get("role") == "tool"
            and e.data.get("tool_call_id") == "call_x"
        ]
        assert len(tool_results_for_call_x) == 1, (
            f"got {len(tool_results_for_call_x)} synthetic tool_result events "
            f"for call_x; only one is permitted by invariant #4. Pre-fix "
            f"symptom: the unlocked read-then-append in find_and_repair_ghosts "
            f"admits both concurrent sweeps to write."
        )
        # Branch assertion (#685): the assistant message was appended manually
        # without ``launch_tool_calls``, so no ``tool_execute_start`` span
        # exists — the surviving sweep MUST hit "did not run".  Locks the
        # branch decision under concurrency so a race that flipped one
        # sweep's classification would surface here.
        assert "did not run" in tool_results_for_call_x[0].data.get("content", "")

    async def test_ghost_in_earlier_batch(self, harness: Harness) -> None:
        """Multi-batch conversation. Tool lost from first batch.

        Model responded to partial results + user messages. The ghost
        from batch 1 is detected even though a later batch completed.
        """
        tool_a_started = asyncio.Event()
        tool_a_proceed = asyncio.Event()

        async def handler_a(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            tool_a_started.set()
            await tool_a_proceed.wait()
            return {"result": "a_done"}

        async def handler_b(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {"result": "b_done"}

        async def handler_c(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {"result": "c_done"}

        harness.register_tool("slow_tool", handler_a)
        harness.register_tool("fast_tool", handler_b)
        harness.register_tool("other_tool", handler_c)

        harness.script_model(
            [
                # Batch 1: slow_tool (will be lost) + fast_tool (completes).
                assistant(
                    tool_calls=[
                        tool_call("slow_tool", {}, call_id="call_slow"),
                        tool_call("fast_tool", {}, call_id="call_fast"),
                    ]
                ),
                # Model sees fast_tool result + user message, slow_tool pending.
                assistant("Fast tool done, still waiting on slow tool..."),
                # After ghost repair: model sees slow_tool error.
                assistant("Slow tool failed. Moving on."),
            ]
        )

        session = await harness.start("run tools", tools=[])

        # Step 1: model calls both tools.
        await harness.run_step(session.id)
        await asyncio.wait_for(tool_a_started.wait(), timeout=5.0)

        async def _call_fast_logged() -> bool:
            events = await harness.events(session.id)
            return any(
                e.data.get("tool_call_id") == "call_fast"
                for e in events
                if e.kind == "message" and e.data.get("role") == "tool"
            )

        await wait_for_predicate(_call_fast_logged, max_wait_s=2.5, interval_s=0.05)

        # Simulate SIGKILL of slow_tool.
        await harness.simulate_sigkill(session.id)

        # Inject user message to move the conversation forward.
        await harness.inject_message(session.id, "what's taking so long?")

        # Step 2: model responds to user + fast_tool result.
        await harness.run_step(session.id)

        # Now ghost repair should find slow_tool from batch 1.
        repaired = await harness.run_ghost_repair(session.id)
        assert len(repaired) == 1
        assert repaired[0] == (session.id, "call_slow")

        # Branch assertion (#685): slow_tool's ``tool_execute_start`` span
        # committed before simulate_sigkill (handler_a's await on the proceed
        # event only fires inside ``_tool_lifecycle``'s yielded body), so the
        # multi-batch ghost MUST hit "may have completed".
        events = await harness.events(session.id)
        slow_tool_result = next(
            e
            for e in events
            if e.kind == "message"
            and e.data.get("role") == "tool"
            and e.data.get("tool_call_id") == "call_slow"
        )
        assert "may have completed" in slow_tool_result.data.get("content", "")

        # Session should need inference (ghost error is unreacted).
        needs = await harness.sessions_needing_inference(session.id)
        assert session.id in needs

        # Step 3: model sees the ghost error.
        await harness.run_step(session.id)

    async def test_floor_advances_and_repairs_orphan_in_later_batch(self, harness: Harness) -> None:
        """The sweep-maintained floor (#1746): fully resolve batch 1, run
        several turns, then orphan a call in a LATER batch.

        The orphan must still be repaired (the floor must never exceed the
        oldest currently-open call's seq), and ``open_tool_call_floor_seq``
        must have advanced to (at least) the resolved batch's seq once batch 1
        is fully settled.
        """
        account_id = "acc_test_stub"  # PR 3 scaffolding

        async def handler_ok(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return {"result": "ok"}

        tool_lost_started = asyncio.Event()
        tool_lost_proceed = asyncio.Event()

        async def handler_lost(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            tool_lost_started.set()
            await tool_lost_proceed.wait()
            return {"result": "lost_done"}

        harness.register_tool("ok_tool", handler_ok)
        harness.register_tool("lost_tool", handler_lost)

        harness.script_model(
            [
                # Batch 1: fully resolves.
                assistant(tool_calls=[tool_call("ok_tool", {}, call_id="call_b1")]),
                # A few turns of plain conversation to advance the log.
                assistant("Turn 2."),
                assistant("Turn 3."),
                # Later batch: lost_tool orphans.
                assistant(tool_calls=[tool_call("lost_tool", {}, call_id="call_b4")]),
                assistant("Lost tool failed. Moving on."),
            ]
        )

        session = await harness.start("start", tools=[])

        # Step 1: batch 1 resolves fully.
        await harness.run_step(session.id)
        await harness.wait_for_tools(session.id)

        # Ghost repair with nothing open — floor must not move (0 stays 0,
        # since open_tool_call_count is 0 and GHOST_ASST_SQL never fetches
        # this session).
        repaired0 = await harness.run_ghost_repair(session.id)
        assert repaired0 == []

        # Turns 2 + 3: plain conversation, no tool calls.
        await harness.inject_message(session.id, "continue")
        await harness.run_step(session.id)
        await harness.inject_message(session.id, "continue again")
        await harness.run_step(session.id)

        # Batch 4: lost_tool dispatches then is lost (SIGKILL simulation).
        await harness.inject_message(session.id, "run lost_tool")
        await harness.run_step(session.id)
        await asyncio.wait_for(tool_lost_started.wait(), timeout=5.0)
        await harness.simulate_sigkill(session.id)

        floor_before = await _read_floor(harness, session.id)

        repaired = await harness.run_ghost_repair(session.id)
        assert repaired == [(session.id, "call_b4")], (
            f"the later-batch orphan must be repaired regardless of any prior "
            f"floor advance; got {repaired}"
        )

        floor_after = await _read_floor(harness, session.id)
        assert floor_after >= floor_before, "the floor must be monotonic (GREATEST-only advance)"

        # The floor must never have exceeded call_b4's owning batch seq —
        # otherwise this repair couldn't have found it at all. Cross-checked
        # here directly against the event log for an independent assertion.
        events = await sessions_service.read_message_events(
            harness._pool, session.id, account_id=account_id
        )
        b4_seq = next(
            e.seq
            for e in events
            if e.kind == "message"
            and e.data.get("role") == "assistant"
            and any(tc.get("id") == "call_b4" for tc in (e.data.get("tool_calls") or []))
        )
        assert floor_before <= b4_seq, (
            f"floor {floor_before} exceeded call_b4's batch seq {b4_seq} — "
            f"the exact permanent-wedge class this design forecloses"
        )

    async def test_over_decrement_dedup_skip_does_not_lose_older_ghost(
        self, harness: Harness
    ) -> None:
        """Over-decrement regression (finding L0-1/L0-2): a duplicate result
        dedup-skip can drive ``open_tool_call_count`` to 0 with a sibling call
        STILL open. This is the exact scenario the rejected write-path-stamp
        design failed on (it would have permanently excluded the older batch
        once the counter transiently hit 0).

        B0 = {X, Y}; resolve Y; deliver a DUPLICATE Y result (dedup-skip
        decrements again, driving the counter toward/at 0 while X is still
        open); a fresh batch B2 opens (count rises again, bypassing the
        batch-completion gate the same way a user message would) — ghost
        repair must still find X, and the floor must never have exceeded
        B0's seq at the moment X is found.

        Manually constructs the event log (pattern:
        ``test_confirmed_always_ask_ghost`` below) rather than driving it
        through scripted model turns, so the dedup-skip and the re-opening
        batch are deterministic and don't depend on ``run_step`` scheduling.
        """
        account_id = "acc_test_stub"  # PR 3 scaffolding

        harness.script_model([assistant("unused")])
        session = await harness.start("run B0", tools=[])

        # B0 = {X, Y}, both dispatched (bash is always_allow by default with
        # no tool_specs restriction here — tools=[] means no MCP/registry
        # gating applies to the ghost classifier's dispatched check for a
        # plain custom-style call name, so this exercises the same
        # "did not run" ghost path as test_all_tools_lost_after_sigkill).
        b0 = await sessions_service.append_event(
            harness._pool,
            session.id,
            "message",
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_X", "type": "function", "function": {"name": "bash"}},
                    {"id": "call_Y", "type": "function", "function": {"name": "bash"}},
                ],
                "reacting_to": 1,
            },
            account_id=account_id,
        )
        b0_seq = b0.seq

        # Resolve Y once (the legitimate result). X has no result and no
        # in-flight task — a "did not run" ghost, same as
        # test_all_tools_lost_after_sigkill / test_confirmed_always_ask_ghost.
        async with harness._pool.acquire() as conn:
            await sessions_service.append_tool_result(
                conn,
                account_id=account_id,
                session_id=session.id,
                tool_call_id="call_Y",
                content="{}",
            )

        b0_row = await _read_session_row(harness, session.id)
        assert b0_row["open_tool_call_count"] == 1  # X still open

        # Deliver a DUPLICATE Y result — dedup-skips (idempotent retry /
        # at-least-once redelivery), applying the compensating -1 a SECOND
        # time and driving the counter to 0 while X is still genuinely open.
        async with harness._pool.acquire() as conn:
            await sessions_service.append_tool_result(
                conn,
                account_id=account_id,
                session_id=session.id,
                tool_call_id="call_Y",
                content="{}",
            )

        after_dup_row = await _read_session_row(harness, session.id)
        assert after_dup_row["open_tool_call_count"] == 0, (
            "the dedup-skip compensation should have driven the counter to 0 "
            "with X still genuinely open — the exact over-decrement scenario"
        )

        # A fresh batch B2 opens (mirrors "a user message bypasses the batch
        # gate; inference opens B2" from the issue) — the counter rises again,
        # re-admitting the session past the open_tool_call_count > 0 gate.
        # B2's own call is left open too so this batch also needs repair.
        await sessions_service.append_event(
            harness._pool,
            session.id,
            "message",
            {"role": "user", "content": "what's taking so long?"},
            account_id=account_id,
        )
        b2 = await sessions_service.append_event(
            harness._pool,
            session.id,
            "message",
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_Z", "type": "function", "function": {"name": "bash"}},
                ],
                "reacting_to": None,
            },
            account_id=account_id,
        )

        raised_row = await _read_session_row(harness, session.id)
        assert raised_row["open_tool_call_count"] > 0, (
            "expected the counter to have risen once B2 opened — this is the "
            "re-admission this regression test exercises"
        )

        floor_before = await _read_floor(harness, session.id)
        assert floor_before <= b0_seq, (
            f"floor {floor_before} already exceeded B0's seq {b0_seq} before "
            f"the repair ran — the floor must never exclude a genuinely open call"
        )

        repaired = await harness.run_ghost_repair(session.id)
        repaired_tcids = {tcid for _sid, tcid in repaired}
        assert "call_X" in repaired_tcids, (
            f"OVER-DECREMENT REGRESSION: call_X (from B0, still genuinely open "
            f"despite the counter having transiently read 0) was not repaired; "
            f"got {repaired}. This is the exact permanent-wedge class the "
            f"rejected write-path-stamp design failed on."
        )
        del b2  # only its seq-side-effect (raising the counter) is needed above

    async def test_confirmed_always_ask_ghost(self, harness: Harness) -> None:
        """always_ask tool confirmed-allow, dispatched, then lost. Is a ghost.

        Manually constructs the event log state: an assistant message
        calling glob (a built-in tool), a tool_confirmed allow lifecycle
        event, but no tool result and no in-flight task. Ghost repair
        should detect it as a dispatched-but-lost tool.
        """
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.models.agents import ToolSpec

        harness.script_model([assistant("The glob tool was interrupted.")])

        session = await harness.start(
            "find files",
            tool_specs=[ToolSpec(type="glob", permission="always_ask")],
        )

        # Manually append assistant message with tool_calls.
        await sessions_service.append_event(
            harness._pool,
            session.id,
            "message",
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_g",
                        "type": "function",
                        "function": {"name": "glob", "arguments": "{}"},
                    }
                ],
                "reacting_to": 1,
            },
            account_id=account_id,
        )
        # Append lifecycle: client confirmed allow.
        await sessions_service.append_event(
            harness._pool,
            session.id,
            "lifecycle",
            {"event": "tool_confirmed", "tool_call_id": "call_g", "result": "allow"},
            account_id=account_id,
        )
        # No tool result, no in-flight task → ghost.

        repaired = await harness.run_ghost_repair(session.id)
        assert len(repaired) == 1
        assert repaired[0] == (session.id, "call_g")

        # Branch assertion (#685): the assistant message was appended via
        # ``append_event`` directly without ``_tool_lifecycle`` ever running,
        # so no ``tool_execute_start`` span exists for ``call_g`` — recovery
        # MUST hit "did not run".  A regression that interpreted the
        # ``tool_confirmed allow`` lifecycle event as a started-marker (in
        # place of, or alongside, the span) would flip this assertion.
        events = await harness.events(session.id)
        glob_result = next(
            e
            for e in events
            if e.kind == "message"
            and e.data.get("role") == "tool"
            and e.data.get("tool_call_id") == "call_g"
        )
        assert "did not run" in glob_result.data.get("content", "")

        # Session should need inference after ghost repair.
        needs = await harness.sessions_needing_inference(session.id)
        assert session.id in needs

        # Model sees the error and responds.
        await harness.run_step(session.id)


# ─── ghost exclusions ────────────────────────────────────────────────────────


@needs_docker
class TestGhostExclusions:
    async def test_unconfirmed_always_ask_not_ghost(self, harness: Harness) -> None:
        """always_ask tool waiting for client confirmation is NOT a ghost.

        Manually constructs event log: assistant calls glob (always_ask),
        no confirmation submitted. Ghost repair should skip it.
        """
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.models.agents import ToolSpec

        harness.script_model([])
        session = await harness.start(
            "find files",
            tool_specs=[ToolSpec(type="glob", permission="always_ask")],
        )

        # Manually append assistant message with tool_calls.
        await sessions_service.append_event(
            harness._pool,
            session.id,
            "message",
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_g",
                        "type": "function",
                        "function": {"name": "glob", "arguments": "{}"},
                    }
                ],
                "reacting_to": 1,
            },
            account_id=account_id,
        )
        # No confirmation, no result, no in-flight task.
        # glob is always_ask for this agent → not dispatched → NOT a ghost.
        repaired = await harness.run_ghost_repair(session.id)
        assert len(repaired) == 0

    async def test_tool_calls_null_not_ghost(self, harness: Harness) -> None:
        """Assistant message with tool_calls: null (JSON null) doesn't crash sweep.

        Some LiteLLM providers return tool_calls: null instead of omitting
        the key. Stored as JSONB null, this used to crash the ghost sweep's
        jsonb_array_length query. The message has no tool calls, so ghost
        repair should return nothing and the inference query should not crash.
        """
        account_id = "acc_test_stub"  # PR 3 scaffolding
        harness.script_model([])
        session = await harness.start("hi", tools=[])

        # Manually append an assistant message with tool_calls: null.
        # This simulates what reaches the DB from providers like kimi-k2.5
        # (the ingestion fix strips it, but existing rows may have it).
        await sessions_service.append_event(
            harness._pool,
            session.id,
            "message",
            {
                "role": "assistant",
                "content": "I have no tools to call.",
                "tool_calls": None,
                "reacting_to": 1,
            },
            account_id=account_id,
        )

        # Ghost repair must not crash and must find no ghosts.
        repaired = await harness.run_ghost_repair(session.id)
        assert len(repaired) == 0

        # Inference detection must not crash either (exercises
        # _filter_incomplete_batches which has the same query pattern).
        needs = await harness.sessions_needing_inference(session.id)
        assert session.id not in needs

    async def test_custom_tool_not_ghost(self, harness: Harness) -> None:
        """Custom (client-executed) tool waiting for result is NOT a ghost."""
        harness.script_model(
            [
                assistant(
                    tool_calls=[tool_call("ask_user", {"question": "yes?"}, call_id="call_u")]
                ),
            ]
        )

        # Don't register "ask_user" — it's a custom tool (not in registry).
        session = await harness.start("ask the user", tools=[])

        # Step 1: model calls the custom tool. Session idles.
        await harness.run_step(session.id)

        # Ghost repair should NOT flag it.
        repaired = await harness.run_ghost_repair(session.id)
        assert len(repaired) == 0


# ─── sweep waking ────────────────────────────────────────────────────────────


@needs_docker
class TestSweepWaking:
    async def test_sweep_finds_first_turn_session(self, harness: Harness) -> None:
        """Session with user message and no assistant — needs inference."""
        harness.script_model([assistant("Hello!")])
        session = await harness.start("hi")

        needs = await harness.sessions_needing_inference(session.id)
        assert session.id in needs

    async def test_batch_completion_gating_via_sweep(self, harness: Harness) -> None:
        """Sweep respects batch completion: waits for all tools in a group."""
        tool_a_started = asyncio.Event()
        tool_b_started = asyncio.Event()
        tool_b_proceed = asyncio.Event()

        async def handler_a(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            tool_a_started.set()
            return {"result": "a_done"}

        async def handler_b(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            tool_b_started.set()
            await tool_b_proceed.wait()
            return {"result": "b_done"}

        harness.register_tool("tool_a", handler_a)
        harness.register_tool("tool_b", handler_b)

        harness.script_model(
            [
                assistant(
                    tool_calls=[
                        tool_call("tool_a", {}, call_id="call_a"),
                        tool_call("tool_b", {}, call_id="call_b"),
                    ]
                ),
                assistant("Both done."),
            ]
        )

        session = await harness.start("run both", tools=[])
        await harness.run_step(session.id)

        # Wait for tool A to complete.
        await asyncio.wait_for(tool_a_started.wait(), timeout=5.0)
        await asyncio.wait_for(tool_b_started.wait(), timeout=5.0)

        async def _call_a_logged() -> bool:
            events = await harness.events(session.id)
            return any(
                e.data.get("tool_call_id") == "call_a"
                for e in events
                if e.kind == "message" and e.data.get("role") == "tool"
            )

        await wait_for_predicate(_call_a_logged, max_wait_s=2.5, interval_s=0.05)

        # Tool B is still in-flight. Sweep should say "not ready."
        needs = await harness.sessions_needing_inference(session.id)
        assert session.id not in needs

        # Let tool B complete.
        tool_b_proceed.set()
        await harness.wait_for_tools(session.id)

        # Now the batch is complete. Sweep should say "ready."
        needs = await harness.sessions_needing_inference(session.id)
        assert session.id in needs

    async def test_user_message_bypasses_batch_gate(self, harness: Harness) -> None:
        """User message always triggers inference, even with in-flight tools."""
        tool_started = asyncio.Event()
        tool_proceed = asyncio.Event()

        async def slow_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            tool_started.set()
            await tool_proceed.wait()
            return {"result": "done"}

        harness.register_tool("slow", slow_handler)

        harness.script_model(
            [
                assistant(tool_calls=[tool_call("slow", {}, call_id="call_s")]),
                assistant("Working on it..."),
            ]
        )

        session = await harness.start("do slow thing", tools=[])
        await harness.run_step(session.id)
        await asyncio.wait_for(tool_started.wait(), timeout=5.0)

        # Tool is in-flight. Sweep says "not ready" (batch incomplete).
        needs = await harness.sessions_needing_inference(session.id)
        assert session.id not in needs

        # User sends a message. Sweep should now say "ready."
        await harness.inject_message(session.id, "status?")
        needs = await harness.sessions_needing_inference(session.id)
        assert session.id in needs

        # Cleanup.
        tool_proceed.set()
        await harness.wait_for_tools(session.id)
