"""E2E: every request a session sends can be rebuilt byte for byte (#2471).

Runs the REAL ``run_session_step`` against the testcontainer Postgres with the
scripted model. Each send's ``model_request_start`` span carries a ``request``
record; :func:`aios.services.requests.rebuild_request` must recompose a request
that hashes to the captured ``payload_sha`` (``exact``), for a plain turn, a turn
that writes reminder rows, a turn with a tool call still in flight, and a turn
after the window dropped history. The steady state stores no new blobs.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from aios.harness import runtime
from aios.harness.request_capture import forget_stored_blobs
from aios.models.events import Event
from aios.services import agents as agents_service
from aios.services.requests import Missing, Rebuilt, rebuild_request
from tests.e2e.harness import Harness, assistant, tool_call

_ACCOUNT = "acc_test_stub"


@pytest.fixture(autouse=True)
def _fresh_blob_cache() -> None:
    """The database is reset between tests; the worker's stored-blob cache must be too."""
    forget_stored_blobs()


def _request_spans(events: list[Event]) -> list[Event]:
    return [
        e
        for e in events
        if e.kind == "span" and e.data.get("event") == "model_request_start" and "request" in e.data
    ]


async def _rebuild_all(harness: Harness, session_id: str) -> list[Rebuilt | Missing]:
    spans = _request_spans(await harness.all_events(session_id))
    assert spans, "no captured request"
    return [
        await rebuild_request(
            runtime.require_pool(),
            account_id=_ACCOUNT,
            session_id=session_id,
            request_event_id=span.id,
        )
        for span in spans
    ]


def _fidelities(results: list[Rebuilt | Missing]) -> list[str]:
    return [r.fidelity if isinstance(r, Rebuilt) else f"missing:{r.what}" for r in results]


async def _blob_count() -> int:
    async with runtime.require_pool().acquire() as conn:
        count: int = await conn.fetchval(
            "SELECT count(*) FROM request_blobs WHERE account_id = $1", _ACCOUNT
        )
    return count


class TestRequestCapture:
    async def test_plain_turns_rebuild_exactly_and_reuse_their_blobs(
        self, harness: Harness
    ) -> None:
        harness.script_model([assistant("first answer"), assistant("second answer")])
        session = await harness.start("hello")
        await harness.run_until_idle(session.id)
        blobs_after_first = await _blob_count()
        await harness.inject_message(session.id, "and again")
        await harness.run_until_idle(session.id)

        assert _fidelities(await _rebuild_all(harness, session.id)) == ["exact", "exact"]
        # Same system prompt, tools and params: the second turn stores nothing new.
        assert await _blob_count() == blobs_after_first

    async def test_a_turn_that_writes_reminder_rows_rebuilds_exactly(
        self, harness: Harness
    ) -> None:
        harness.script_model([assistant("short answer")])
        session = await harness.start("hello", output_style="concise")
        await harness.run_until_idle(session.id)

        [span] = _request_spans(await harness.all_events(session.id))
        assert span.data["request"]["reminder_seqs"], "the concise reminder row was written"
        assert _fidelities(await _rebuild_all(harness, session.id)) == ["exact"]

    async def test_a_turn_with_a_tool_call_in_flight_rebuilds_exactly(
        self, harness: Harness
    ) -> None:
        release = asyncio.Event()

        async def slow_tool(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
            await release.wait()
            return {"ok": True}

        harness.register_tool("slow_tool", slow_tool)
        harness.script_model(
            [
                assistant(tool_calls=[tool_call("slow_tool", {}, call_id="call_slow")]),
                assistant("still waiting on the tool"),
                assistant("done"),
            ]
        )
        session = await harness.start("start the slow thing")
        await harness.run_step(session.id)
        await harness.inject_message(session.id, "any news?")
        await harness.run_step(session.id)
        release.set()
        await harness.run_until_idle(session.id)

        spans = _request_spans(await harness.all_events(session.id))
        assert spans[1].data["request"]["inflight_tool_call_ids"] == ["call_slow"]
        assert set(_fidelities(await _rebuild_all(harness, session.id))) == {"exact"}

    async def test_a_turn_after_the_window_dropped_history_rebuilds_exactly(
        self, harness: Harness
    ) -> None:
        harness.script_model([assistant(f"answer {i}") for i in range(6)])
        session = await harness.start("message 0 " + "lorem ipsum " * 300)
        agent = await harness.session(session.id)
        assert agent.agent_id is not None
        current = await agents_service.get_agent(
            runtime.require_pool(), agent.agent_id, account_id=_ACCOUNT
        )
        await agents_service.update_agent(
            runtime.require_pool(),
            agent.agent_id,
            account_id=_ACCOUNT,
            expected_version=current.version,
            window_min=1_500,
            window_max=3_000,
        )
        await harness.run_until_idle(session.id)
        for i in range(1, 6):
            await harness.inject_message(session.id, f"message {i} " + "lorem ipsum " * 300)
            await harness.run_until_idle(session.id)

        spans = _request_spans(await harness.all_events(session.id))
        snapped = [s for s in spans if s.data["request"]["omission"] is not None]
        assert snapped, "the window never dropped history"
        assert snapped[-1].data["request"]["slate"]["after_seq"] is not None
        assert set(_fidelities(await _rebuild_all(harness, session.id))) == {"exact"}
