"""E2E test harness — real system, scripted model.

Provides the :class:`Harness` class (wired into tests via a pytest
fixture in ``conftest.py``) plus response builder helpers and assertion
helpers. A test author only needs to script the model's responses and
verify the event log.

Example::

    async def test_tool_round_trip(harness):
        harness.script_model([
            assistant(tool_calls=[bash("echo hello")]),
            assistant("The output was hello."),
        ])
        session = await harness.start("Say hello via bash")
        await harness.run_until_idle(session.id)

        events = await harness.events(session.id)
        assert "hello" in first_tool_result(events)["content"]
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import asyncpg

from aios.harness.inflight_tool_registry import InflightToolRegistry
from aios.harness.loop import run_session_step
from aios.models.events import Event
from aios.models.sessions import Session
from aios.services import agents as agents_service
from aios.services import environments as environments_service
from aios.services import sessions as sessions_service
from aios.tools.registry import ToolHandler, registry

# ─── response builder helpers ────────────────────────────────────────────────


def assistant(
    content: str = "",
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    """Build a scripted assistant response dict.

    ``finish_reason`` (when set) is threaded onto the fake litellm envelope's
    ``choices[0]`` so tests can exercise refusal handling
    (``finish_reason="content_filter"``). It is stored under a reserved key and
    stripped before the dict becomes the assistant message.
    """
    d: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        d["tool_calls"] = tool_calls
    if finish_reason is not None:
        d["_finish_reason"] = finish_reason
    return d


_call_counter = 0


def _next_call_id() -> str:
    global _call_counter
    _call_counter += 1
    return f"call_{_call_counter}"


def tool_call(
    name: str,
    arguments: dict[str, Any] | str = "{}",
    *,
    call_id: str | None = None,
) -> dict[str, Any]:
    """Build a tool_call dict for an assistant response."""
    args_str = json.dumps(arguments) if isinstance(arguments, dict) else arguments
    return {
        "id": call_id or _next_call_id(),
        "type": "function",
        "function": {"name": name, "arguments": args_str},
    }


def bash(command: str, *, call_id: str | None = None) -> dict[str, Any]:
    """Shorthand for tool_call("bash", {"command": ...})."""
    return tool_call("bash", {"command": command}, call_id=call_id)


# ─── fake litellm response ──────────────────────────────────────────────────


def _split_finish_reason(resp: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Split a scripted response into ``(message, finish_reason)``.

    The ``_finish_reason`` key (set by :func:`assistant`) is a test-only
    envelope field, not part of the assistant message litellm returns — strip
    it so it doesn't leak into the persisted message dict.
    """
    msg = {k: v for k, v in resp.items() if k != "_finish_reason"}
    return msg, resp.get("_finish_reason")


class _FakeMessage:
    """Mimics the litellm Message object with model_dump()."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def model_dump(self) -> dict[str, Any]:
        return dict(self._data)


# ─── fake streaming response ────────────────────────────────────────────────


class _FakeDelta:
    """Mimics litellm's streaming Delta object."""

    def __init__(
        self,
        content: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _FakeStreamChoice:
    """Mimics a streaming chunk's choices[0]."""

    def __init__(self, delta: _FakeDelta) -> None:
        self.delta = delta


class _FakeChunk:
    """Mimics a single streaming chunk from litellm."""

    def __init__(self, delta: _FakeDelta) -> None:
        self.choices = [_FakeStreamChoice(delta)]


class _FakeStream:
    """Async iterator mimicking litellm's CustomStreamWrapper."""

    def __init__(self, chunks: list[_FakeChunk]) -> None:
        self._chunks = chunks
        self._idx = 0

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self) -> _FakeChunk:
        if self._idx >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._idx]
        self._idx += 1
        return chunk


# ─── assertion helpers ───────────────────────────────────────────────────────


def first_tool_result(events: list[Event]) -> dict[str, Any]:
    """Return the data dict of the first tool-role message event."""
    for e in events:
        if e.kind == "message" and e.data.get("role") == "tool":
            return e.data
    raise AssertionError("no tool result found in events")


def tool_results(events: list[Event]) -> list[dict[str, Any]]:
    """Return all tool-role message event data dicts."""
    return [e.data for e in events if e.kind == "message" and e.data.get("role") == "tool"]


def last_assistant_content(events: list[Event]) -> str:
    """Return the content of the last assistant message event."""
    for e in reversed(events):
        if e.kind == "message" and e.data.get("role") == "assistant":
            return e.data.get("content") or ""
    raise AssertionError("no assistant message found in events")


def msg_text(msg: dict[str, Any]) -> str:
    """Extract plain text from a message's content (string or content blocks)."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


# ─── structural assertion helpers ────────────────────────────────────────────
#
# These extend the read-only block above. They never replace
# first_tool_result/tool_results/last_assistant_content/msg_text. They assert
# the *shape* of a turn — which tools were called, that each call got a paired
# result, in what order, and that the turn cleanly ended — by folding over the
# captured event log. They append nothing and wake nothing.
#
# The pairing mechanism is the existing tool_call_id join: an assistant
# event carries data["tool_calls"][*]["id"]; the matching tool-role event
# carries data["tool_call_id"] (the same key the context builder uses).
#
# events()/all_events() rule (load-bearing): the lifecycle helpers
# (terminal_lifecycle/assert_ended) MUST be fed all_events() because a
# turn_ended marker is a kind=="lifecycle" event; Harness.events() returns
# message-kind events only and will never carry it. The other helpers accept
# either, but the canonical call site passes all_events() so a test never
# silently misses a lifecycle.


def requested_tool_calls(events: list[Event]) -> list[tuple[str, str, str]]:
    """(tool_call_id, name, raw_arguments_json) for every assistant tool_call,
    in log order. Reads e.data['tool_calls'][*].{id, function.name, function.arguments}.
    arguments is the raw JSON string the model emitted (the harness's tool_call()
    helper stores it as a JSON string)."""
    out: list[tuple[str, str, str]] = []
    for e in events:
        if e.kind == "message" and e.data.get("role") == "assistant":
            for tc in e.data.get("tool_calls") or []:
                fn = tc.get("function", {})
                out.append((tc["id"], fn.get("name", ""), fn.get("arguments", "")))
    return out


def result_for(events: list[Event], tool_call_id: str) -> dict[str, Any] | None:
    """The tool-role message event data whose data['tool_call_id'] == tool_call_id,
    else None (unresolved / in-flight). Last write wins if duplicated."""
    found: dict[str, Any] | None = None
    for e in events:
        if (
            e.kind == "message"
            and e.data.get("role") == "tool"
            and e.data.get("tool_call_id") == tool_call_id
        ):
            found = e.data
    return found


def paired_calls(
    events: list[Event],
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    """(assistant tool_call dict, matching tool result data | None) by tool_call_id,
    in request order. None == genuinely unresolved (in-flight or absent) — never a
    silent skip."""
    pairs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for e in events:
        if e.kind == "message" and e.data.get("role") == "assistant":
            for tc in e.data.get("tool_calls") or []:
                pairs.append((tc, result_for(events, tc["id"])))
    return pairs


def assert_tool_called(events: list[Event], name: str) -> dict[str, Any]:
    """Raise AssertionError if no assistant tool_call with that name; else return
    the FIRST matching tool_call dict."""
    for e in events:
        if e.kind == "message" and e.data.get("role") == "assistant":
            tc: dict[str, Any]
            for tc in e.data.get("tool_calls") or []:
                if tc.get("function", {}).get("name") == name:
                    return tc
    raise AssertionError(f"tool {name!r} was never called")


def assert_tool_result_contains(events: list[Event], name: str, needle: str) -> None:
    """Assert SOME result of a call to `name` contains `needle` in its content.
    Raises AssertionError if the tool was never called, never resulted, or no
    result content contains the needle."""
    matched_any = False
    for e in events:
        if e.kind == "message" and e.data.get("role") == "assistant":
            for tc in e.data.get("tool_calls") or []:
                if tc.get("function", {}).get("name") != name:
                    continue
                res = result_for(events, tc["id"])
                if res is None:
                    continue
                matched_any = True
                if needle in msg_text(res):
                    return
    if not matched_any:
        raise AssertionError(f"tool {name!r} produced no result to search")
    raise AssertionError(f"no {name!r} result contained {needle!r}")


def terminal_lifecycle(events: list[Event]) -> dict[str, Any] | None:
    """The LAST kind=='lifecycle' event data marking turn end, or None.
    Matches data['event'] in {turn_ended, interrupted} OR data['status']=='terminated'
    — mirrors chat.py:_is_turn_end. MUST be fed all_events() (kind!='message');
    events() filters to message kind and will never contain it."""
    found: dict[str, Any] | None = None
    for e in events:
        if e.kind != "lifecycle":
            continue
        d = e.data
        if d.get("event") in {"turn_ended", "interrupted"} or d.get("status") == "terminated":
            found = d
    return found


def assert_ended(events: list[Event]) -> None:
    """Assert terminal_lifecycle(events) is present. Feed all_events()."""
    if terminal_lifecycle(events) is None:
        raise AssertionError("no terminal lifecycle event (turn_ended/interrupted/terminated)")


# ─── the harness ─────────────────────────────────────────────────────────────


class Harness:
    """E2E test harness. Real Postgres, real step function, scripted model."""

    def __init__(
        self,
        pool: asyncpg.Pool[Any],
        inflight_tool_registry: InflightToolRegistry,
    ) -> None:
        self._pool = pool
        self._inflight_tool_registry = inflight_tool_registry
        self._responses: list[dict[str, Any]] = []
        self._response_idx = 0
        self._env_id: str | None = None
        self.model_calls: list[dict[str, Any]] = []  # captured kwargs from each litellm call
        self._last_streaming_response: dict[str, Any] | None = None

    # ── scripting ────────────────────────────────────────────────────────

    def script_model(self, responses: list[dict[str, Any]]) -> None:
        """Set the sequence of canned model responses."""
        global _call_counter
        _call_counter = 0
        self._responses = list(responses)
        self._response_idx = 0
        self.model_calls = []

    def register_tool(
        self,
        name: str,
        handler: ToolHandler,
        *,
        description: str = "test tool",
        schema: dict[str, Any] | None = None,
    ) -> None:
        """Register a custom tool for this test."""
        registry.register(
            name=name,
            description=description,
            parameters_schema=schema
            or {"type": "object", "properties": {}, "additionalProperties": True},
            handler=handler,
        )

    # ── session lifecycle ────────────────────────────────────────────────

    async def start(
        self,
        message: str,
        *,
        tools: list[str] | None = None,
        tool_specs: list[Any] | None = None,
        system: str = "You are a test assistant.",
        environment_config: Any | None = None,
    ) -> Session:
        """Create an agent + session and append the initial user message.

        Pass ``tools=["bash", "read"]`` for simple built-in lists, or
        ``tool_specs=[ToolSpec(...)]`` for full control (e.g. permission
        policies). Only one of the two should be provided.

        Pass ``environment_config`` to create a fresh environment with
        that config (e.g. for networking tests). When omitted, a shared
        default environment is reused across calls.
        """
        account_id = "acc_test_stub"  # PR 3 scaffolding
        from aios.ids import make_id

        if environment_config is not None:
            env = await environments_service.create_environment(
                self._pool,
                name=f"test-env-{make_id('env')[-8:]}",
                config=environment_config,
                account_id=account_id,
            )
            env_id = env.id
        else:
            if self._env_id is None:
                env = await environments_service.create_environment(
                    self._pool, name=f"test-env-{make_id('env')[-8:]}", account_id=account_id
                )
                self._env_id = env.id
            env_id = self._env_id

        from aios.models.agents import ToolSpec

        if tool_specs is not None:
            final_specs = tool_specs
        else:
            final_specs = [ToolSpec(type=t) for t in (tools or [])]
        agent = await agents_service.create_agent(
            self._pool,
            name=f"test-agent-{make_id('agent')[-8:]}",
            model="fake/test",
            system=system,
            tools=final_specs,
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
            account_id=account_id,
        )
        session = await sessions_service.create_session(
            self._pool,
            agent_id=agent.id,
            environment_id=env_id,
            title="e2e-test",
            metadata={},
            account_id=account_id,
        )
        await sessions_service.append_user_message(
            self._pool, session.id, message, account_id=account_id
        )
        return await sessions_service.get_session(self._pool, session.id, account_id=account_id)

    async def inject_message(self, session_id: str, content: str) -> None:
        """Append a user message mid-turn."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        await sessions_service.append_user_message(
            self._pool, session_id, content, account_id=account_id
        )

    async def confirm_tool(
        self,
        session_id: str,
        tool_call_id: str,
        result: str = "allow",
        deny_message: str | None = None,
    ) -> None:
        """Submit a tool confirmation (allow or deny) and wake the session."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        if result == "allow":
            await sessions_service.confirm_tool_allow(
                self._pool, session_id, tool_call_id, account_id=account_id
            )
        else:
            await sessions_service.confirm_tool_deny(
                self._pool,
                session_id,
                tool_call_id,
                deny_message or "Denied.",
                account_id=account_id,
            )

    # ── step execution ───────────────────────────────────────────────────

    async def run_step(self, session_id: str) -> None:
        """Run one inference step. Mocks are installed at fixture scope."""
        await run_session_step(session_id)

    async def wait_for_tools(self, session_id: str) -> None:
        """Wait for all in-flight tool tasks to complete."""
        for _ in range(100):  # safety cap
            tasks = self._inflight_tool_registry._tasks.get(session_id, {})
            pending = [t for t in tasks.values() if not t.done()]
            if not pending:
                return
            await asyncio.gather(*pending, return_exceptions=True)
        raise RuntimeError("wait_for_tools: tasks never completed")

    async def run_until_idle(self, session_id: str, *, max_steps: int = 20) -> None:
        """Run steps until the sweep says no inference is needed."""
        for _ in range(max_steps):
            await self.wait_for_tools(session_id)
            await self.run_ghost_repair(session_id)
            needs = await self.sessions_needing_inference(session_id)
            if session_id not in needs:
                return
            await self.run_step(session_id)
        raise RuntimeError(f"run_until_idle: hit max_steps={max_steps}")

    async def simulate_sigkill(self, session_id: str) -> None:
        """Simulate SIGKILL: cancel all in-flight tasks for a session
        without letting any cancel-path code append events.

        Mocks the lowest-level ``queries.append_event`` so every write
        path — the cancellation handler's tool_result, the finally
        block's ``tool_execute_end`` span, and the tail
        ``_trigger_sweep``'s ghost-repair via ``append_tool_result`` —
        is suppressed during cancellation cleanup, then restores it.
        After this call, the tasks are gone and the log shows what a
        real SIGKILL would leave: no events from the cancelled tasks.

        **The crash-state contract this simulates (issue #1757 audit).** A
        real worker SIGKILL leaves exactly:

        (a) a committed event prefix — ``append_event`` (db/queries/events.py)
            is one transaction under a per-session row lock, so Postgres
            atomicity guarantees the log ends at a clean event boundary, never
            mid-write. This mock's suppression of ``append_event`` during
            cancel-path cleanup is what makes that boundary land exactly where
            a real kill would have landed it — not one write further.
        (b) procrastinate rows exactly as-committed, including a ``doing`` row
            still holding the job's ``lock``/``queueing_lock`` (the #147
            shape, enumerated against the real schema in
            ``tests/e2e/test_reap_stalled_jobs.py``).
        (c) RUNNING sandbox containers — graceful shutdown STOPs every managed
            container; a SIGKILL runs no shutdown code at all, so a container
            provisioned pre-kill is left RUNNING, not stopped (see
            ``tests/e2e/test_sandbox_salvage.py::test_running_corpse_is_salvaged_then_resumed``).
        (d) empty process-local state — ``InflightToolRegistry`` and the
            in-flight-harvest key set (``aios.harness.model_workflow``) are
            in-memory only; a killed process takes them to zero along with
            itself. The successor worker starts with fresh empty registries
            by construction — nothing to reset there — but any DURABLE
            residue that in-memory state was tracking (a park with no live
            harvester, a tool call with no live task) needs its own boot-time
            sweep to notice. See :func:`aios.harness.worker.run_startup_recovery`.

        **Checklist for future process-local worker state:** any NEW
        in-memory registry the harness adds (mirroring
        ``InflightToolRegistry`` / ``_INFLIGHT_HARVESTS``) MUST ship (1) a
        boot-time reset analogous to ``reset_inflight_harvests`` (#1635) if a
        stale in-memory key could suppress correct recovery, and (2) a seeded
        crash-recovery test exercising the durable state that in-memory
        registry was shadowing — not just a unit test of the registry itself.

        **Fidelity note — this is a simulator, not a real kill.** This method
        runs REAL cancel-path code (the ``except asyncio.CancelledError``
        handlers in ``tool_dispatch.py``) with only ``queries.append_event``
        patched out; a genuine SIGKILL runs NO code at all past the signal.
        The two are equivalent for DB-visible state only because every write
        those handlers would have made is suppressed here — if a cancel-path
        branch is ever added that has an externally-visible side effect NOT
        mediated by ``append_event`` (a direct network call, a second query
        function, etc.), this simulator would silently stop being faithful.
        Note in particular that ``sweep.py``'s ``defer_wake`` import (line 41
        as of the #1757 audit) is NOT patched by this method or by
        ``conftest.py``'s fixtures that install it — the sweep's wake-side
        effects during ghost repair are real writes, not cancel-path writes,
        so they are outside this simulator's suppression scope by design.
        Keep the cancel path itself write-free at the queries layer (routing
        all real persistence through ``append_event``/``append_tool_result``)
        so this mock's single patch point stays sufficient.
        """
        from unittest import mock

        session_tasks = self._inflight_tool_registry._tasks.get(session_id, {})
        raw_tasks = list(session_tasks.values())
        # Remove from registry first so shutdown won't find them.
        self._inflight_tool_registry._tasks.pop(session_id, None)
        if raw_tasks:
            # Suppress DB writes during cancellation cleanup. Patch at the
            # queries layer rather than the service layer so both
            # ``services.append_event`` and ``services.append_tool_result``
            # (used by the sweep's ghost-repair path) are covered.
            with mock.patch("aios.db.queries.append_event"):
                for t in raw_tasks:
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*raw_tasks, return_exceptions=True)

    # ── sweep ────────────────────────────────────────────────────────────

    async def run_ghost_repair(self, session_id: str | None = None) -> list[tuple[str, str]]:
        """Run ghost detection and repair.

        Returns ``(session_id, tool_call_id)`` pairs for each ghost repaired.
        """
        from aios.harness.sweep import find_and_repair_ghosts

        return await find_and_repair_ghosts(
            self._pool, self._inflight_tool_registry, session_id=session_id
        )

    async def sessions_needing_inference(self, session_id: str | None = None) -> set[str]:
        """Return session IDs that the sweep considers ready for inference."""
        from aios.harness.sweep import find_sessions_needing_inference

        return await find_sessions_needing_inference(
            self._pool, self._inflight_tool_registry, session_id=session_id
        )

    # ── inspection ───────────────────────────────────────────────────────

    async def events(self, session_id: str) -> list[Event]:
        """Read all message events for the session."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        return await sessions_service.read_message_events(
            self._pool, session_id, account_id=account_id
        )

    async def all_events(self, session_id: str) -> list[Event]:
        """Read all events (all kinds) for the session."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        return await sessions_service.read_events(
            self._pool, session_id, limit=500, account_id=account_id
        )

    async def session(self, session_id: str) -> Session:
        """Fetch the current session record."""
        account_id = "acc_test_stub"  # PR 3 scaffolding
        return await sessions_service.get_session(self._pool, session_id, account_id=account_id)

    # ── internal ─────────────────────────────────────────────────────────

    def _pop_response(self, **kwargs: Any) -> dict[str, Any]:
        """Return the next scripted response wrapped in litellm envelope.

        Also records the kwargs (including ``messages``) on
        ``self.model_calls`` for test assertions about context shape.
        """
        self.model_calls.append(kwargs)
        if self._response_idx >= len(self._responses):
            raise AssertionError(
                f"model called {self._response_idx + 1} times but only "
                f"{len(self._responses)} responses were scripted"
            )
        resp = self._responses[self._response_idx]
        self._response_idx += 1
        msg, finish_reason = _split_finish_reason(resp)
        return {
            "choices": [{"message": _FakeMessage(msg), "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def _pop_streaming_response(self, **kwargs: Any) -> _FakeStream:
        """Return a fake async stream wrapping the next scripted response.

        Splits text content into small chunks for realistic streaming
        simulation. Stores the full response for ``_build_chunk_response``
        to return when ``stream_chunk_builder`` is called.
        """
        self.model_calls.append(kwargs)
        if self._response_idx >= len(self._responses):
            raise AssertionError(
                f"model called {self._response_idx + 1} times but only "
                f"{len(self._responses)} responses were scripted"
            )
        resp = self._responses[self._response_idx]
        self._response_idx += 1
        self._last_streaming_response = resp

        chunks: list[_FakeChunk] = []
        content = resp.get("content", "")
        if content:
            # Split into 4-char chunks for realistic streaming
            for i in range(0, len(content), 4):
                chunks.append(_FakeChunk(_FakeDelta(content=content[i : i + 4])))

        tool_calls = resp.get("tool_calls")
        if tool_calls:
            chunks.append(_FakeChunk(_FakeDelta(tool_calls=tool_calls)))

        # Always emit at least one chunk (finish-reason chunk)
        if not chunks:
            chunks.append(_FakeChunk(_FakeDelta()))

        return _FakeStream(chunks)

    def _build_chunk_response(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Mock for ``litellm.stream_chunk_builder``.

        Returns the full scripted response that was stored by the most
        recent ``_pop_streaming_response`` call.
        """
        assert self._last_streaming_response is not None
        msg, finish_reason = _split_finish_reason(self._last_streaming_response)
        return {
            "choices": [{"message": _FakeMessage(msg), "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
