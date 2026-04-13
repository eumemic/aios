"""Context builder for the step function.

Two public functions:

* :func:`should_call_model` — predicate: should the current wake result
  in an inference call, or should the step early-out?

* :func:`build_messages` — assemble the chat-completions message list
  from the event log, synthesizing ``"pending"`` results for in-flight
  tool calls and reordering tool results so they appear immediately
  after their requesting assistant message.

Both are pure functions: no DB access, no side effects, easy to test.

The ``reacting_to`` field on assistant messages is the key coordination
mechanism. Each assistant message records the seq of the latest user or
tool_result event that was in its context. ``should_call_model`` uses
this to define "new" as events after ``reacting_to``, not after the
assistant's own seq. This correctly handles the race where a tool result
arrives during inference — the model's response has a ``reacting_to``
that predates the tool result, so the result is "new" and triggers a
follow-up step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from aios.harness.window import select_window
from aios.models.events import Event

# ─── should_call_model ──────────────────────────────────────────────────────


def should_call_model(events: list[Event]) -> bool:
    """Decide whether this wake should produce an inference call.

    Returns ``True`` when:

    1. There are no assistant messages yet (first turn).
    2. A user message or completed tool batch exists that the model
       hasn't reacted to (determined via the ``reacting_to`` field).

    ``reacting_to`` is the seq of the latest user/tool event in the
    context the model was given. Events after ``reacting_to`` (excluding
    the assistant message itself) are "new." This handles the race where
    a tool result arrives during inference — the result has a seq after
    ``reacting_to`` and triggers a follow-up step.
    """
    if not events:
        return False

    # Find the most recent assistant message.
    last_asst: Event | None = None
    for e in reversed(events):
        if e.kind == "message" and e.data.get("role") == "assistant":
            last_asst = e
            break

    if last_asst is None:
        return True  # first turn — no assistant response yet

    # Use reacting_to if available; fall back to the assistant's own seq
    # (backward compat with events written before this field existed).
    reacting_to: int = last_asst.data.get("reacting_to", last_asst.seq)

    # "New" = events the model hasn't reacted to.
    new_events = [e for e in events if e.seq > reacting_to and e.seq != last_asst.seq]
    if not new_events:
        return False  # duplicate / stale wake

    # User injection → always proceed.
    if any(e.kind == "message" and e.data.get("role") == "user" for e in new_events):
        return True

    # Collect ALL real tool_result tool_call_ids from the full log.
    all_real_results: set[str] = set()
    for e in events:
        if e.kind == "message" and e.data.get("role") == "tool":
            tcid = e.data.get("tool_call_id")
            if tcid:
                all_real_results.add(tcid)

    # Check if any tool batch became fully resolved in the "new" window.
    new_tool_results = [
        e
        for e in new_events
        if e.kind == "message" and e.data.get("role") == "tool" and e.data.get("tool_call_id")
    ]
    for tr in new_tool_results:
        parent = _find_assistant_for_tool_call(events, tr.data["tool_call_id"])
        if parent is None:
            continue
        parent_call_ids = {tc["id"] for tc in (parent.data.get("tool_calls") or [])}
        if parent_call_ids <= all_real_results:
            return True  # this batch is complete

    return False


# ─── build_messages ──────────────────────────────────────────────────────────


_PENDING_CONTENT = json.dumps(
    {
        "status": "pending",
        "message": (
            "This tool is still executing in the background. "
            "Its result will arrive when ready. "
            "Do not re-request this tool."
        ),
    }
)


@dataclass(slots=True)
class ContextResult:
    """Return value of :func:`build_messages`."""

    messages: list[dict[str, Any]]
    reacting_to: int  # max seq of user/tool events included in context


def build_messages(
    events: list[Event],
    *,
    system_prompt: str | None,
    window_min: int,
    window_max: int,
    model: str = "",
) -> ContextResult:
    """Assemble a chat-completions message list from the event log.

    Returns a :class:`ContextResult` containing the messages and the
    ``reacting_to`` seq — the highest seq of any user or tool_result
    event included in the context. The step function injects this into
    the assistant message so :func:`should_call_model` can determine
    what's "new" on the next wake.
    """
    # Pass 1: build tool_call_id → real tool_result data map + track seqs.
    real_results: dict[str, dict[str, Any]] = {}
    real_result_seqs: dict[str, int] = {}
    for e in events:
        if e.kind == "message" and e.data.get("role") == "tool":
            tcid = e.data.get("tool_call_id")
            if tcid:
                real_results[tcid] = e.data
                real_result_seqs[tcid] = e.seq

    # Pass 2: walk events, emitting messages in API-valid order.
    # Track the max seq of user/tool events we include.
    emitted_tcids: set[str] = set()
    messages: list[dict[str, Any]] = []
    max_stimulus_seq: int = 0  # "stimulus" = user or tool_result

    for e in events:
        if e.kind != "message":
            continue

        role = e.data.get("role")

        if role == "user":
            messages.append(e.data)
            max_stimulus_seq = max(max_stimulus_seq, e.seq)

        elif role == "assistant":
            messages.append(e.data)
            # Emit tool results (real or synthetic) for each tool_call.
            for tc in e.data.get("tool_calls") or []:
                tcid = tc.get("id")
                if not tcid or tcid in emitted_tcids:
                    continue
                if tcid in real_results:
                    messages.append(real_results[tcid])
                    max_stimulus_seq = max(max_stimulus_seq, real_result_seqs[tcid])
                else:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tcid,
                            "content": _PENDING_CONTENT,
                        }
                    )
                emitted_tcids.add(tcid)

        elif role == "tool":
            tcid = e.data.get("tool_call_id")
            if tcid and tcid not in emitted_tcids:
                messages.append(e.data)
                emitted_tcids.add(tcid)
                max_stimulus_seq = max(max_stimulus_seq, e.seq)

    # Apply windowing on the assembled messages.
    if messages:
        messages = select_window(
            messages,
            min_tokens=window_min,
            max_tokens=window_max,
            token_counter=lambda m: _approx_tokens(m, model),
        )

    # Prepend system prompt.
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    return ContextResult(messages=messages, reacting_to=max_stimulus_seq)


# ─── helpers ─────────────────────────────────────────────────────────────────


def _find_assistant_for_tool_call(events: list[Event], tool_call_id: str) -> Event | None:
    """Find the assistant message that contains ``tool_call_id``."""
    for e in reversed(events):
        if e.kind != "message" or e.data.get("role") != "assistant":
            continue
        for tc in e.data.get("tool_calls") or []:
            if tc.get("id") == tool_call_id:
                return e
    return None


def _approx_tokens(message: dict[str, Any], model: str) -> int:
    """Rough token count for a message dict."""
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    total_chars = len(content)
    for tc in tool_calls:
        fn = tc.get("function") or {}
        total_chars += len(fn.get("name") or "") + len(fn.get("arguments") or "")
    return max(1, total_chars // 4)
