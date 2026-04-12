"""Context builder for the step function.

Two public functions:

* :func:`should_call_model` — predicate: should the current wake result
  in an inference call, or should the step early-out? The answer is yes
  when a user spoke OR when a tool batch just became fully resolved.

* :func:`build_messages` — assemble the chat-completions message list
  from the event log, synthesizing ``"pending"`` results for in-flight
  tool calls and reordering tool results so they always appear
  immediately after their requesting assistant message (which the API
  requires, even though the log may have them interleaved with user
  messages).

Both are pure functions: no DB access, no side effects, easy to test.
"""

from __future__ import annotations

import json
from typing import Any

from aios.harness.window import select_window
from aios.models.events import Event

# ─── should_call_model ──────────────────────────────────────────────────────


def should_call_model(events: list[Event]) -> bool:
    """Decide whether this wake should produce an inference call.

    Returns ``True`` when:

    1. There are no assistant messages yet (first turn).
    2. A user message arrived since the last assistant message.
    3. A tool batch (all tool_calls from one assistant message) just
       became fully resolved since the last assistant message.

    Returns ``False`` otherwise — the wake was a duplicate, a stale
    retry, or only partial tool results have arrived so far.
    """
    if not events:
        return False

    # Find the most recent assistant message.
    last_asst_seq: int | None = None
    for e in reversed(events):
        if e.kind == "message" and e.data.get("role") == "assistant":
            last_asst_seq = e.seq
            break

    if last_asst_seq is None:
        return True  # first turn — no assistant response yet

    new_events = [e for e in events if e.seq > last_asst_seq]
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

    # Check if any tool batch just became fully resolved.
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


def build_messages(
    events: list[Event],
    *,
    system_prompt: str | None,
    window_min: int,
    window_max: int,
    model: str = "",
) -> list[dict[str, Any]]:
    """Assemble a chat-completions message list from the event log.

    Handles three things the old linear builder didn't:

    1. **Reordering.** Tool results are placed immediately after their
       requesting assistant message, regardless of where they appear in
       the log's seq order.
    2. **Pending synthesis.** In-flight tool calls (no matching
       tool_result in the log) get a synthetic ``"pending"`` result so
       the chat-completions API constraint is satisfied.
    3. **Windowing.** The assembled message list is passed through
       :func:`select_window` with the agent's token bounds.
    """
    # Pass 1: build tool_call_id → real tool_result data map.
    real_results: dict[str, dict[str, Any]] = {}
    for e in events:
        if e.kind == "message" and e.data.get("role") == "tool":
            tcid = e.data.get("tool_call_id")
            if tcid:
                real_results[tcid] = e.data

    # Pass 2: walk events, emitting messages in API-valid order.
    emitted_tcids: set[str] = set()
    messages: list[dict[str, Any]] = []

    for e in events:
        if e.kind != "message":
            continue

        role = e.data.get("role")

        if role == "user":
            messages.append(e.data)

        elif role == "assistant":
            messages.append(e.data)
            # Emit tool results (real or synthetic) for each tool_call.
            for tc in e.data.get("tool_calls") or []:
                tcid = tc.get("id")
                if not tcid or tcid in emitted_tcids:
                    continue
                if tcid in real_results:
                    messages.append(real_results[tcid])
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
                # Orphan result — shouldn't happen normally but be safe.
                messages.append(e.data)
                emitted_tcids.add(tcid)
            # Otherwise: already emitted via the assistant step, skip.

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

    return messages


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
    """Rough token count for a message dict.

    Uses a simple heuristic (chars / 4) rather than pulling in the
    full litellm tokenizer, since this only drives windowing thresholds
    and doesn't need to be exact.
    """
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    total_chars = len(content)
    for tc in tool_calls:
        fn = tc.get("function") or {}
        total_chars += len(fn.get("name") or "") + len(fn.get("arguments") or "")
    return max(1, total_chars // 4)
