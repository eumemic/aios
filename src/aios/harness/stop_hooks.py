"""Stop hook helpers shared by the prelude and the harness loop.

Three concerns live here so the loop and the prelude don't grow
hook-type ``if/elif`` ladders independently:

1. :func:`augment_with_stop_hook` — append the hook's termination prose
   to the system prompt (matching ``augment_with_focal_paradigm`` /
   ``augment_with_memory_stores`` pattern).
2. :func:`should_inject_task_complete` — whether the prelude needs to
   add ``task_complete`` to the agent's tool list.
3. :func:`evaluate_self_check` — interpret the agent's final assistant
   message text against a ``self_check`` hook's ``stop_on`` field.
"""

from __future__ import annotations

import re
from typing import Any

from aios.models.sessions import (
    DEFAULT_CONTINUATION_MESSAGE,
    DEFAULT_SELF_CHECK_CONTINUATION_MESSAGE,
    AlwaysContinueStopHook,
    SelfCheckStopHook,
    StopHookSpec,
    TaskCallStopHook,
)


def _self_check_block(hook: SelfCheckStopHook) -> str:
    return (
        "## Stop condition (self-check)\n\n"
        "This session continues across conversational end-of-turns. "
        "At each would-stop point the harness evaluates your final "
        "message against the condition below:\n\n"
        f"> {hook.prompt}\n\n"
        f"If your message's first word is `{hook.stop_on}` "
        "(case-insensitive), the session pauses for supervisor input. "
        "Otherwise you'll be woken with a continuation reminder. Only "
        f"answer with `{hook.stop_on}` when the condition is fully and "
        "clearly met."
    )


def _task_call_block(_hook: TaskCallStopHook) -> str:
    return (
        "## Stop condition (task_complete)\n\n"
        "This session continues across conversational end-of-turns. "
        "Final text without tool calls will NOT stop the session — "
        "you'll be woken to continue. The only way to surrender the "
        "loop is to call the `task_complete()` tool. Call it when "
        "your task is fully complete."
    )


def _always_continue_block(_hook: AlwaysContinueStopHook) -> str:
    return (
        "## Stop condition (always_continue)\n\n"
        "This session is in continuous mode. Conversational end-of-turn "
        "does not apply — you'll be woken to continue your task. The "
        "only way this session pauses is an external interrupt (e.g. a "
        "supervisor message or budget exhaustion)."
    )


def build_stop_hook_block(hook: StopHookSpec | None) -> str:
    """Render the system-prompt section describing the active hook (or ``''``)."""
    if hook is None:
        return ""
    if isinstance(hook, SelfCheckStopHook):
        return _self_check_block(hook)
    if isinstance(hook, TaskCallStopHook):
        return _task_call_block(hook)
    return _always_continue_block(hook)


def augment_with_stop_hook(base_system: str, hook: StopHookSpec | None) -> str:
    """Append the stop-hook block to ``base_system`` when a hook is set."""
    block = build_stop_hook_block(hook)
    if not block:
        return base_system
    if base_system:
        return base_system + "\n\n" + block
    return block


def should_inject_task_complete(hook: StopHookSpec | None) -> bool:
    """True iff the prelude must add ``task_complete`` to the tool list.

    ``task_call`` hooks need it (the model has no other way to
    surrender the loop); other hook types do not.
    """
    return isinstance(hook, TaskCallStopHook)


# Word-extraction regex matches the OpenAI assistant-message convention
# where the model leads with the term it wants the harness to key on.
# Allows letters, digits, hyphens; rejects punctuation so trailing
# commas / periods don't confuse the comparison.
_FIRST_WORD = re.compile(r"[A-Za-z0-9_-]+")


def evaluate_self_check(hook: SelfCheckStopHook, assistant_text: str) -> bool:
    """Return True iff ``assistant_text`` opens with ``stop_on`` (case-insensitive).

    The match is "first word of the stripped message vs. stripped
    ``stop_on``", case-insensitive.  Anything richer would invite
    subtle false positives (a model that explains why the task is NOT
    yet done, but mentions the literal stop word mid-sentence, would
    falsely terminate).
    """
    match = _FIRST_WORD.search(assistant_text.strip())
    if not match:
        return False
    return match.group(0).lower() == hook.stop_on.strip().lower()


def continuation_message(hook: StopHookSpec) -> str:
    """The user-role text to inject when a hook denies stop.

    Operators can override per-hook; otherwise the default depends on
    the hook type (self_check gets a stop-condition-tuned default so
    the wake reminder isn't the same prose as the always_continue
    case).
    """
    if hook.continuation_message:
        return hook.continuation_message
    if isinstance(hook, SelfCheckStopHook):
        return DEFAULT_SELF_CHECK_CONTINUATION_MESSAGE
    return DEFAULT_CONTINUATION_MESSAGE


def extract_assistant_text(assistant_msg: dict[str, Any]) -> str:
    """Pull the plain-text content out of a chat-completions assistant message.

    ``content`` may be ``None`` (tool-only response), a plain string,
    or a multimodal content-parts list (text + images).  Returns the
    concatenated text portions, ``""`` if nothing usable is present.
    """
    content = assistant_msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""
