"""The task_complete tool — explicit terminator for ``task_call`` Stop hooks.

A session configured with a ``task_call`` Stop hook continues across
conversational end-of-turns (final text without tool calls); the only
way for the agent to surrender the loop is to call ``task_complete``.
The handler is intentionally trivial: the harness's tool-calls branch
in :mod:`aios.harness.loop` recognises ``task_complete`` and short-
circuits dispatch so the result returned here is what shows up in the
log.

When the model emits ``task_complete`` alongside other tool calls in
the same response, the harness's supersession logic runs ``task_complete``
inline and skips the siblings (see ``apply_task_complete_supersession``
in :mod:`aios.harness.loop`).
"""

from __future__ import annotations

from typing import Any

from aios.tools.registry import registry

TASK_COMPLETE_DESCRIPTION = (
    "Signal that you have finished the task this session was created for. "
    "Calling this tool surrenders the loop: the session goes idle and "
    "waits for a supervisor signal before doing anything else. Only call "
    "this when your task is fully complete — once stopped, you cannot "
    "self-resume."
)

TASK_COMPLETE_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": (
                "Optional short summary of what you completed. Stored on the "
                "tool result so a supervisor reviewing the session can see "
                "the outcome at a glance."
            ),
        },
    },
    "additionalProperties": False,
}


async def task_complete_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Return a structured completion record.

    Side-effect-free: the loop intercept runs *before* this handler in
    the supersession path, and the no-supersession path (a sole
    ``task_complete`` call) reaches dispatch normally and stores this
    result like any other tool.
    """
    summary = arguments.get("summary")
    result: dict[str, Any] = {"completed": True}
    if isinstance(summary, str) and summary:
        result["summary"] = summary
    return result


def _register() -> None:
    registry.register(
        name="task_complete",
        description=TASK_COMPLETE_DESCRIPTION,
        parameters_schema=TASK_COMPLETE_PARAMETERS_SCHEMA,
        handler=task_complete_handler,
    )


_register()
