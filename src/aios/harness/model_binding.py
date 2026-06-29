"""The ``workflow:`` model binding — bind an agent's inference to a workflow run.

Part of the **Workflows-as-Models** epic (issue #1634). When an agent's
``model`` is ``"workflow:<wf_id>[@version]"`` the step's model dispatch is
produced by a *workflow run* rather than a raw provider call: the step opens an
**awaited** run of the bound workflow, parks owing an assistant message, and a
later step harvests the run's structured return into an ``LlmResponse`` that the
existing append/charge/dispatch tail consumes.

This module owns the **binding boundary** — the two pure transforms that bracket
that flow, kept here (free of harness/db imports) so they can be unit-tested in
isolation and reused by the dispatch site and the harvest site:

* :func:`parse_workflow_model` — recognise + split ``workflow:<id>[@version]``.
* :func:`map_run_output_to_response` — validate the inner run's structured
  return at the binding boundary and project it into an :class:`LlmResponse`,
  mapping the inner ``finish_reason`` onto the standardized outer value the tail
  branches on (notably ``content_filter`` for a refusal — ``loop.py`` keys its
  bricked-turn handling on it).

**The structured return contract (the binding boundary).** A bound workflow's
output is the same payload shape ``call_llm`` produces (``run_llm.py``): a dict
carrying ``content`` (assistant text), ``tool_calls`` (the proposed calls — the
*outer* session dispatches them under its own authority), and an optional
``finish_reason``. The full provider ``message`` is optional; when absent we
synthesize a minimal normalized assistant message from ``content``/``tool_calls``
so the persisted turn round-trips. ``usage``/``cost`` are accepted but NOT
recharged at harvest (the inference already charged at its own site — a bound
workflow meters its own ``call_llm`` spend), so they are recorded for the span
only.

**finish_reason mapping.** The inner value is normalized to the outer
standardized vocabulary:

* a refusal — inner ``finish_reason in {"content_filter", "refusal"}`` — maps to
  the outer ``content_filter`` so the tail's bricked-turn handling fires.
* an **empty** result — no ``content`` AND no ``tool_calls`` — maps to
  ``content_filter`` too: an empty inner deliberation is a non-answer the outer
  turn must not persist+dispatch as if it succeeded (it would idle the session
  owing a response with nothing to say). This is the "empty inner result maps to
  the correct outer finish_reason behavior" acceptance criterion.
* ``length`` passes through unchanged (a truncated-but-real answer).
* anything else (``stop``/``tool_calls``/``None``/unknown) normalizes to the
  presence-derived terminal: ``tool_calls`` when calls were proposed, else
  ``stop``.

A structurally invalid return (not a dict, non-string ``content``, malformed
``tool_calls``) raises :class:`BindingBoundaryError` — the binding boundary is
load-bearing, so a malformed assistant shape fails loud rather than silently
dispatching a half-shaped turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aios.harness.completion import LlmResponse

# The model-string scheme that binds an agent's inference to a workflow run.
WORKFLOW_MODEL_PREFIX = "workflow:"

# The standardized outer ``finish_reason`` for a refusal / non-answer. Kept in
# lockstep with ``aios.harness.loop.REFUSAL_FINISH_REASON`` (a refusal is a
# bricked turn the tail must not persist+dispatch); duplicated here rather than
# imported to keep this module free of the harness loop's import surface.
_REFUSAL_FINISH_REASON = "content_filter"

# Inner ``finish_reason`` values that signal a refusal at the binding boundary.
# ``refusal`` is Anthropic's raw stop_reason; ``content_filter`` is the already
# normalized value. Either maps to the outer refusal terminal.
_INNER_REFUSAL_REASONS = frozenset({"content_filter", "refusal"})


class BindingBoundaryError(ValueError):
    """A bound workflow's structured return did not satisfy the assistant-message shape.

    Raised at the binding boundary (:func:`map_run_output_to_response`) when the
    inner run's output cannot be projected into a valid assistant turn. The
    dispatch tail treats this as a failed inference (no persist, no dispatch).
    """


@dataclass(frozen=True, slots=True)
class WorkflowModelRef:
    """A parsed ``workflow:<id>[@version]`` binding.

    * ``workflow_id`` — the bound workflow's id (the part after ``workflow:`` and
      before any ``@version``).
    * ``version`` — the pinned historical version (the ``@version`` suffix), or
      ``None`` to float to the workflow's current version.
    """

    workflow_id: str
    version: int | None = None


def is_workflow_model(model: str) -> bool:
    """True iff ``model`` names a ``workflow:`` binding (cheap prefix probe)."""
    return model.startswith(WORKFLOW_MODEL_PREFIX)


def parse_workflow_model(model: str) -> WorkflowModelRef | None:
    """Parse ``workflow:<id>[@version]`` → :class:`WorkflowModelRef`, else ``None``.

    Returns ``None`` for any model that is not a ``workflow:`` binding (the common
    case — a raw provider model). Raises :class:`BindingBoundaryError` for a
    ``workflow:`` string that is malformed (empty id, empty/non-integer version),
    so a misconfigured binding fails loud at the dispatch site rather than
    launching a garbage run.
    """
    if not is_workflow_model(model):
        return None
    body = model[len(WORKFLOW_MODEL_PREFIX) :]
    version: int | None = None
    if "@" in body:
        workflow_id, _, version_str = body.partition("@")
        if not version_str:
            raise BindingBoundaryError(
                f"workflow model binding {model!r} has an empty @version pin"
            )
        try:
            version = int(version_str)
        except ValueError as exc:
            raise BindingBoundaryError(
                f"workflow model binding {model!r} has a non-integer @version pin ({version_str!r})"
            ) from exc
    else:
        workflow_id = body
    if not workflow_id:
        raise BindingBoundaryError(f"workflow model binding {model!r} names no workflow id")
    return WorkflowModelRef(workflow_id=workflow_id, version=version)


def _coerce_tool_calls(raw: Any) -> list[dict[str, Any]]:
    """Validate the inner ``tool_calls`` into a list of dicts (``[]`` when absent)."""
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(tc, dict) for tc in raw):
        raise BindingBoundaryError("bound workflow return 'tool_calls' must be a list of objects")
    return raw


def map_finish_reason(inner: str | None, *, has_content: bool, has_tool_calls: bool) -> str:
    """Map an inner ``finish_reason`` onto the outer standardized vocabulary.

    See the module docstring for the full mapping. ``has_content`` /
    ``has_tool_calls`` are the presence flags of the projected assistant turn —
    they drive both the empty-result refusal and the default terminal.
    """
    if inner in _INNER_REFUSAL_REASONS:
        return _REFUSAL_FINISH_REASON
    # An empty deliberation (no text, no calls) is a non-answer: route it through
    # the refusal terminal so the outer turn bricks rather than persisting+idling
    # a session that owes a response with nothing to say.
    if not has_content and not has_tool_calls:
        return _REFUSAL_FINISH_REASON
    if inner == "length":
        return "length"
    # Default: derive the terminal from what the turn actually carries.
    return "tool_calls" if has_tool_calls else "stop"


def map_run_output_to_response(output: Any) -> LlmResponse:
    """Project a bound workflow run's structured return into an :class:`LlmResponse`.

    The binding boundary's load-bearing validation. ``output`` is the inner run's
    terminal output (the ``{ok}`` value). On a structurally valid return this
    returns the :class:`LlmResponse` the harvest folds into ``assistant_msg``;
    otherwise it raises :class:`BindingBoundaryError`.

    ``usage``/``cost`` are carried through for the harvest span ONLY — the harvest
    records a span but does NOT re-charge (the inference already charged at its
    own ``call_llm`` site inside the run).
    """
    if not isinstance(output, dict):
        raise BindingBoundaryError(
            f"bound workflow must return an object assistant turn, got {type(output).__name__}"
        )

    content_raw = output.get("content")
    if content_raw is None:
        content = ""
    elif isinstance(content_raw, str):
        content = content_raw
    else:
        raise BindingBoundaryError("bound workflow return 'content' must be a string or null")

    tool_calls = _coerce_tool_calls(output.get("tool_calls"))

    has_content = bool(content)
    has_tool_calls = bool(tool_calls)
    finish_reason = map_finish_reason(
        output.get("finish_reason"),
        has_content=has_content,
        has_tool_calls=has_tool_calls,
    )

    # The opaque provider message: use the inner one verbatim when supplied (it
    # carries role/thinking_blocks/extensions), else synthesize a minimal
    # normalized assistant message from the projected fields so the persisted
    # turn round-trips.
    message_raw = output.get("message")
    if message_raw is None:
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
    elif isinstance(message_raw, dict):
        message = dict(message_raw)
    else:
        raise BindingBoundaryError("bound workflow return 'message' must be an object or null")

    usage_raw = output.get("usage")
    usage: dict[str, int] = usage_raw if isinstance(usage_raw, dict) else {}
    cost_raw = output.get("cost")
    cost: float | None = cost_raw if isinstance(cost_raw, (int, float)) else None

    return LlmResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
        cost=cost,
        message=message,
    )
