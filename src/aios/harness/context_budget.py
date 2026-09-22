"""O(1) request-budget formation from metadata and already-known parameters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Served ceilings can differ materially from public model-card context windows.
# Unknown models deliberately have no entry: they retain window_max-only behavior.
_SERVED_CEILINGS: dict[str, int] = {
    "openai/responses/gpt-5.6-sol": 370_000,
}


def served_ceiling(model: str) -> int | None:
    """Return the empirically served input+output ceiling, when declared."""
    return _SERVED_CEILINGS.get(model)


# THE one list of spellings a caller may use to name its own output cap, in
# precedence order (first positive value wins). Defined here, in the module with
# no aios imports, because three separate surfaces must agree on it and they sit
# at different depths of the import graph:
#
#   * ``completion.resolve_output_cap`` — the ONE wire cap, which is also the
#     windowing reserve ``loop`` hands to :func:`effective_window_max`;
#   * ``output_reservation`` below — the windowing fallback when no cap resolves;
#   * ``context_admission._output_reserve`` — the final-wire admission gate.
#
# They HAD drifted: ``max_completion_tokens`` was added to the first two and not
# the third, so a request carrying only that spelling reached the wire with a
# perfectly good cap and was still reported ``unverified`` (and rejected under
# enforcement) for having "no enforced output token cap". One tuple, imported
# everywhere, is what makes that class of drift unrepresentable rather than
# merely fixed once.
#
# Sharing the tuple fixed only half of it. A spelling list has TWO halves — which
# keys name a cap, and which VALUES count as one — and the second half had
# drifted exactly the same way: the injection gate accepted bare key presence
# while both readers here required a positive int. So the three surfaces read one
# list and still disagreed. :func:`explicit_output_cap_entry` below is the whole
# answer, list and validity rule together; no caller may re-derive either half.
EXPLICIT_OUTPUT_CAP_KEYS: tuple[str, ...] = (
    "max_output_tokens",
    "max_tokens",
    "max_completion_tokens",
)


def is_output_cap_value(value: Any) -> bool:
    """THE validity rule for a caller-supplied output cap: a positive ``int``.

    ``bool`` is excluded deliberately — it is an ``int`` subclass in Python, so
    ``max_tokens=True`` would otherwise parse as a cap of 1 and truncate every
    reply to a single token.

    Sharing the *list* of spellings was necessary but not sufficient: the three
    readers also have to agree on what counts as a cap. They did not. This
    module's readers required a positive int while the (since deleted)
    injection gate in ``completion`` tested mere key presence, so
    ``{"max_tokens": None}`` / ``0`` / ``False`` / ``"x"`` read as "the caller
    capped it" on the injection side and "no cap named" on both reservation
    sides. One predicate, one answer.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def explicit_output_cap_entry(params: Mapping[str, Any] | None) -> tuple[str, int] | None:
    """The winning ``(spelling, value)`` of the caller's explicit output cap, else ``None``.

    THE one parser of caller caps: the first key in :data:`EXPLICIT_OUTPUT_CAP_KEYS`
    (precedence order) whose value passes :func:`is_output_cap_value`. An invalid
    value is not a cap and never wins, even when it outranks a valid one.
    ``completion.resolve_output_cap`` needs the spelling (non-Anthropic routes send
    it verbatim); everyone else wants just the value via :func:`explicit_output_cap`.
    """
    if not params:
        return None
    for key in EXPLICIT_OUTPUT_CAP_KEYS:
        value = params.get(key)
        if is_output_cap_value(value):
            assert isinstance(value, int)
            return key, value
    return None


def explicit_output_cap(params: Mapping[str, Any] | None) -> int | None:
    """The caller's explicit output cap under any accepted spelling, else ``None``.

    ``None`` means "no cap named" (distinct from a cap of 0): the admission gate
    must tell "uncapped, therefore unverifiable" from "capped at some value".
    **An invalid value is also ``None``** — a cap the provider cannot honour is
    not a cap. Shared by ``context_admission._output_reserve`` and
    :func:`output_reservation`; ``completion.resolve_output_cap`` uses the same
    parser via :func:`explicit_output_cap_entry`.
    """
    entry = explicit_output_cap_entry(params)
    return entry[1] if entry is not None else None


def output_reservation(params: Mapping[str, Any] | None) -> int:
    """Return the request's explicit maximum output/reasoning reservation."""
    cap = explicit_output_cap(params)
    if cap is not None:
        return cap
    if not params:
        return 0
    thinking = params.get("thinking")
    if isinstance(thinking, dict):
        value = thinking.get("budget_tokens")
        if is_output_cap_value(value):
            assert isinstance(value, int)
            return value
    return 0


def effective_window_max(
    *,
    model: str,
    window_max: int,
    params: dict[str, Any] | None,
    output_reserve: int | None = None,
    context_limit: int | None = None,
    shrink_factor: float = 1.0,
) -> int:
    """Form the total input budget before the windower subtracts class masses.

    The caller passes the result to ``read_windowed_events``; that function
    subtracts its already-computed system/tools/current-turn reserves using the
    calibrated class coefficients. No assembled-context token pass is involved.

    **The output reservation is only ever subtracted from a CEILING, never from
    ``window_max``.** The two are different quantities: ``window_max`` is the
    operator's own input budget for the agent, while the ceiling is the total
    the provider charges ``input + max_tokens`` against. Subtracting the
    reservation from ``window_max`` would silently shrink every agent's
    configured history by the model's output ceiling, and for any agent whose
    ``window_max`` is at or below that ceiling (``window_max`` only validates
    ``ge=1``) it collapses the budget to 1 — at which point
    ``read_windowed_events`` raises ``"no budget remains for events"`` and the
    step hard-fails. ``min(window_max, ceiling - reservation)`` is
    correct-by-construction instead: the operator's cap and the provider's cap
    each bind independently, and neither is corrupted by the other.

    ``context_limit`` supplies that ceiling for routes that have no entry in
    :data:`_SERVED_CEILINGS` but whose limit the caller can resolve from model
    metadata (see :func:`~aios.harness.completion.resolve_output_cap` —
    Anthropic-shaped routes, where ``input + max_tokens`` is charged against one
    limit). ``served_ceiling`` still wins when declared: an empirically measured
    served ceiling outranks a published model-card number.
    """
    ceiling = served_ceiling(model)
    if ceiling is None:
        ceiling = context_limit
    reservation = output_reservation(params) if output_reserve is None else output_reserve
    if ceiling is None:
        # Unmapped models retain window_max-only behavior at full budget
        # (shrink_factor == 1.0, today's semantics). But an overflow retry
        # (shrink_factor < 1) MUST still tighten the budget here — otherwise the
        # retry re-sends the identical oversized request and loops verbatim up
        # the reschedule ladder (the 2026-07-09 Ultron/sol outage class).
        return max(1, int(window_max * shrink_factor))
    input_cap = min(window_max, max(1, ceiling - reservation))
    return max(1, int(input_cap * shrink_factor))
