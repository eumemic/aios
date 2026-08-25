"""Concise output style: per-agent steering toward short, direct output.

When ``agent.concise`` is true the harness adds two injections, both
assembled at step time — nothing is ever persisted to the ``agent_events``
transcript:

* a cache-stable rules block joined into the system prompt
  (:func:`augment_with_concise_style`, called from ``compute_step_prelude``),
  and
* a one-line tail reminder — the "nag" — appended as the final message of
  the composed payload (:func:`build_concise_nag_message`, called from
  ``compose_step_context``).

The nag is injected at assembly time each step so there is exactly ONE
copy, always at maximum recency, and it never pollutes the persisted
transcript. A plain user-content message is the only mid-transcript
steering channel that survives LiteLLM's provider transforms: Anthropic
and Gemini hoist any ``role: "system"`` message into the top-level system
param, destroying position.
"""

from __future__ import annotations

from typing import Any

from aios.harness._text import join_blocks
from aios.harness.context import EPHEMERAL_TAIL_KEY

CONCISE_STYLE_BLOCK = (
    "# Output style: Concise\n"
    "Keep your responses short and direct while doing the work just as "
    "thoroughly.\n"
    "- Lead with the result or the answer. No preamble, no closing recap.\n"
    "- Cut narration, keep substance: don't restate the request, your plan, "
    "or the steps you took. Report outcomes, decisions, and anything the "
    "recipient must act on.\n"
    "- Short by default: answer simple questions in 1-3 sentences.\n"
    "- State things plainly; skip hedging boilerplate. Mention a caveat only "
    "when it changes what the recipient should do next.\n"
    "- Give full detail on request: conciseness never means withholding "
    "requested information.\n"
    "- Never trade correctness for brevity: error reports, warnings, and "
    "confirmations for destructive actions keep their full content.\n"
    "Where these rules conflict with more general style guidance elsewhere "
    "in your instructions, these rules win."
)
"""Cache-stable system-prompt rules block. Constant across steps, so the
prompt prefix stays hot; joined into the prelude's system prompt iff the
agent is concise."""

CONCISE_NAG_CONTENT = (
    "<system-reminder>Concise output style is active. Be concise: lead with "
    "the result, skip preamble and narration, keep only what the recipient "
    "needs.</system-reminder>"
)
"""The tail reminder's exact text. Constant — only its position moves (it is
always re-appended at the tail), which is why the message carries
:data:`~aios.harness.context.EPHEMERAL_TAIL_KEY`."""


def augment_with_concise_style(base_system: str, concise: bool) -> str:
    """Join :data:`CONCISE_STYLE_BLOCK` onto the system prompt iff ``concise``."""
    if not concise:
        return base_system
    return join_blocks(base_system, CONCISE_STYLE_BLOCK)


def build_concise_nag_message() -> dict[str, Any]:
    """The tail-reminder user message, built fresh each step.

    Marked :data:`~aios.harness.context.EPHEMERAL_TAIL_KEY` so the
    cache-breakpoint recognizer never hosts the stable-prefix
    ``cache_control`` breakpoint on it: the content is constant but the
    *position* is per-step (always last), so a prefix cached through it
    would never be re-usable.
    """
    return {"role": "user", "content": CONCISE_NAG_CONTENT, EPHEMERAL_TAIL_KEY: True}


def concise_nag_upper_bound_local() -> int:
    """Local-token cost of the nag, reserved at windowing time.

    Mirrors :func:`~aios.harness.channels.max_tail_block_local`: the nag is
    appended after windowing, so ``read_windowed_events`` must subtract its
    cost from the budget up front or the send-time payload can overshoot
    ``window_max``. Costing the nag as a standalone message is a correct
    upper bound for the merged case too — per-message framing overhead
    exceeds the ``"\\n\\n"`` join ``merge_adjacent_user_messages`` uses.
    """
    from aios.harness.tokens import approx_tokens

    return approx_tokens([{"role": "user", "content": CONCISE_NAG_CONTENT}])
