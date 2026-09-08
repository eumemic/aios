"""Concise output style: per-agent steering toward short, direct output.

When ``agent.output_style == "concise"`` the harness adds two injections:

* a cache-stable rules block joined into the system prompt
  (:func:`augment_with_concise_style`, called from ``compute_step_prelude``),
  and
* a one-line reminder — the "nag" — written ONCE per context window as a
  durable user-role reminder row (:data:`CONCISE_NAG_CONTENT` /
  :data:`CONCISE_NAG_CONTENT_CHANNELS`, planned by
  ``aios.harness.reminders`` and written by ``compose_step_context`` when no
  row of the current variant is in the window).

A plain user-content message is the only mid-transcript steering channel
that survives LiteLLM's provider transforms: Anthropic and Gemini hoist any
``role: "system"`` message into the top-level system param, destroying
position. The row is replayed at its seq on every later step, so the
prompt stays a byte-prefix of its successor (the reason it is a row and
not a per-step append: an appended tail busts the OpenAI prompt cache,
which checkpoints through the END of the prompt).
"""

from __future__ import annotations

from typing import Final

from aios.harness._text import join_blocks

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
    "These rules govern the LENGTH and SHAPE of what you write, and where "
    "they conflict with more general style guidance elsewhere in your "
    "instructions, these rules win. They do NOT govern where your output "
    "goes, which tools you call, or whether a reply is owed: any "
    "instruction about how your output reaches a person — the channel it "
    "goes to, the tool that sends it, when you owe a response — outranks "
    "these rules absolutely. Brevity is about saying less, never about "
    "saying it to no one."
)
"""Cache-stable system-prompt rules block. Constant across steps, so the
prompt prefix stays hot; joined into the prelude's system prompt iff the
agent is concise."""

CONCISE_NAG_BODY: Final[str] = (
    "Concise output style is active. Be concise: lead with the result, "
    "skip preamble and narration, keep only what the recipient needs."
)
"""The steering sentence both nag variants share."""

CONCISE_NAG_DELIVERY_CLAUSE: Final[str] = (
    " Being concise never means going silent: a reply to a person still "
    "requires the connector's send tool — bare assistant text is private "
    "thinking and reaches no one."
)
"""Appended for a channel-attached session ONLY.

The mute this guards against (#2262) is a steering pressure the nag itself
applies, so its counter-pressure rides the same message — the "stay terse,
don't post" reading was winning wherever the nag sat. Rendered only when
the session has bound channels — for a channel-less agent there is no
connector to send through and the clause would be noise. A change of
variant (a channel bound or unbound) is a content change, so the planner
writes the new variant's row."""

CONCISE_NAG_CONTENT: Final[str] = f"<system-reminder>{CONCISE_NAG_BODY}</system-reminder>"
"""The reminder row content for a session with no bound channels."""

CONCISE_NAG_CONTENT_CHANNELS: Final[str] = (
    f"<system-reminder>{CONCISE_NAG_BODY}{CONCISE_NAG_DELIVERY_CLAUSE}</system-reminder>"
)
"""The reminder row content for a channel-attached session."""

CONCISE_NAG_OFF_CONTENT: Final[str] = (
    "<system-reminder>Concise output style is no longer active: the earlier "
    "concise-style reminder in this transcript no longer applies.</system-reminder>"
)
"""Written once when the style is turned OFF while a nag is still in the
window: the system-prompt rules block is gone at that point, so without this
row the stale nag would be the only — and contradictory — steering left."""

# Local-token reserve for the nag at windowing time, mirroring
# ``OMISSION_MARKER_UPPER_BOUND_LOCAL`` (context.py): the row this step may
# write lands on top of the windowed slate, so without a reserve it could
# push the send-time payload past ``window_max``.  Reserved UNCONDITIONALLY —
# like the omission marker, any reserve may not be used; the named tradeoff
# is the same one the omission marker already accepted: a non-concise agent
# over-reserves against a >=50k window floor.  The bound must cover the
# LONGEST variant — the channel-attached nag, which carries
# :data:`CONCISE_NAG_DELIVERY_CLAUSE` — because the reserve is computed at
# windowing time from the agent alone, before the composer knows which
# variant renders.  ``TestConciseNagReserve`` pins BOTH variants and the OFF
# row under this bound, so it is verified rather than guessed: if a future
# edit lengthens any of them past it, raise the constant — never shorten the
# clause to fit.
CONCISE_NAG_UPPER_BOUND_LOCAL: Final[int] = 128


def augment_with_concise_style(base_system: str, concise: bool) -> str:
    """Join :data:`CONCISE_STYLE_BLOCK` onto the system prompt iff ``concise``."""
    if not concise:
        return base_system
    return join_blocks(base_system, CONCISE_STYLE_BLOCK)
