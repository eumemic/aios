"""Concise output style: per-agent steering toward short, direct output.

When ``agent.output_style == "concise"`` the harness adds two injections, both
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

from typing import Any, Final

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

The mute this guards against (#2262) is a per-step pressure, so its
counter-pressure has to sit at the same recency: the nag is the final
message of the payload, landing after the channels tail, which is exactly
where the "stay terse, don't post" reading was winning. Rendered only when
the session has bound channels — for a channel-less agent there is no
connector to send through and the clause would be noise."""

CONCISE_NAG_CONTENT: Final[str] = f"<system-reminder>{CONCISE_NAG_BODY}</system-reminder>"
"""The tail reminder for a session with no bound channels."""

CONCISE_NAG_CONTENT_CHANNELS: Final[str] = (
    f"<system-reminder>{CONCISE_NAG_BODY}{CONCISE_NAG_DELIVERY_CLAUSE}</system-reminder>"
)
"""The tail reminder for a channel-attached session.

Both variants are constant — only the nag's *position* moves (it is always
re-appended at the tail), which is why the message carries
:data:`~aios.harness.context.EPHEMERAL_TAIL_KEY`."""

# Local-token reserve for the nag at windowing time, mirroring
# ``OMISSION_MARKER_UPPER_BOUND_LOCAL`` (context.py): the nag is appended
# after windowing runs, so without a reserve it could push the send-time
# payload past ``window_max``.  Reserved UNCONDITIONALLY — like the omission
# marker and the tail block, any reserve may not render; the named tradeoff
# is the same one the omission marker already accepted: a non-concise agent
# over-reserves against a >=50k window floor.  The bound must cover the
# LONGEST variant — the channel-attached nag, which carries
# :data:`CONCISE_NAG_DELIVERY_CLAUSE` — because the reserve is computed at
# windowing time from the agent alone, before the composer knows which
# variant renders.  ``TestConciseNagReserve`` pins BOTH real builder outputs
# under this bound, so it is verified rather than guessed: if a future edit
# lengthens either variant past it, raise the constant — never shorten the
# clause to fit.
CONCISE_NAG_UPPER_BOUND_LOCAL: Final[int] = 128


def augment_with_concise_style(base_system: str, concise: bool) -> str:
    """Join :data:`CONCISE_STYLE_BLOCK` onto the system prompt iff ``concise``."""
    if not concise:
        return base_system
    return join_blocks(base_system, CONCISE_STYLE_BLOCK)


def build_concise_nag_message(*, has_channels: bool = False) -> dict[str, Any]:
    """The tail-reminder user message, built fresh each step.

    ``has_channels`` selects the variant: a channel-attached session also
    gets :data:`CONCISE_NAG_DELIVERY_CLAUSE`, so the reminder that steers
    the model shorter can never be read as licence to stop posting (#2262).

    Marked :data:`~aios.harness.context.EPHEMERAL_TAIL_KEY` so the
    cache-breakpoint recognizer never hosts the stable-prefix
    ``cache_control`` breakpoint on it: the content is constant but the
    *position* is per-step (always last), so a prefix cached through it
    would never be re-usable.
    """
    content = CONCISE_NAG_CONTENT_CHANNELS if has_channels else CONCISE_NAG_CONTENT
    return {"role": "user", "content": content, EPHEMERAL_TAIL_KEY: True}
