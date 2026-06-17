"""Slack event → normalized :class:`InboundMessage` + the four connector gates.

MVP slice 2/4 — the decision layer (design §3.4, §3.6).  This module is
the pure, no-network core that turns a raw Slack Socket-Mode event
payload into a normalized inbound and decides whether it should be
forwarded to the model.  The transport (slice A) acks the envelope and
hands the raw ``event`` dict here; the connector (slice C) wires the
outcome into ``emit_inbound`` / outbound tool calls.

Everything here is a **pure function over the event dict + cached
connection identity** (``bot_user_id`` / ``team_id`` / ``api_app_id`` /
the per-thread sent-``ts`` set).  No I/O, no Slack Web API calls — the
missing-``thread_ts`` backfill (§3.4) is a separate drain-task concern,
not a parse concern.

Two load-bearing shapes:

* :class:`InboundMessage` — a frozen, slots dataclass; the normalized
  inbound the connector emits.  ``chat_id`` is the **bare** Slack
  conversation id (``C…``/``G…``/``D…``) for both top-level and thread
  messages (threads share the channel session, §3.4); ``thread_ts``
  rides separately so the model can thread its replies via
  ``slack_send(thread_ts=…)``.
* :func:`gate` — runs the four connector-side gates in order and
  returns a :class:`GateDecision` describing the outcome (forward, drop,
  or divert to a non-emitting system path).  ``InboundMessage`` is only
  built for events that pass.

Gate order (design §3.6), all *before* ``emit_inbound``:

1. **self / bot-loop filter** — drop the bot's own messages and
   bot-authored traffic.  For ``subtype == message_changed`` the author
   identity is **nested** under ``event.message`` — a top-level read
   fails open for every edit — so the gate reads ``event.message.user`` /
   ``.bot_id`` / ``.edited.user``.  ``message_changed`` is routed to a
   **non-emitting** system path (``DIVERT_EDIT``) rather than into
   ``emit_inbound``.
2. **cross-app / cross-team filter** — drop on ``api_app_id`` /
   ``team_id`` mismatch against the connection's own identity.
3. **subtype filter** — drop non-plain-message subtypes (besides the
   ``message_changed`` diversion handled in gate 1).
4. **mention-gate** — encoded as a ``chat_kind`` discriminant, not
   booleans: ``im`` always responds; ``mpim`` / ``channel`` / ``group``
   require an explicit ``<@bot_user_id>`` mention, **with the
   ``bot_thread_participant`` implicit-mention bypass** (a thread reply
   bypasses the requirement when the thread already has bot activity).
   Gated-out messages are recorded to pending-history and dropped
   (fail-quiet).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

ChatKind = Literal["im", "mpim", "channel", "group"]

# Slack conversation-id → chat_kind discriminant.  The conversation id's
# leading glyph is the durable kind signal, but Slack also sends
# ``channel_type`` on most message events; we prefer the explicit
# ``channel_type`` and fall back to the id prefix.
_CHANNEL_TYPE_TO_KIND: dict[str, ChatKind] = {
    "im": "im",
    "mpim": "mpim",
    "channel": "channel",
    "group": "group",
}

# Length cap for the sanitized sender display name (§3.6 — strip
# newlines, length-cap so injected multi-line content can't ride into the
# rendered ``from=`` clause).  Generous enough for any real Slack display
# name; the cap is a hygiene ceiling, not a product constraint.
_DISPLAY_NAME_MAX = 256

# ``<@U0123ABCD>`` or ``<@U0123ABCD|alice>`` — Slack encodes a user
# mention as an angle-bracketed link token; the optional ``|label`` is the
# cached display label.  We only ever match against the bot's own id.
_MENTION_RE = re.compile(r"<@(?P<uid>[A-Z0-9]+)(?:\|[^>]*)?>")


@dataclass(slots=True, frozen=True)
class InboundMessage:
    """A normalized Slack inbound, ready for ``emit_inbound``.

    ``chat_id`` is the **bare** Slack conversation id (``C…``/``G…``/
    ``D…``) for both top-level and thread messages — threads share the
    channel session (§3.4), so a thread reply keys on the same
    ``chat_id`` as a top-level message in that channel.  ``thread_ts``
    rides alongside (and onto ``connector_metadata`` under a non-reserved
    key) as the model's ``slack_send(thread_ts=…)`` source.
    """

    chat_kind: ChatKind
    channel_id: str
    sender_id: str
    sender_name: str
    message_ts: str
    thread_ts: str | None
    text: str
    edited: bool
    edit_ts: str | None
    mentions: tuple[str, ...]

    @property
    def chat_id(self) -> str:
        """The bare Slack conversation id — the session/address key.

        Identical for top-level and thread messages: threads share the
        channel session (§3.4).  ``thread_ts`` is the lone thread
        authority and rides ``connector_metadata`` separately.
        """
        return self.channel_id

    @property
    def event_id(self) -> str:
        """Deterministic, redelivery- and restart-stable inbound id (§3.4).

        * messages: ``slack-{channel}-{ts}``
        * edits:    ``slack-{channel}-{ts}-e{edit_ts}``

        The edit suffix keeps an edit from dedup-colliding with the
        original on the ``connector_inbound_acks`` PK.
        """
        base = f"slack-{self.channel_id}-{self.message_ts}"
        if self.edit_ts is not None:
            return f"{base}-e{self.edit_ts}"
        return base


class GateOutcome(StrEnum):
    """The terminal disposition the gates assign to a raw Slack event."""

    #: Passes every gate — build an ``InboundMessage`` and ``emit_inbound``.
    FORWARD = "forward"
    #: Dropped by a gate (self/bot, cross-app/team, subtype, or
    #: mention-gate).  ``reason`` carries which.  Fail-quiet; the
    #: connector may record mention-gated drops to pending-history.
    DROP = "drop"
    #: ``message_changed`` routed to the non-emitting system path.
    #: The author-identity self-filter already ran on the nested
    #: ``event.message`` before this is returned, so a bot-authored edit
    #: is a ``DROP``, not a ``DIVERT_EDIT``.
    DIVERT_EDIT = "divert_edit"


@dataclass(slots=True, frozen=True)
class GateDecision:
    """The result of running the four gates over one raw Slack event.

    ``message`` is populated only when ``outcome is FORWARD``.
    ``reason`` is a short machine-readable tag for observability on the
    non-forward paths (e.g. ``"self"``, ``"bot"``, ``"cross_team"``,
    ``"subtype"``, ``"mention_required"``).
    """

    outcome: GateOutcome
    reason: str | None = None
    message: InboundMessage | None = None
    # When a mention-gated message is dropped it is still recorded to
    # per-channel pending-history for later context (§3.6); the connector
    # reads this flag to decide whether to stash it.
    record_pending: bool = False
    mentions: tuple[str, ...] = field(default_factory=tuple)


# ── normalization helpers ─────────────────────────────────────────────


def sanitize_display_name(raw: str | None, fallback: str) -> str:
    """Strip newlines and length-cap a sender display name (§3.6).

    A Slack display name is attacker-controlled free text that lands in
    the model-rendered ``from=`` clause.  Collapsing CR/LF/other control
    whitespace to single spaces and capping the length keeps it from
    smuggling multi-line injected content into that clause.  This is
    cheap hygiene, not a trust boundary — authz keys on the opaque
    ``U…`` id, never the display name.
    """
    if not raw:
        return fallback
    # Replace any run of control/separator whitespace (newlines, tabs,
    # vertical tabs, form feeds, NEL, line/paragraph separators) with a
    # single space, then collapse and trim.
    flattened = re.sub(r"[\r\n\t\v\f\x1c-\x1f\x85\u2028\u2029]+", " ", raw)
    flattened = re.sub(r"\s+", " ", flattened).strip()
    if not flattened:
        return fallback
    if len(flattened) > _DISPLAY_NAME_MAX:
        flattened = flattened[:_DISPLAY_NAME_MAX].rstrip()
    return flattened


def chat_kind_of(event: dict[str, Any]) -> ChatKind:
    """Derive the typed ``chat_kind`` discriminant for an event.

    Prefers the explicit ``channel_type`` Slack stamps on message events,
    falling back to the conversation-id leading glyph (``D…`` → ``im``,
    ``G…`` → ``group``/``mpim``, ``C…`` → ``channel``).  An ``mpim``
    can't be distinguished from a private ``group`` by id prefix alone
    (both ``G…``/``C…`` depending on workspace era), so ``channel_type``
    is authoritative when present; both ``mpim`` and ``group`` map to the
    channel mention rule anyway (§3.6), so a mis-bucket here is policy-neutral.
    """
    ctype = event.get("channel_type")
    if isinstance(ctype, str) and ctype in _CHANNEL_TYPE_TO_KIND:
        return _CHANNEL_TYPE_TO_KIND[ctype]
    channel = _channel_id_of(event)
    if channel.startswith("D"):
        return "im"
    if channel.startswith("G"):
        return "group"
    return "channel"


def _channel_id_of(event: dict[str, Any]) -> str:
    channel = event.get("channel")
    return channel if isinstance(channel, str) else ""


def extract_mentions(text: str) -> tuple[str, ...]:
    """Return the ordered tuple of user ids ``<@U…>``-mentioned in ``text``."""
    return tuple(m.group("uid") for m in _MENTION_RE.finditer(text))


def _message_body(event: dict[str, Any]) -> dict[str, Any]:
    """The author-bearing body of an event.

    For ``message_changed`` the new text + author live nested under
    ``event.message``; for a plain message they are top-level.  Reading
    the wrong level for an edit fails open on the self-filter (§3.6), so
    this is the single chokepoint every author/text read goes through.
    """
    if event.get("subtype") == "message_changed":
        nested = event.get("message")
        if isinstance(nested, dict):
            return nested
    return event


# ── gate 1: self / bot-loop filter ────────────────────────────────────


def _is_self_or_bot(event: dict[str, Any], *, bot_user_id: str) -> str | None:
    """Return a drop reason if the (possibly nested) author is self/bot.

    Reads through :func:`_message_body` so a ``message_changed`` edit is
    judged on its nested ``event.message.user`` / ``.bot_id`` /
    ``.edited.user`` — never the (absent) top-level fields.  Returns
    ``"self"``, ``"bot"``, or ``None``.
    """
    body = _message_body(event)
    if body.get("user") == bot_user_id:
        return "self"
    # An edit whose *editor* is the bot (``message.edited.user``) is just
    # as much a self-loop as a bot-authored original.
    edited = body.get("edited")
    if isinstance(edited, dict) and edited.get("user") == bot_user_id:
        return "self"
    if body.get("bot_id"):
        return "bot"
    return None


# ── gate 4: mention-gate ──────────────────────────────────────────────


def _bot_thread_participant(
    event: dict[str, Any],
    *,
    bot_user_id: str,
    bot_thread_ts: frozenset[str],
) -> bool:
    """Does this thread already have bot activity? (the implicit-mention bypass)

    True when the inbound's ``thread_ts`` matches a thread the bot has
    posted in (the connector's per-thread sent-``ts`` set) OR the direct
    reply-to-bot case (``parent_user_id == bot_user_id``).  Ties to the
    channel-session model: per-thread *participation*, not a per-thread
    session (§3.6).
    """
    thread_ts = event.get("thread_ts")
    if isinstance(thread_ts, str) and thread_ts in bot_thread_ts:
        return True
    return event.get("parent_user_id") == bot_user_id


def _passes_mention_gate(
    event: dict[str, Any],
    *,
    chat_kind: ChatKind,
    mentions: tuple[str, ...],
    bot_user_id: str,
    bot_thread_ts: frozenset[str],
) -> bool:
    """The mention policy as a pure decision over the ``chat_kind`` kind.

    * ``im`` → always respond.
    * ``mpim`` / ``channel`` / ``group`` → require an explicit
      ``<@bot_user_id>`` mention, **unless** the ``bot_thread_participant``
      implicit-mention bypass fires.
    """
    if chat_kind == "im":
        return True
    if bot_user_id in mentions:
        return True
    return _bot_thread_participant(event, bot_user_id=bot_user_id, bot_thread_ts=bot_thread_ts)


# ── the gate pipeline ─────────────────────────────────────────────────


def gate(
    event: dict[str, Any],
    *,
    bot_user_id: str,
    team_id: str,
    api_app_id: str | None = None,
    allow_bots: bool = False,
    bot_thread_ts: frozenset[str] = frozenset(),
) -> GateDecision:
    """Run the four connector-side gates over one raw Slack ``event``.

    Returns a :class:`GateDecision`.  Only a ``FORWARD`` outcome carries a
    built :class:`InboundMessage`.  Pure: no I/O, no Slack calls.

    Order (§3.6): self/bot-loop → cross-app/team → subtype →
    mention-gate.  ``message_changed`` is diverted to the non-emitting
    system path (``DIVERT_EDIT``) **after** the nested-identity self
    filter, so a bot-authored edit is a ``DROP`` rather than reaching the
    system path.
    """
    subtype = event.get("subtype")

    # ── gate 1: self / bot-loop (reads nested body for message_changed) ──
    self_bot = _is_self_or_bot(event, bot_user_id=bot_user_id)
    if self_bot == "self":
        return GateDecision(GateOutcome.DROP, reason="self")
    if self_bot == "bot" and not allow_bots:
        return GateDecision(GateOutcome.DROP, reason="bot")

    # ── gate 2: cross-app / cross-team ──────────────────────────────────
    event_team = event.get("team") or event.get("user_team")
    if isinstance(event_team, str) and event_team != team_id:
        return GateDecision(GateOutcome.DROP, reason="cross_team")
    event_app = event.get("api_app_id")
    if api_app_id is not None and isinstance(event_app, str) and event_app != api_app_id:
        return GateDecision(GateOutcome.DROP, reason="cross_app")

    # ── message_changed diversion (after the nested self-filter) ────────
    # A bot-authored edit was already dropped above.  Any surviving
    # message_changed is a human edit routed to the non-emitting system
    # path — never into ``emit_inbound``.
    if subtype == "message_changed":
        return GateDecision(GateOutcome.DIVERT_EDIT, reason="message_changed")

    # ── gate 3: subtype filter ──────────────────────────────────────────
    # Plain user messages have no subtype.  Everything else (channel_join,
    # message_deleted, bot_message, file_share-as-subtype, …) is dropped.
    if subtype is not None:
        return GateDecision(GateOutcome.DROP, reason="subtype")

    # ── normalize text + mentions for the mention-gate and the inbound ──
    body = _message_body(event)
    text = body.get("text")
    text = text if isinstance(text, str) else ""
    mentions = extract_mentions(text)
    chat_kind = chat_kind_of(event)

    # ── gate 4: mention-gate ────────────────────────────────────────────
    if not _passes_mention_gate(
        event,
        chat_kind=chat_kind,
        mentions=mentions,
        bot_user_id=bot_user_id,
        bot_thread_ts=bot_thread_ts,
    ):
        # Fail-quiet: record to per-channel pending-history, then drop.
        return GateDecision(
            GateOutcome.DROP,
            reason="mention_required",
            record_pending=True,
            mentions=mentions,
        )

    message = build_inbound(event, chat_kind=chat_kind, mentions=mentions)
    return GateDecision(GateOutcome.FORWARD, message=message, mentions=mentions)


def build_inbound(
    event: dict[str, Any],
    *,
    chat_kind: ChatKind | None = None,
    mentions: tuple[str, ...] | None = None,
) -> InboundMessage:
    """Normalize a (gate-passing) plain message event into an InboundMessage.

    ``message_ts`` is the message's own ``ts``; for an edit the original
    text/ts come from the nested ``event.message`` and ``edit_ts`` is the
    nested ``edited.ts`` (load-bearing for the distinct edit ``event_id``,
    §3.4).  ``thread_ts`` is the bare thread anchor or ``None`` for a
    top-level message.  ``sender_name`` is sanitized in this module
    before it ever reaches ``emit_inbound``.
    """
    body = _message_body(event)
    channel_id = _channel_id_of(event)
    sender_id = body.get("user")
    sender_id = sender_id if isinstance(sender_id, str) else ""
    message_ts = body.get("ts")
    message_ts = message_ts if isinstance(message_ts, str) else ""

    edited_block = body.get("edited")
    edited = isinstance(edited_block, dict)
    edit_ts = edited_block.get("ts") if isinstance(edited_block, dict) else None
    edit_ts = edit_ts if isinstance(edit_ts, str) else None

    thread_ts = event.get("thread_ts") or body.get("thread_ts")
    thread_ts = thread_ts if isinstance(thread_ts, str) else None
    # A message whose thread_ts equals its own ts is a thread *root*, not
    # a reply — surface None so it reads as top-level (the model only
    # threads when replying into an existing thread).
    if thread_ts is not None and thread_ts == message_ts:
        thread_ts = None

    text = body.get("text")
    text = text if isinstance(text, str) else ""

    if chat_kind is None:
        chat_kind = chat_kind_of(event)
    if mentions is None:
        mentions = extract_mentions(text)

    # display_name lives on the user-profile lookup the connector performs;
    # parse only carries the opaque id when no name is present.  Whatever
    # the caller provides via ``event['_display_name']`` (an enrichment the
    # connector stamps) is sanitized here.
    raw_name = event.get("_display_name")
    raw_name = raw_name if isinstance(raw_name, str) else None
    sender_name = sanitize_display_name(raw_name, fallback=sender_id)

    return InboundMessage(
        chat_kind=chat_kind,
        channel_id=channel_id,
        sender_id=sender_id,
        sender_name=sender_name,
        message_ts=message_ts,
        thread_ts=thread_ts,
        text=text,
        edited=edited,
        edit_ts=edit_ts,
        mentions=mentions,
    )
