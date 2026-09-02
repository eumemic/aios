"""API models for the browser takeover surface (jarbot#106 §5.7).

The product layer's view of a takeover: open/heartbeat/input/close on one
grant, plus the account-computer status/revocation operations. Input events
use the wire vocabulary pinned in
:mod:`aios.sandbox.browser_protocol` (``INPUT_EVENT_TYPES``) — raw
pointer/key/text events, deliberately lower-level than the ``browser_*``
tool arms, since they carry a human's live driving.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TakeoverOpenRequest(BaseModel):
    """Open a takeover of the account's computer for one agent session's page."""

    session_id: str
    reason: str = Field(default="", max_length=200)


class TakeoverOpenResponse(BaseModel):
    """The opened grant. The viewer PINS ``target``/``boot``/``epoch`` and
    refuses frames or input that do not match (the trusted-chrome binding,
    jarbot#106 §5.6)."""

    grant_id: str
    target: dict[str, Any]
    boot: str
    epoch: int
    ttl_seconds: int


class TakeoverCloseRequest(BaseModel):
    outcome: Literal["done", "cancelled"] = "done"


class HandbackPayload(BaseModel):
    """What the human left behind: the post-takeover page snapshot, an inline
    screenshot, and which sites are now signed in (the cookie-jar-derived
    delta). ``None`` fields mean the browser died before the handback could
    be captured — the grant still closed."""

    snapshot: str | None = None
    screenshot_data_url: str | None = None
    signed_in_hosts: list[str] = Field(default_factory=list)
    url: str | None = None


class TakeoverCloseResponse(BaseModel):
    handback: HandbackPayload


class InputEvent(BaseModel):
    """One raw viewer input event (§5.6 vocabulary).

    The nav quartet (``navigate``/``back``/``forward``/``reload``) is the
    viewer's browser chrome — URL bar and nav buttons — riding the same spool
    as raw input so a typed URL stays off the event log exactly like a typed
    password. The driver guards ``navigate`` (public http(s) only) and treats
    the rest as history moves needing no URL.
    """

    type: Literal[
        "pointer_move",
        "pointer_down",
        "pointer_up",
        "wheel",
        "key_down",
        "key_up",
        "text",
        "navigate",
        "back",
        "forward",
        "reload",
    ]
    x: float | None = None
    y: float | None = None
    button: Literal["left", "middle", "right"] | None = None
    dx: float | None = None
    dy: float | None = None
    key: str | None = Field(default=None, max_length=64)
    text: str | None = Field(default=None, max_length=2000)
    url: str | None = Field(default=None, max_length=2048)


class InputBatch(BaseModel):
    """One coalesced batch of input events, epoch-stamped.

    The API pre-checks the epoch against the grant record; the DRIVER is the
    enforcement authority and drops stale-epoch spool lines regardless.
    """

    epoch: int
    seq: int
    events: list[InputEvent] = Field(min_length=1, max_length=200)


class BrowserTakeoverStatus(BaseModel):
    """The open grant on this account's computer, if any."""

    grant_id: str
    session_id: str
    reason: str
    epoch: int
    boot: str
    created_at: str


class BrowserStatusResponse(BaseModel):
    """The account computer's state: not running, or running with its page
    and any open takeover + signed-in sites the driver reports."""

    running: bool
    url: str | None = None
    title: str | None = None
    signed_in_hosts: list[str] = Field(default_factory=list)
    takeover: BrowserTakeoverStatus | None = None


class BrowserPeekPage(BaseModel):
    """One JPEG of a page's viewport plus its trusted chrome. ``origin`` and
    ``security`` come from the driver's committed URL, never the pixels."""

    jpeg_b64: str
    w: int
    h: int
    origin: str | None = None
    security: Literal["secure", "insecure"] | None = None


class BrowserPeekResponse(BaseModel):
    """A read-only look at the computer: not running, running with no page
    to show, or running with the page. Never provisions, never creates a
    page, and is refused while a human holds the computer."""

    running: bool
    url: str | None = None
    title: str | None = None
    page: BrowserPeekPage | None = None
