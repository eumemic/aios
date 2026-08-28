"""The worker ↔ browser-driver wire contract (jarbot#106).

ONE module, TWO consumers: the worker (tool handlers + the control-plane
executor) imports it in-process, and the browser image ``COPY``s this exact
file so the driver implements the same contract — worker and driver can never
fork. That sharing imposes the module's hard rule, enforced by a unit test:

    **stdlib + pydantic only — no ``aios`` imports.**

Invocation
----------
The worker execs, inside the account's browser container::

    browser-cli invoke '<request JSON, shlex-quoted>'

stdout is exactly one JSON response document.  **Exit-code contract:
``browser-cli`` exits 0 iff it produced a response document — INCLUDING
``ok: false`` responses.** A nonzero exit or unparseable stdout is a
transport/daemon failure (driver not running, binary absent — exit 127 on a
pre-driver deployment), which the worker surfaces as "browser unavailable"
rather than an action error.

Driver obligations (implemented by the browser image, pinned here)
------------------------------------------------------------------
* The ``epoch`` is a BARRIER advanced on EVERY takeover transition — BOTH
  ``takeover_open`` AND ``takeover_close`` rotate it.
    - ``takeover_open`` blocks through the drain of any in-flight agent action,
      then rotates the epoch; it is **idempotent per** ``grant_id``
      (re-invocation returns the standing takeover) — this is what makes the
      worker's lost-NOTIFY redrive safe after a restart.
    - ``takeover_close`` rotates it again as it hands control back. This is
      load-bearing for the close-with-no-reopen case: a human input line that
      raced the close (passed the API's open+epoch check just before close, got
      appended just after) carries the pre-close epoch, which is no longer
      current — so the driver drops it and it can NEVER land in the
      agent-controlled browser. Without the close-side rotation the epoch would
      still match and the stale line would be applied during agent control.
* A takeover self-closes after :data:`DRIVER_TAKEOVER_IDLE_TIMEOUT_S` of
  liveness silence.  The driver cannot see frame consumption directly, so
  liveness = ``max(opened_at, last_input_line, mtime(TAKEOVER_HEARTBEAT_MARKER))``:
  the API's heartbeat route touches :data:`TAKEOVER_HEARTBEAT_MARKER`, so a human
  who is *watching* (not typing) still counts as live.  A takeover that has
  received ZERO input lines AND ZERO marker touches since it opened self-closes
  far sooner, after :data:`DRIVER_TAKEOVER_UNCLAIMED_TIMEOUT_S` — the backstop
  for an ack whose grant row was never written (the exec reply was lost).
* Input-spool lines whose ``epoch`` is not current are dropped — the driver
  is the enforcement authority (the API's open+epoch check is convenience, and
  the barrier above is what makes that authority sufficient across a handback).
  A viewer's ``seq`` is strictly increasing per grant and MUST resume (not
  restart) across a reconnect; the driver's per-grant de-dup resets on every
  ``takeover_open``.
* Driver processes run as uid:gid :data:`PLANE_OWNER_UID` so plane files
  (frames, shots) are readable and the input spool writable by the API.
* The frames manifest carries the trusted-chrome fields per frame
  (jarbot#106 §5.6): ``origin``/``security`` are derived from the CDP
  Security domain and the committed main-frame URL — never parsed from
  pixels.  ``manifest["file"]`` is a frames-dir-relative BASENAME
  (``"frame-<seq>.jpg"``); the API resolves it under ``frames/`` and a
  plane-relative value would never resolve.
* Every action response's ``snapshot`` respects
  :data:`SNAPSHOT_MAX_ELEMENTS` / :data:`SNAPSHOT_MAX_CHARS` with
  ``snapshot_truncated`` set when clipped.  The snapshot is OPAQUE TEXT to
  the worker — its rendering may change freely without an aios change.
* A response emitted while a takeover holds the browser (``takeover_active``
  refusals and the idempotent ``takeover_open`` echo) is PAGE-BLIND:
  ``snapshot``/``url``/``title`` are null and ``tabs`` empty.  The worker
  renders whatever ``snapshot`` arrives on ``ok: false``, so a snapshot here
  would leak the human's live (login/OTP) page into agent context — the
  reverse of the boundary the epoch barrier protects.  ``status`` is the one
  control op the gate does NOT block (the product polls it during takeovers),
  and it too stays page-blind beyond ``url``/``title`` of the current page.
* ``takeover_close.args.outcome`` is an OPAQUE string the driver records and
  never validates (values today: ``done`` | ``cancelled`` | ``expired`` — the
  reaper mints ``expired``).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

PROTOCOL_VERSION = 1

# ── budgets and deadlines ─────────────────────────────────────────────────
# The a11y snapshot budget (≈6k tokens at 4 chars/token).
SNAPSHOT_MAX_ELEMENTS = 120
SNAPSHOT_MAX_CHARS = 24_000
# Idle backstop: the driver self-closes a takeover after this long of liveness
# silence, where liveness folds in the heartbeat marker (see below) so a
# watching-but-not-typing human still counts as present.
DRIVER_TAKEOVER_IDLE_TIMEOUT_S = 900
# Unclaimed backstop: a takeover that has received ZERO input lines AND ZERO
# heartbeat-marker touches since it opened self-closes this fast — the case
# where the ``takeover_open`` ack was produced but its exec reply was lost, so
# the grant row was never written and no viewer can ever attach.
DRIVER_TAKEOVER_UNCLAIMED_TIMEOUT_S = 60

# ── the shared plane (host bind mount at /workspace in the container) ─────
# The uid:gid every driver process runs as, so the API process (same id via
# the deployment's workspaces_owner settings) can read frames/shots and
# append to the input spool.
PLANE_OWNER_UID = 1000
PROFILE_DIR = "/workspace/profile"
SHOTS_DIR = "/workspace/shots"
FRAMES_DIR = "/workspace/frames"
DOWNLOADS_DIR = "/workspace/downloads"
INPUT_SPOOL = "/workspace/input/spool.jsonl"
# The API's takeover-heartbeat route touches this marker (``os.utime``); the
# driver folds its mtime into the takeover liveness clock (plane-relative,
# like FRAMES_MANIFEST — each side joins it to its own plane root).
TAKEOVER_HEARTBEAT_MARKER = "input/.heartbeat"
# The driver rewrites this manifest for every emitted frame; the API tails it.
# ``manifest["file"]`` is a frames-dir-relative basename ("frame-<seq>.jpg").
FRAMES_MANIFEST = "frames/manifest.json"

# The action ops (one per browser_* tool) and the control ops the worker's
# control-plane executor invokes. ``clear_state`` is deliberately NOT a
# driver op: the worker implements it as release-container + recreate the
# plane subdirs.
BrowserOp = Literal[
    "snapshot",
    "navigate",
    "click",
    "click_xy",
    "type",
    "press_key",
    "scroll",
    "drag",
    "hover",
    "select_option",
    "screenshot",
    "takeover_open",
    "takeover_close",
    "status",
    "revoke_site",
]

# Driver error codes. The worker maps every code to a model-visible tool
# error and must TOLERATE unknown codes (forward compatibility): render the
# code + message rather than failing the parse.
ERROR_CODES = frozenset(
    {
        "invalid_request",
        "unknown_op",
        "no_such_ref",
        "stale_snapshot",
        "not_interactable",
        "action_timeout",
        "navigation_failed",
        "takeover_active",
        "no_takeover",
        "grant_mismatch",
        "browser_crashed",
        "internal",
    }
)

# Input-spool event vocabulary (jarbot#106 §5.6) — one JSONL line per batch:
#   {"grant_id": "...", "epoch": 7, "seq": 12, "ts_ms": ...,
#    "events": [{"type": "pointer_move", "x": 412, "y": 230}, ...]}
# Deliberately lower-level than the browser_* arms: raw pointer/key/text
# events the human's viewer emits.
INPUT_EVENT_TYPES = frozenset(
    {
        "pointer_move",
        "pointer_down",
        "pointer_up",
        "wheel",
        "key_down",
        "key_up",
        "text",
    }
)


class BrowserError(BaseModel):
    """A driver-reported action failure (the ``ok: false`` payload)."""

    code: str
    message: str


class BrowserTab(BaseModel):
    """One open tab, informational only — no page id is ever a model input."""

    index: int
    url: str
    title: str
    active: bool = False


class BrowserRequest(BaseModel):
    """One ``browser-cli invoke`` request document."""

    v: int = PROTOCOL_VERSION
    op: BrowserOp
    # The calling agent session for action ops (the driver keys its page
    # registry on it — server-authored, never a model argument); ``None`` for
    # control ops, which act on the whole browser — EXCEPT ``takeover_open``,
    # which carries the requesting session's id here so the driver takes over
    # that session's page (jarbot#106 §5.6 page-scoped grants).
    session_id: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    # Driver-side soft deadline; always below the in-container ``timeout``
    # wrapper the exec applies.
    timeout_ms: int = 30_000


class BrowserResponse(BaseModel):
    """One response document (stdout of ``browser-cli invoke``)."""

    v: int = PROTOCOL_VERSION
    ok: bool
    # Driver-minted boot ULID identifying the current Chromium context; it
    # changes on every browser (re)launch — a driver restart OR an in-container
    # Chromium relaunch, both of which lose page state. The anchor for
    # grant/epoch validity and the model-visible "the computer restarted"
    # signal (a mid-takeover relaunch rotates it, which is what ends the frames
    # stream and tells the model its page is gone).
    boot: str
    # The takeover barrier counter (jarbot#106 §5.6).
    epoch: int
    url: str | None = None
    title: str | None = None
    tabs: list[BrowserTab] = Field(default_factory=list)
    # The budgeted a11y snapshot with [ref=eN] handles — opaque text.
    snapshot: str | None = None
    snapshot_truncated: bool = False
    duration_ms: int = 0
    # Plane-relative path (e.g. "shots/....png") from ``screenshot``; the
    # worker resolves it against the plane dir.
    shot_path: str | None = None
    # True when the calling session's page had to be recreated because the
    # driver (re)booted since the page last existed — the per-call
    # surfacing of "the computer restarted; page state was lost".
    driver_restarted: bool = False
    # Op-specific payload (takeover_open target, status body, signed-in
    # hosts on takeover_close, ...).
    data: dict[str, Any] = Field(default_factory=dict)
    error: BrowserError | None = None
