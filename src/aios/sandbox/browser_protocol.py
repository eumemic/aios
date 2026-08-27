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
* ``takeover_open`` blocks through the drain of any in-flight agent action,
  then rotates the epoch; it is **idempotent per** ``grant_id`` (re-invocation
  returns the standing takeover) — this is what makes the worker's
  lost-NOTIFY redrive safe after a restart.
* A takeover with no input lines and no frame consumption for
  :data:`DRIVER_TAKEOVER_IDLE_TIMEOUT_S` auto-closes (orphan backstop for a
  crash between driver ack and the grant record).
* Input-spool lines whose ``epoch`` is not current are dropped — the driver
  is the enforcement authority; any upstream check is convenience.
* Driver processes run as uid:gid :data:`PLANE_OWNER_UID` so plane files
  (frames, shots) are readable and the input spool writable by the API.
* The frames manifest carries the trusted-chrome fields per frame
  (jarbot#106 §5.6): ``origin``/``security`` are derived from the CDP
  Security domain and the committed main-frame URL — never parsed from
  pixels.
* Every action response's ``snapshot`` respects
  :data:`SNAPSHOT_MAX_ELEMENTS` / :data:`SNAPSHOT_MAX_CHARS` with
  ``snapshot_truncated`` set when clipped.  The snapshot is OPAQUE TEXT to
  the worker — its rendering may change freely without an aios change.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

PROTOCOL_VERSION = 1

# ── budgets and deadlines ─────────────────────────────────────────────────
# The a11y snapshot budget (≈6k tokens at 4 chars/token).
SNAPSHOT_MAX_ELEMENTS = 120
SNAPSHOT_MAX_CHARS = 24_000
# Orphaned-takeover backstop: the driver self-closes a takeover after this
# long with no input lines and no frame consumption.
DRIVER_TAKEOVER_IDLE_TIMEOUT_S = 900

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
# The driver rewrites this manifest for every emitted frame; the API tails it.
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
    # registry on it — server-authored, never a model argument); ``None``
    # for control ops, which act on the whole browser.
    session_id: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    # Driver-side soft deadline; always below the in-container ``timeout``
    # wrapper the exec applies.
    timeout_ms: int = 30_000


class BrowserResponse(BaseModel):
    """One response document (stdout of ``browser-cli invoke``)."""

    v: int = PROTOCOL_VERSION
    ok: bool
    # Driver-minted boot ULID; changes on every driver restart. The anchor
    # for grant/epoch validity and the model-visible "the computer
    # restarted" signal.
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
    # Plane-relative path (e.g. "shots/....png") for ``screenshot`` and
    # on-error captures; the worker resolves it against the plane dir.
    shot_path: str | None = None
    # True when the calling session's page had to be recreated because the
    # driver (re)booted since the page last existed — the per-call
    # surfacing of "the computer restarted; page state was lost".
    driver_restarted: bool = False
    # Op-specific payload (takeover_open target, status body, signed-in
    # hosts on takeover_close, ...).
    data: dict[str, Any] = Field(default_factory=dict)
    error: BrowserError | None = None
