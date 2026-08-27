"""The eleven ``browser_*`` builtins — the model's hands on the account computer.

Every account has (at most) one shared browser ("the computer"), running in
its own container, holding one page per agent session. These tools are the
ONLY route an agent has to it: they run worker-side (``transport="agent_tool"``
— the in-sandbox CLI cannot invoke them), resolve the account and page from
the calling ``session_id`` server-side (no page id is ever a model argument),
and exec the driver through :func:`aios.sandbox.browser.driver_call`.

Perception is text-first: every action returns a budgeted accessibility
snapshot with ``[ref=eN]`` handles — the fast path for targeting — while the
pointer arms also take viewport coordinates for surfaces the accessibility
tree cannot express (canvas, sliders, maps, drag-and-drop). Screenshots are
explicit (``browser_screenshot``) or attached by the driver on errors, never
per-action.

Failure currency: driver action failures and transport faults surface as
``ToolBail`` — model-visible, self-correctable, and NEVER the calling
session's sandbox being evicted (a browser-container fault is not the
caller's sandbox being unhealthy).
"""

from __future__ import annotations

import base64
import os
from typing import Any

from aios.config import get_settings
from aios.harness import runtime
from aios.harness.image_resize import maybe_downsample
from aios.harness.vision import (
    INLINE_SIZE_CAP_BYTES,
    PROVIDER_INLINE_IMAGE_FORMATS,
    human_size,
    inline_image_format,
    make_image_url_part,
    supports_vision,
)
from aios.sandbox.browser import EXEC_KILL_MARGIN_S, BrowserUnavailableError, driver_call
from aios.sandbox.browser_protocol import BrowserRequest, BrowserResponse
from aios.sandbox.spec import BrowserImageUnconfiguredError
from aios.services import agents as agents_service
from aios.services import sessions as sessions_service
from aios.tools.invoke import ToolBail
from aios.tools.registry import ToolHandler, ToolResult, registry

# ── shared schema fragments ───────────────────────────────────────────────

# Chorded pointer input: the modifiers held while the pointer acts. Together
# with key chords in browser_press_key this covers chorded input without raw
# key_down/key_up arms.
_MODIFIERS = {
    "type": "array",
    "items": {"type": "string", "enum": ["Alt", "Control", "Meta", "Shift"]},
    "description": "Modifier keys held during the action.",
}
# Required on every state-changing arm: a short statement of intent. It is
# recorded with the action in the session log, so the owner can audit what
# the agent did on their computer.
_DESCRIPTION = {
    "type": "string",
    "maxLength": 500,
    "description": "What this action does, briefly (e.g. 'Submit the checkout form'). "
    "Recorded in the activity trail.",
}
_POINT = {
    "type": "object",
    "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
    "required": ["x", "y"],
    "additionalProperties": False,
}
# A drag endpoint: a snapshot ref OR viewport coordinates.
_ENDPOINT = {
    "type": "object",
    "properties": {
        "ref": {"type": "string"},
        "x": {"type": "number"},
        "y": {"type": "number"},
    },
    "additionalProperties": False,
}

_SNAPSHOT_TAIL = (
    " Returns a fresh accessibility snapshot of the page: interactive elements "
    "and headings with [ref=eN] handles for targeting follow-up actions."
)

# ── the eleven arms: (name, description, parameters schema) ───────────────

_ARMS: list[tuple[str, str, dict[str, Any]]] = [
    (
        "browser_snapshot",
        "Re-observe the computer's current page without acting on it."
        + _SNAPSHOT_TAIL
        + " Use it when refs may be stale or after the page changed on its own.",
        {"type": "object", "properties": {}, "additionalProperties": False},
    ),
    (
        "browser_navigate",
        "Navigate the computer's page to a public http(s) URL." + _SNAPSHOT_TAIL,
        {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Absolute http:// or https:// URL."}
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    ),
    (
        "browser_click",
        "Click an element by its [ref=eN] handle from the latest snapshot." + _SNAPSHOT_TAIL,
        {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "Element handle, e.g. 'e12'."},
                "modifiers": _MODIFIERS,
                "description": _DESCRIPTION,
            },
            "required": ["ref", "description"],
            "additionalProperties": False,
        },
    ),
    (
        "browser_click_xy",
        "Click a viewport coordinate — for surfaces the accessibility tree cannot "
        "express (canvas controls, maps, visual editors). Take a browser_screenshot "
        "first to pick the point; prefer browser_click with a ref when one exists."
        + _SNAPSHOT_TAIL,
        {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "button": {"type": "string", "enum": ["left", "middle", "right"]},
                "count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3,
                    "description": "Click count (2 = double-click).",
                },
                "modifiers": _MODIFIERS,
                "description": _DESCRIPTION,
            },
            "required": ["x", "y", "description"],
            "additionalProperties": False,
        },
    ),
    (
        "browser_type",
        "Type text into an element (focuses it first)." + _SNAPSHOT_TAIL,
        {
            "type": "object",
            "properties": {
                "ref": {"type": "string"},
                "text": {"type": "string", "maxLength": 2000},
                "submit": {
                    "type": "boolean",
                    "description": "Press Enter after typing.",
                },
                "description": _DESCRIPTION,
            },
            "required": ["ref", "text", "description"],
            "additionalProperties": False,
        },
    ),
    (
        "browser_press_key",
        "Press a key or chord on the focused element (e.g. 'Enter', 'Tab', "
        "'Control+a', 'Shift+Tab')." + _SNAPSHOT_TAIL,
        {
            "type": "object",
            "properties": {"key": {"type": "string", "maxLength": 64}},
            "required": ["key"],
            "additionalProperties": False,
        },
    ),
    (
        "browser_scroll",
        "Scroll the page, an element (by ref), or wheel-at-a-point (for map/canvas "
        "zoom and pan)." + _SNAPSHOT_TAIL,
        {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
                "amount": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Scroll steps; omit for one page.",
                },
                "ref": {"type": "string", "description": "Scrollable container handle."},
                "at": {
                    **_POINT,
                    "description": "Viewport point to wheel at (map/canvas zoom-pan).",
                },
            },
            "required": ["direction"],
            "additionalProperties": False,
        },
    ),
    (
        "browser_drag",
        "Drag from one point to another — pointer down, interpolated move (through "
        "optional waypoints), pointer up. Endpoints are refs or viewport "
        "coordinates; use coordinates for sliders, reordering, drawing." + _SNAPSHOT_TAIL,
        {
            "type": "object",
            "properties": {
                "from": _ENDPOINT,
                "to": _ENDPOINT,
                "path": {
                    "type": "array",
                    "items": _POINT,
                    "maxItems": 20,
                    "description": "Intermediate waypoints.",
                },
                "modifiers": _MODIFIERS,
                "description": _DESCRIPTION,
            },
            "required": ["from", "to", "description"],
            "additionalProperties": False,
        },
    ),
    (
        "browser_hover",
        "Hover an element (by ref) or a viewport point — hover menus, tooltips, "
        "hover-revealed controls." + _SNAPSHOT_TAIL,
        {
            "type": "object",
            "properties": {
                "ref": {"type": "string"},
                "x": {"type": "number"},
                "y": {"type": "number"},
                "description": _DESCRIPTION,
            },
            "required": ["description"],
            "additionalProperties": False,
        },
    ),
    (
        "browser_select_option",
        "Select option(s) in a native <select> element — its popup is browser "
        "chrome no pointer event reaches." + _SNAPSHOT_TAIL,
        {
            "type": "object",
            "properties": {
                "ref": {"type": "string"},
                "values": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "description": _DESCRIPTION,
            },
            "required": ["ref", "values", "description"],
            "additionalProperties": False,
        },
    ),
    (
        "browser_screenshot",
        "Capture the page as an image — for visual layouts the snapshot cannot "
        "express, and to pick coordinates for browser_click_xy/browser_drag. "
        "Prefer snapshots for reading and targeting; screenshots cost far more.",
        {
            "type": "object",
            "properties": {
                "full_page": {
                    "type": "boolean",
                    "description": "Capture the full page instead of the viewport.",
                }
            },
            "additionalProperties": False,
        },
    ),
]

_UNAVAILABLE_MESSAGE = (
    "Browser tools are not available in this deployment: no browser image is "
    "configured. This capability is not yet enabled."
)
_UNREACHABLE_MESSAGE = (
    "The computer is unavailable right now (its browser failed to start or is "
    "not responding). Try again shortly."
)


async def _check_arm_granted(session_id: str, account_id: str, tool_name: str) -> None:
    """Defense-in-depth grant re-check (jarbot#106 §6.3).

    The step-frozen surface enforcement already refuses an unoffered builtin
    before dispatch; this second, independent deny re-reads the session's
    CURRENT effective tool surface so a grant revoked mid-step (or a
    deployment predating the surface enforcement) still denies at the
    handler. Two independent denies, either sufficient.
    """
    pool = runtime.require_pool()
    session = await sessions_service.get_session_basic(pool, session_id, account_id=account_id)
    surface = await agents_service.load_for_session(pool, session, account_id=account_id)
    for spec in surface.tools:
        if spec.type == tool_name and spec.enabled:
            return
    raise ToolBail(f"{tool_name} is not enabled for this agent")


def _browser_metadata(tool_name: str, response: BrowserResponse) -> dict[str, Any]:
    """The ``metadata.browser`` block — the product-layer wire contract."""
    return {
        "browser": {
            "action": tool_name.removeprefix("browser_"),
            "url": response.url,
            "title": response.title,
            "duration_ms": response.duration_ms,
            "tabs": len(response.tabs),
            "epoch": response.epoch,
            "shared_profile": True,
            "driver_restarted": response.driver_restarted,
            "screenshot": response.shot_path is not None,
        }
    }


def _render_result(tool_name: str, response: BrowserResponse) -> ToolResult:
    """Assemble the standard action result: header + snapshot + metadata."""
    lines = []
    if response.url:
        title = f" — {response.title!r}" if response.title else ""
        lines.append(f"{response.url}{title}")
    if response.driver_restarted:
        lines.append(
            "[The computer's browser restarted since this page was last used; "
            "page state was lost. Navigate again if needed.]"
        )
    if response.snapshot:
        truncated = (
            " (truncated — narrow with browser_snapshot)" if response.snapshot_truncated else ""
        )
        lines.append(f"\n## Page{truncated}\n{response.snapshot}")
    content = "\n".join(lines) if lines else "ok"
    return ToolResult(content=content, metadata=_browser_metadata(tool_name, response))


async def _invoke_driver(
    tool_name: str, session_id: str, arguments: dict[str, Any]
) -> tuple[BrowserResponse, str]:
    """Shared handler core: grant re-check → driver call → ok/error split.

    Returns ``(response, account_id)`` on ``ok: true``; raises ``ToolBail``
    for every driver-level failure and transport fault (never a bare
    exception — an unrecognized exception from a tool handler evicts the
    CALLING session's sandbox, and a browser fault is not the caller's
    sandbox being unhealthy).
    """
    settings = get_settings()
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    await _check_arm_granted(session_id, account_id, tool_name)

    op = tool_name.removeprefix("browser_")
    request = BrowserRequest(
        op=op,
        session_id=session_id,
        args=arguments,
        timeout_ms=settings.sandbox_browser_action_timeout_seconds * 1000,
    )
    try:
        response = await driver_call(
            runtime.require_sandbox_registry(),
            account_id,
            request,
            timeout_s=settings.sandbox_browser_action_timeout_seconds + EXEC_KILL_MARGIN_S,
        )
    except BrowserImageUnconfiguredError as err:
        raise ToolBail(_UNAVAILABLE_MESSAGE) from err
    except BrowserUnavailableError as err:
        raise ToolBail(_UNREACHABLE_MESSAGE, detail={"cause": str(err)}) from err

    if not response.ok:
        code = response.error.code if response.error else "internal"
        message = response.error.message if response.error else "unknown driver failure"
        text = f"{tool_name} failed: {code}: {message}"
        if response.snapshot:
            # The driver attaches a fresh snapshot on failure so the model can
            # self-correct (stale ref → re-target) without a re-observe call.
            text += f"\n\n## Page\n{response.snapshot}"
        raise ToolBail(text, detail={"code": code})
    return response, account_id


def _make_action_handler(tool_name: str) -> ToolHandler:
    async def handler(session_id: str, arguments: dict[str, Any]) -> ToolResult:
        response, _ = await _invoke_driver(tool_name, session_id, arguments)
        return _render_result(tool_name, response)

    handler.__name__ = f"{tool_name}_handler"
    return handler


async def browser_screenshot_handler(session_id: str, arguments: dict[str, Any]) -> ToolResult:
    """Capture pixels: driver writes the shot to the plane; we inline it.

    Reuses the ``read`` tool's image ladder verbatim: vision gate → format
    gate → downsample → data-URI part. A non-vision model gets a text
    explanation instead of pixels, exactly like ``read``.
    """
    from aios.sandbox.volumes import browser_plane_dir

    response, account_id = await _invoke_driver("browser_screenshot", session_id, arguments)
    if not response.shot_path:
        raise ToolBail("browser_screenshot failed: the driver returned no image")

    plane = browser_plane_dir(account_id)
    shot = (plane / response.shot_path).resolve()
    if not shot.is_relative_to(plane):
        raise ToolBail(
            "browser_screenshot failed: the driver returned an invalid image path",
            detail={"shot_path": response.shot_path},
        )
    try:
        data = shot.read_bytes()
    except OSError as err:
        raise ToolBail(f"browser_screenshot failed: could not read the image: {err}") from err

    pool = runtime.require_pool()
    model = await sessions_service.get_session_model(pool, session_id, account_id=account_id)
    metadata = _browser_metadata("browser_screenshot", response)
    header = f"Screenshot: {os.path.basename(response.shot_path)}"

    vision_support = supports_vision(model)
    if len(data) > INLINE_SIZE_CAP_BYTES or vision_support is not True:
        if len(data) > INLINE_SIZE_CAP_BYTES:
            reason = "the image exceeds the inline size cap"
        elif vision_support is None:
            # Three-state truthfulness (mirrors read): unknown is not "no".
            reason = f"image support for model {model!r} is unknown"
        else:
            reason = f"model {model!r} does not support image input"
        return ToolResult(
            content=f"{header} captured ({human_size(len(data))}), but {reason}.",
            metadata=metadata,
        )
    image_format = inline_image_format(data)
    if image_format is None or image_format not in PROVIDER_INLINE_IMAGE_FORMATS:
        reason = (
            "its bytes could not be decoded as an image"
            if image_format is None
            else f"providers do not accept {image_format} inline"
        )
        return ToolResult(
            content=f"{header} captured, but {reason}.",
            metadata=metadata,
        )
    mime = f"image/{image_format.lower()}"
    try:
        resized = await maybe_downsample(data, mime)
    except Exception as err:
        # A downsample/re-encode fault is a WORKER-side fault, not the calling
        # session's sandbox being unhealthy — the browser eviction contract is
        # stricter than read's here, so a residual raw Pillow error must NOT
        # escape and evict. The action already succeeded (the shot was
        # captured); degrade to text rather than bailing. (ImageDownsampleError
        # is the expected member; the broaden covers encode edge cases.)
        return ToolResult(
            content=f"{header} captured, but could not be processed: {err}",
            metadata=metadata,
        )
    if resized is not None:
        data, mime = resized.data, resized.content_type

    parts: list[dict[str, Any]] = [
        {"type": "text", "text": f"{header} ({mime}, {human_size(len(data))})"},
        make_image_url_part(content_type=mime, data_b64=base64.b64encode(data).decode("ascii")),
    ]
    return ToolResult(content=parts, metadata=metadata)


def _register() -> None:
    for name, description, schema in _ARMS:
        registry.register(
            name=name,
            description=description,
            parameters_schema=schema,
            handler=(
                browser_screenshot_handler
                if name == "browser_screenshot"
                else _make_action_handler(name)
            ),
            transport="agent_tool",
        )


_register()
