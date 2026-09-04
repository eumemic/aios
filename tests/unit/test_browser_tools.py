"""Unit coverage for the ``browser_*`` builtins (jarbot#106 Phase 1).

Handlers are exercised directly (never through dispatch), with the
``driver_call`` seam patched — the drivers themselves are a later phase, so
these tests pin the aios-side contract: rendering, failure currency
(everything → ``ToolBail``, never a bare exception that would evict the
CALLER's sandbox), the screenshot image ladder, wiring, and two structural
invariants (protocol purity, no session-sandbox resolution).
"""

from __future__ import annotations

import ast
import base64
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aios.config import get_settings
from aios.harness import vision
from aios.sandbox.browser import BrowserUnavailableError
from aios.sandbox.browser_protocol import BrowserError, BrowserResponse
from aios.sandbox.spec import BrowserImageUnconfiguredError
from aios.services import agents as agents_service
from aios.services import sessions as sessions_service
from aios.tools import browser as browser_mod
from aios.tools.invoke import ToolBail
from aios.tools.registry import ToolResult, registry
from tests.helpers.images import valid_png_bytes

_SESSION_ID = "sess_01BROWSERTEST00000000000000"
_ACCOUNT_ID = "acc_01BROWSERTEST000000000000000"

_BROWSER_TOOLS = [
    "browser_snapshot",
    "browser_navigate",
    "browser_click",
    "browser_click_xy",
    "browser_type",
    "browser_press_key",
    "browser_scroll",
    "browser_drag",
    "browser_hover",
    "browser_select_option",
    "browser_screenshot",
]


def _response(**overrides: Any) -> BrowserResponse:
    base: dict[str, Any] = {
        "ok": True,
        "boot": "01BOOT",
        "epoch": 4,
        "url": "https://example.com/checkout",
        "title": "Checkout",
        "snapshot": '- button "Place order" [ref=e12]',
        "duration_ms": 812,
    }
    base.update(overrides)
    return BrowserResponse.model_validate(base)


@pytest.fixture
def seams(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the handler's collaborators; return the mocks for assertions."""
    driver = AsyncMock(return_value=_response())
    granted = AsyncMock()
    monkeypatch.setattr(browser_mod, "driver_call", driver)
    monkeypatch.setattr(browser_mod, "_check_arm_granted", granted)
    monkeypatch.setattr(
        sessions_service,
        "load_session_account_id",
        AsyncMock(return_value=_ACCOUNT_ID),
    )
    fake_runtime = MagicMock()
    fake_runtime.require_pool.return_value = MagicMock()
    fake_runtime.require_sandbox_registry.return_value = MagicMock()
    monkeypatch.setattr(browser_mod, "runtime", fake_runtime)
    return {"driver": driver, "granted": granted}


class TestActionHandlers:
    async def test_success_renders_header_snapshot_and_metadata(
        self, seams: dict[str, Any]
    ) -> None:
        handler = registry.get("browser_click").handler
        result = await handler(_SESSION_ID, {"ref": "e12", "description": "Place the order"})
        assert isinstance(result, ToolResult)
        assert isinstance(result.content, str)
        assert "https://example.com/checkout" in result.content
        assert "[ref=e12]" in result.content
        assert result.metadata is not None
        browser_meta = result.metadata["browser"]
        assert browser_meta["action"] == "click"
        assert browser_meta["epoch"] == 4
        assert browser_meta["shared_profile"] is True
        assert browser_meta["screenshot"] is False
        # The driver request carried the op + the calling session, and the
        # arguments verbatim (page identity is server-side, never a model arg).
        request = seams["driver"].await_args.args[2]
        assert request.op == "click"
        assert request.session_id == _SESSION_ID
        assert request.args == {"ref": "e12", "description": "Place the order"}

    async def test_driver_restart_is_surfaced_in_content(self, seams: dict[str, Any]) -> None:
        seams["driver"].return_value = _response(driver_restarted=True)
        handler = registry.get("browser_snapshot").handler
        result = await handler(_SESSION_ID, {})
        assert isinstance(result, ToolResult)
        assert isinstance(result.content, str)
        assert "restarted" in result.content
        assert result.metadata is not None
        assert result.metadata["browser"]["driver_restarted"] is True

    async def test_driver_error_raises_toolbail_with_fresh_snapshot(
        self, seams: dict[str, Any]
    ) -> None:
        seams["driver"].return_value = _response(
            ok=False,
            error=BrowserError(code="stale_snapshot", message="ref e12 is gone"),
            snapshot='- link "Home" [ref=e1]',
        )
        handler = registry.get("browser_click").handler
        with pytest.raises(ToolBail) as excinfo:
            await handler(_SESSION_ID, {"ref": "e12", "description": "click it"})
        assert "stale_snapshot" in excinfo.value.message
        assert "[ref=e1]" in excinfo.value.message  # self-correction snapshot
        assert excinfo.value.detail["code"] == "stale_snapshot"

    async def test_restart_on_failed_response_is_surfaced_in_the_bail(
        self, seams: dict[str, Any]
    ) -> None:
        """The driver spends its once-per-restart signal on a DELIVERED
        response — including an ok:false one (a stale ref is often the
        restart's first symptom). The bail must carry the page-state-lost
        notice, or the fact is silently lost."""
        seams["driver"].return_value = _response(
            ok=False,
            driver_restarted=True,
            error=BrowserError(code="stale_snapshot", message="ref e12 is gone"),
            snapshot='- link "Home" [ref=e1]',
        )
        handler = registry.get("browser_click").handler
        with pytest.raises(ToolBail) as excinfo:
            await handler(_SESSION_ID, {"ref": "e12", "description": "click it"})
        assert "restarted" in excinfo.value.message
        assert "[ref=e1]" in excinfo.value.message  # self-correction snapshot still attached

    async def test_unknown_error_code_is_tolerated(self, seams: dict[str, Any]) -> None:
        """Forward compatibility: an error code this build doesn't know still
        renders as a model-visible failure, never a parse error."""
        seams["driver"].return_value = _response(
            ok=False, error=BrowserError(code="quantum_flux", message="??")
        )
        handler = registry.get("browser_navigate").handler
        with pytest.raises(ToolBail, match="quantum_flux"):
            await handler(_SESSION_ID, {"url": "https://example.com"})

    async def test_transport_failure_raises_toolbail_not_bare_exception(
        self, seams: dict[str, Any]
    ) -> None:
        """A browser fault must never escape as a plain Exception — that would
        evict the CALLING session's sandbox."""
        seams["driver"].side_effect = BrowserUnavailableError("exit 127")
        handler = registry.get("browser_snapshot").handler
        with pytest.raises(ToolBail) as excinfo:
            await handler(_SESSION_ID, {})
        assert "unavailable" in excinfo.value.message
        assert excinfo.value.detail["cause"] == "exit 127"

    async def test_unconfigured_image_gives_the_deployment_message(
        self, seams: dict[str, Any]
    ) -> None:
        seams["driver"].side_effect = BrowserImageUnconfiguredError("no image")
        handler = registry.get("browser_snapshot").handler
        with pytest.raises(ToolBail, match="not available in this deployment"):
            await handler(_SESSION_ID, {})

    async def test_runtime_unsupported_gives_the_runtime_message(
        self, seams: dict[str, Any]
    ) -> None:
        """A ``BrowserRuntimeUnsupportedError`` is a ``BrowserImageUnconfiguredError``
        subclass, so it must be caught by its OWN arm (before the image arm) and
        render the accurate runtime message — not the false 'no browser image is
        configured'."""
        from aios.sandbox.spec import BrowserRuntimeUnsupportedError

        seams["driver"].side_effect = BrowserRuntimeUnsupportedError("custom runtime")
        handler = registry.get("browser_snapshot").handler
        with pytest.raises(ToolBail, match="runtime that is not supported"):
            await handler(_SESSION_ID, {})

    async def test_grant_recheck_refusal_propagates(self, seams: dict[str, Any]) -> None:
        seams["granted"].side_effect = ToolBail("browser_click is not enabled for this agent")
        handler = registry.get("browser_click").handler
        with pytest.raises(ToolBail, match="not enabled"):
            await handler(_SESSION_ID, {"ref": "e1", "description": "x"})
        seams["driver"].assert_not_awaited()


class TestDriverCall:
    async def test_unconfigured_image_passes_through_unlaundered(self) -> None:
        """``BrowserImageUnconfiguredError`` subclasses ``SandboxBackendError``;
        the transport-wrapping arm must NOT launder it into
        ``BrowserUnavailableError`` ("try again shortly" — retrying will never
        help on an unconfigured deployment). Pinned at the driver_call layer
        because the handler tests patch driver_call away."""
        from aios.sandbox.browser import driver_call
        from aios.sandbox.browser_protocol import BrowserRequest

        registry_mock = MagicMock()
        registry_mock.get_or_provision_browser = AsyncMock(
            side_effect=BrowserImageUnconfiguredError("no image configured")
        )
        with pytest.raises(BrowserImageUnconfiguredError):
            await driver_call(
                registry_mock, _ACCOUNT_ID, BrowserRequest(op="snapshot"), timeout_s=5
            )

    async def test_runtime_unsupported_passes_through_unlaundered(self) -> None:
        """``BrowserRuntimeUnsupportedError`` (a ``BrowserImageUnconfiguredError``
        subclass) must ride the same non-retryable pass-through, NOT be laundered
        into a retryable ``BrowserUnavailableError`` — retrying never helps a
        statically-misconfigured runtime. Reordering driver_call's except arms or
        rebasing the exception on ``SandboxBackendError`` would flip this."""
        from aios.sandbox.browser import driver_call
        from aios.sandbox.browser_protocol import BrowserRequest
        from aios.sandbox.spec import BrowserRuntimeUnsupportedError

        registry_mock = MagicMock()
        registry_mock.get_or_provision_browser = AsyncMock(
            side_effect=BrowserRuntimeUnsupportedError("custom runtime")
        )
        with pytest.raises(BrowserRuntimeUnsupportedError):
            await driver_call(
                registry_mock, _ACCOUNT_ID, BrowserRequest(op="snapshot"), timeout_s=5
            )

    async def test_capacity_pressure_becomes_transient_toolbail_not_eviction(self) -> None:
        """A cold provision refused for snapshot-pool pressure raises
        ``SandboxCapacityError`` (a ``SandboxBackendError``). driver_call's
        transport arm must wrap it into ``BrowserUnavailableError`` so the
        handler ToolBails "try again shortly" — NOT let it escape as a bare
        exception the tool dispatcher would classify as evicting the CALLING
        session's sandbox (jarbot#106). This is the exact hole the typed
        exception closes; pinned at the driver_call layer because the handler
        tests patch driver_call away."""
        from aios.sandbox.backends.base import SandboxCapacityError
        from aios.sandbox.browser import driver_call
        from aios.sandbox.browser_protocol import BrowserRequest

        registry_mock = MagicMock()
        registry_mock.get_or_provision_browser = AsyncMock(
            side_effect=SandboxCapacityError("snapshot capacity pressure")
        )
        with pytest.raises(BrowserUnavailableError):
            await driver_call(
                registry_mock, _ACCOUNT_ID, BrowserRequest(op="snapshot"), timeout_s=5
            )


class TestGrantRecheck:
    async def test_absent_arm_is_refused_and_present_arm_admitted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aios.models.agents import ToolSpec

        fake_runtime = MagicMock()
        fake_runtime.require_pool.return_value = MagicMock()
        monkeypatch.setattr(browser_mod, "runtime", fake_runtime)
        monkeypatch.setattr(
            sessions_service, "get_session_basic", AsyncMock(return_value=MagicMock())
        )
        surface = MagicMock()
        surface.tools = [ToolSpec(type="bash"), ToolSpec(type="browser_click")]
        monkeypatch.setattr(agents_service, "load_for_session", AsyncMock(return_value=surface))

        await browser_mod._check_arm_granted(_SESSION_ID, _ACCOUNT_ID, "browser_click")
        with pytest.raises(ToolBail, match="browser_drag is not enabled"):
            await browser_mod._check_arm_granted(_SESSION_ID, _ACCOUNT_ID, "browser_drag")


class TestScreenshot:
    @pytest.fixture(autouse=True)
    def _vision(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        monkeypatch.setitem(vision._VISION_OVERRIDES, "model/vision", True)
        monkeypatch.setitem(vision._VISION_OVERRIDES, "model/text", False)
        monkeypatch.setattr(
            sessions_service,
            "get_session_model",
            AsyncMock(return_value="model/vision"),
        )

    @pytest.fixture
    def plane(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        from aios.sandbox.volumes import ensure_browser_plane_dir

        monkeypatch.setattr(get_settings(), "workspace_root", tmp_path)
        return ensure_browser_plane_dir(_ACCOUNT_ID) / "shots"

    async def test_inlines_downscaled_image_as_parts(
        self, seams: dict[str, Any], plane: Path
    ) -> None:
        payload = valid_png_bytes()
        (plane / "shot.png").write_bytes(payload)
        seams["driver"].return_value = _response(shot_path="shots/shot.png")

        result = await browser_mod.browser_screenshot_handler(_SESSION_ID, {})

        assert isinstance(result.content, list)
        assert result.content[0]["type"] == "text"
        assert "Screenshot: shot.png" in result.content[0]["text"]
        assert result.content[1] == {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64.b64encode(payload).decode()}"},
        }
        assert result.metadata is not None
        assert result.metadata["browser"]["screenshot"] is True

    async def test_unknown_model_inlines_screenshot(
        self, seams: dict[str, Any], plane: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            sessions_service,
            "get_session_model",
            AsyncMock(return_value="future/model"),
        )
        monkeypatch.setattr(
            "litellm.get_model_info",
            lambda _model: (_ for _ in ()).throw(Exception("unknown model")),
        )
        payload = valid_png_bytes()
        (plane / "shot.png").write_bytes(payload)
        seams["driver"].return_value = _response(shot_path="shots/shot.png")

        result = await browser_mod.browser_screenshot_handler(_SESSION_ID, {})

        assert isinstance(result.content, list)
        assert result.content[1]["type"] == "image_url"

    async def test_non_vision_model_degrades_to_text(
        self, seams: dict[str, Any], plane: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            sessions_service,
            "get_session_model",
            AsyncMock(return_value="model/text"),
        )
        (plane / "shot.png").write_bytes(valid_png_bytes())
        seams["driver"].return_value = _response(shot_path="shots/shot.png")

        result = await browser_mod.browser_screenshot_handler(_SESSION_ID, {})
        assert isinstance(result.content, str)
        assert "does not support image input" in result.content

    async def test_hostile_shot_path_is_refused(self, seams: dict[str, Any], plane: Path) -> None:
        seams["driver"].return_value = _response(shot_path="../../../etc/passwd")
        with pytest.raises(ToolBail, match="could not read the image"):
            await browser_mod.browser_screenshot_handler(_SESSION_ID, {})

    async def test_symlinked_shot_is_refused(self, seams: dict[str, Any], plane: Path) -> None:
        """The TOCTOU shape: the shot bytes go into MODEL context, so a shot
        swapped for a symlink into another account's plane must be refused at
        open time (no-follow walk), not by a checkable-then-swappable resolve."""
        victim = plane.parent.parent / "acc_VICTIM" / "profile" / "Cookies"
        victim.parent.mkdir(parents=True)
        victim.write_bytes(b"cookie-jar")
        (plane / "shot.png").symlink_to(victim)
        seams["driver"].return_value = _response(shot_path="shots/shot.png")
        with pytest.raises(ToolBail, match="could not read the image"):
            await browser_mod.browser_screenshot_handler(_SESSION_ID, {})

    async def test_missing_shot_raises_toolbail(self, seams: dict[str, Any], plane: Path) -> None:
        seams["driver"].return_value = _response(shot_path="shots/never-written.png")
        with pytest.raises(ToolBail, match="could not read the image"):
            await browser_mod.browser_screenshot_handler(_SESSION_ID, {})


class TestWiring:
    """Wiring — the union edit and registration (the skill_management shape)."""

    def test_all_eleven_registered_agent_tool_only(self) -> None:
        for name in _BROWSER_TOOLS:
            tool = registry.get(name)
            assert tool is not None, name
            assert tool.transport == "agent_tool", name
            assert tool.parameters_schema.get("additionalProperties") is False, name

    def test_mutating_arms_require_a_description(self) -> None:
        """The per-action intent statement feeds the owner-facing audit trail;
        every state-changing arm must require it."""
        for name in (
            "browser_click",
            "browser_click_xy",
            "browser_type",
            "browser_drag",
            "browser_hover",
            "browser_select_option",
        ):
            schema = registry.get(name).parameters_schema
            assert "description" in schema["required"], name


class TestStructuralInvariants:
    def test_protocol_module_has_no_aios_imports(self) -> None:
        """The browser image COPYs browser_protocol.py verbatim — it must
        import nothing from aios (stdlib + pydantic only)."""
        import aios.sandbox.browser_protocol as proto

        tree = ast.parse(Path(proto.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert not name.startswith("aios"), f"protocol imports {name}"

    def test_handlers_never_resolve_the_calling_sessions_sandbox(self) -> None:
        """The design's central claim: a browser handler touches only the
        BROWSER container. Its registry surface is get_or_provision_browser
        (via driver_call) — never the session/run provision paths, never the
        session workspace."""
        source = Path(browser_mod.__file__).read_text()
        assert "get_or_provision(" not in source
        assert "get_or_provision_run(" not in source
        assert "workspace_dir_for" not in source
