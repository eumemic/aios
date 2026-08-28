"""The input injector maps the raw event vocabulary onto playwright's page
API and detects multi-clicks (the wire carries no click count)."""

from __future__ import annotations

from typing import Any

from aios_browser_driver.takeover.injector import InputInjector


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _record(self, name: str) -> Any:
        async def method(*args: Any, **kwargs: Any) -> None:
            self.calls.append((name, args, kwargs))

        return method


class _FakeMouse(_Recorder):
    def __init__(self) -> None:
        super().__init__()
        self.move = self._record("move")
        self.down = self._record("down")
        self.up = self._record("up")
        self.wheel = self._record("wheel")


class _FakeKeyboard(_Recorder):
    def __init__(self) -> None:
        super().__init__()
        self.down = self._record("down")
        self.up = self._record("up")
        self.insert_text = self._record("insert_text")


class _FakePage:
    def __init__(self) -> None:
        self.mouse = _FakeMouse()
        self.keyboard = _FakeKeyboard()


async def test_pointer_and_wheel_map_to_mouse() -> None:
    page = _FakePage()
    inj = InputInjector(page)  # type: ignore[arg-type]
    await inj.apply({"type": "pointer_move", "x": 5, "y": 6}, at=0.0)
    await inj.apply({"type": "pointer_down", "x": 5, "y": 6, "button": "right"}, at=0.0)
    await inj.apply({"type": "pointer_up", "x": 5, "y": 6, "button": "right"}, at=0.0)
    await inj.apply({"type": "wheel", "x": 5, "y": 6, "dx": 0, "dy": 120}, at=0.0)

    assert ("move", (5.0, 6.0), {}) in page.mouse.calls
    down = next(c for c in page.mouse.calls if c[0] == "down")
    assert down[2]["button"] == "right" and down[2]["click_count"] == 1
    assert ("wheel", (0.0, 120.0), {}) in page.mouse.calls


async def test_keys_and_text_map_to_keyboard() -> None:
    page = _FakePage()
    inj = InputInjector(page)  # type: ignore[arg-type]
    await inj.apply({"type": "key_down", "key": "Enter"}, at=0.0)
    await inj.apply({"type": "key_up", "key": "Enter"}, at=0.0)
    await inj.apply({"type": "text", "text": "hi"}, at=0.0)

    assert ("down", ("Enter",), {}) in page.keyboard.calls
    assert ("up", ("Enter",), {}) in page.keyboard.calls
    assert ("insert_text", ("hi",), {}) in page.keyboard.calls


async def test_double_click_detected_from_two_rapid_downs() -> None:
    page = _FakePage()
    inj = InputInjector(page)  # type: ignore[arg-type]
    await inj.apply({"type": "pointer_down", "x": 10, "y": 10, "button": "left"}, at=0.0)
    await inj.apply({"type": "pointer_up", "x": 10, "y": 10, "button": "left"}, at=0.05)
    await inj.apply({"type": "pointer_down", "x": 10, "y": 10, "button": "left"}, at=0.1)

    downs = [c for c in page.mouse.calls if c[0] == "down"]
    assert downs[0][2]["click_count"] == 1
    assert downs[1][2]["click_count"] == 2


async def test_far_apart_downs_are_single_clicks() -> None:
    page = _FakePage()
    inj = InputInjector(page)  # type: ignore[arg-type]
    await inj.apply({"type": "pointer_down", "x": 10, "y": 10, "button": "left"}, at=0.0)
    await inj.apply({"type": "pointer_down", "x": 200, "y": 200, "button": "left"}, at=0.1)
    downs = [c for c in page.mouse.calls if c[0] == "down"]
    assert downs[0][2]["click_count"] == 1
    assert downs[1][2]["click_count"] == 1  # moved too far to be a double-click


async def test_unknown_event_type_is_ignored() -> None:
    page = _FakePage()
    inj = InputInjector(page)  # type: ignore[arg-type]
    await inj.apply({"type": "teleport"}, at=0.0)
    assert page.mouse.calls == [] and page.keyboard.calls == []


async def test_wheel_uses_last_pointer_position_when_absent() -> None:
    page = _FakePage()
    inj = InputInjector(page)  # type: ignore[arg-type]
    await inj.apply({"type": "pointer_move", "x": 30, "y": 40}, at=0.0)
    await inj.apply({"type": "wheel", "dx": 0, "dy": 10}, at=0.0)
    moves = [c for c in page.mouse.calls if c[0] == "move"]
    assert moves[-1] == ("move", (30.0, 40.0), {})  # wheel re-moved to last xy
