"""The renderer, pinned (#2471).

A fixed event log rendered to a fixed request must hash to a pinned value. A
change to how requests are rendered fails here until ``RENDER_VERSION`` is
bumped and the pins are updated: rendered bytes are what request capture
reproduces, and what every session's prompt cache is keyed on.

The log exercises the renderer's branches: an omission marker, merged user
turns in a non-UTC timezone, reasoning content, tool results with images and a
mislabeled MIME type, resolved, blind-spot, in-flight and external tool calls,
a model-visible lifecycle notice, inlined and missing attachments, a non-focal
channel notification carrying a chat name and sender, a persisted reminder row
and this step's reminder rows.
It renders for three gate models: an unknown model (vision allowed, no
thinking), a thinking model and a model without vision.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import litellm
import pytest

from aios.harness import vision
from aios.harness.completion import model_descriptor
from aios.harness.context import build_messages, finalize_messages
from aios.harness.reminders import reminder_event_data
from aios.harness.request_capture import RENDER_VERSION, sha256_hex
from aios.harness.window import WindowOmission
from aios.models.events import Event, EventKind
from tests.helpers.images import valid_png_bytes

_SESSION = "sess_golden"
_T0 = datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC)
_SYSTEM = "You are the golden agent.\n\nBe brief."
_TOOLS = [
    {"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}},
    {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}},
]
_PARAMS = {"temperature": 0.2}
_PNG = valid_png_bytes()

# Bump RENDER_VERSION and re-pin when a rendering change is deliberate.
_PINNED = {
    "golden/plain": "a1a51c9e7e684a5a7f5ae20ae1dd7af245d75a512357282941271322a2918185",
    "golden/thinker": "e526285b9acd27651804eff46e20c1bbd7dfd7d534bdf71aa1d18ccae4216dbc",
    "golden/blind": "5c06668b405caf676d9ac987d9fdc43fa96749a4780ce0e09216f2e7fd54b077",
}


def _event(seq: int, data: dict[str, Any], *, kind: EventKind = "message", **extra: Any) -> Event:
    return Event(
        id=f"evt_{seq:03d}",
        session_id=_SESSION,
        seq=seq,
        kind=kind,
        data=data,
        created_at=_T0 + timedelta(minutes=seq),
        **extra,
    )


def _call(call_id: str, name: str) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


def _log(workspace: Path) -> list[Event]:
    (workspace / "chart.png").write_bytes(_PNG)
    png_as_jpeg = "data:image/jpeg;base64," + base64.b64encode(_PNG).decode()
    return [
        _event(
            1,
            {
                "role": "user",
                "content": "first question",
                "metadata": {"request": {"request_id": "req_1"}},
            },
        ),
        _event(2, {"role": "user", "content": "and a follow-up"}),
        _event(
            3,
            {
                "role": "assistant",
                "content": "On it.",
                "reasoning_content": "Plan: run, read, wait.",
                "tool_calls": [
                    _call("t_done", "bash"),
                    _call("t_blind", "read"),
                    _call("t_running", "bash"),
                    _call("t_external", "custom_tool"),
                ],
                "reacting_to": 2,
            },
        ),
        _event(
            4,
            {
                "role": "tool",
                "tool_call_id": "t_done",
                "content": [
                    {"type": "text", "text": "exit 0"},
                    {"type": "image_url", "image_url": {"url": png_as_jpeg}},
                ],
            },
        ),
        _event(
            5,
            {"event": "sandbox_fs_reset", "reason": "snapshot_missing"},
            kind="lifecycle",
        ),
        _event(
            6,
            {
                "role": "user",
                "content": "here are the files",
                "metadata": {
                    "channel": "sig/a",
                    "attachments": [
                        {
                            "in_sandbox_path": "/workspace/chart.png",
                            "name": "chart.png",
                            "content_type": "image/png",
                            "size": len(_PNG),
                        },
                        {
                            "in_sandbox_path": "/mnt/attachments/gone.png",
                            "name": "gone.png",
                            "content_type": "image/png",
                            "size": 10,
                        },
                    ],
                },
            },
            orig_channel="sig/a",
            focal_channel_at_arrival="sig/a",
        ),
        _event(7, {"role": "assistant", "content": "Looking now.", "reacting_to": 6}),
        _event(8, {"role": "tool", "tool_call_id": "t_blind", "content": "late file body"}),
        _event(
            9,
            {
                "role": "user",
                "content": "ping from elsewhere",
                "metadata": {
                    "channel": "sig/b",
                    "chat_type": "group",
                    "chat_name": "Ops",
                    "sender_name": "Bob",
                },
            },
            orig_channel="sig/b",
            focal_channel_at_arrival="sig/a",
        ),
        _event(10, reminder_event_data("obligations", "You owe a reply to req_1.")),
    ]


def render(gate_model: str, workspace: Path) -> dict[str, Any]:
    """The render ``compose_step_context`` performs, from its inputs."""
    ctx = build_messages(
        _log(workspace),
        system_prompt=_SYSTEM,
        model=gate_model,
        session_id=_SESSION,
        workspace_path=workspace,
        in_flight_tool_call_ids=frozenset({"t_running"}),
        tz_name="Asia/Kolkata",
        omission=WindowOmission(began_at=_T0 - timedelta(days=3), omitted_messages=12),
    )
    messages = finalize_messages(
        ctx.messages,
        reminder_contents=("Channels: sig/a (focal), sig/b.",),
        model=gate_model,
    )
    return {"messages": messages, "tools": _TOOLS, "params": _PARAMS}


@pytest.fixture(autouse=True)
def _fixed_catalog(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin the model verdicts the renderer reads, independent of LiteLLM's catalog."""
    model_descriptor.cache_clear()
    monkeypatch.setitem(vision._VISION_OVERRIDES, "golden/blind", False)
    real = litellm.supports_reasoning
    monkeypatch.setattr(
        litellm,
        "supports_reasoning",
        lambda model, *a, **k: model == "golden/thinker" or real(model, *a, **k),
    )
    yield
    model_descriptor.cache_clear()


@pytest.mark.parametrize("gate_model", sorted(_PINNED))
def test_rendering_matches_the_pinned_hash(gate_model: str, tmp_path: Path) -> None:
    assert RENDER_VERSION == 2
    assert sha256_hex(render(gate_model, tmp_path)) == _PINNED[gate_model]
