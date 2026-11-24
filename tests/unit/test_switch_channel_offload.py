"""WP-25 P6: ``switch_channel_handler`` renders the recap off the event loop.

``render_reorient_block`` can reach the synchronous cache-miss
``host_path.read_bytes()`` in ``aios.harness.context._apply_attachments``
(it threads ``session_id`` through to ``render_user_event``), so the handler
must call it via ``asyncio.to_thread`` rather than inline on the loop.

Patch target: the handler resolves ``render_reorient_block`` as a module
global of ``aios.tools.switch_channel`` (it is defined there), so the stub is
installed on that module.
"""

from __future__ import annotations

import pathlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.harness import context as context_mod
from aios.tools import switch_channel as sc
from aios.tools.registry import ToolResult
from tests.unit.conftest import fake_pool_yielding_conn

# Reuse the recap-vision module's autouse vision override + workspace fixture.
from tests.unit.test_switch_channel_recap_vision import (  # noqa: F401
    _VALID_JPEG,
    CHAN_A,
    SESSION_ID,
    _stage_image,
    _stub_supports_vision,
    _user_with_image,
    temp_workspace_root,
)


@pytest.fixture(autouse=True)
def _fresh_attachment_cache() -> Iterator[None]:
    context_mod._clear_attachment_cache()
    yield
    context_mod._clear_attachment_cache()


def _conn() -> Any:
    conn = MagicMock()
    txn = MagicMock()
    txn.__aenter__ = AsyncMock(return_value=None)
    txn.__aexit__ = AsyncMock(return_value=None)
    conn.transaction = MagicMock(return_value=txn)
    return conn


@contextmanager
def _handler_env(events: list[Any], model: str) -> Iterator[None]:
    pool = fake_pool_yielding_conn(_conn())
    q = "aios.tools.switch_channel.queries"
    with (
        patch("aios.tools.switch_channel.runtime.require_pool", return_value=pool),
        patch(
            "aios.tools.switch_channel.sessions_service.load_session_account_id",
            new_callable=AsyncMock,
            return_value="acct_1",
        ),
        patch(
            "aios.tools.switch_channel.accounts_service.resolve_effective_timezone_on",
            new_callable=AsyncMock,
            return_value="UTC",
        ),
        patch(f"{q}.is_session_focal_locked", new_callable=AsyncMock, return_value=False),
        patch(f"{q}.get_session_focal_channel", new_callable=AsyncMock, return_value=None),
        patch(f"{q}.list_session_channels", new_callable=AsyncMock, return_value=[CHAN_A]),
        patch(f"{q}.set_session_focal_channel", new_callable=AsyncMock),
        patch(f"{q}.read_message_events", new_callable=AsyncMock, return_value=events),
        patch(f"{q}.get_session_model", new_callable=AsyncMock, return_value=model),
    ):
        yield


async def test_render_reorient_block_runs_off_loop_thread() -> None:
    loop_thread = threading.get_ident()
    seen: list[int] = []
    sentinel = "RECAP-SENTINEL"

    def _stub(*_args: Any, **_kwargs: Any) -> str:
        seen.append(threading.get_ident())
        return sentinel

    with _handler_env([], "model/vision"), patch.object(sc, "render_reorient_block", _stub):
        result = await sc.switch_channel_handler(SESSION_ID, {"channel_id": CHAN_A})

    assert seen, "stub never called — wrong patch target"
    assert seen[0] != loop_thread
    assert isinstance(result, ToolResult)
    assert result.content == sentinel


async def test_cache_miss_read_bytes_happens_off_loop(temp_workspace_root: Path) -> None:  # noqa: F811
    path = _stage_image(temp_workspace_root, "signal", "p.jpg", _VALID_JPEG)
    events = [
        _user_with_image(
            1, channel=CHAN_A, content="look", sandbox_path=path, size=len(_VALID_JPEG)
        )
    ]
    loop_thread = threading.get_ident()
    read_threads: list[int] = []
    real_read_bytes = pathlib.Path.read_bytes

    def _spy(self: pathlib.Path) -> bytes:
        read_threads.append(threading.get_ident())
        return real_read_bytes(self)

    with _handler_env(events, "model/vision"), patch.object(pathlib.Path, "read_bytes", _spy):
        result = await sc.switch_channel_handler(SESSION_ID, {"channel_id": CHAN_A})

    assert read_threads, "no cache-miss read happened"
    assert all(t != loop_thread for t in read_threads)

    context_mod._clear_attachment_cache()
    inline = sc.render_reorient_block(
        events, CHAN_A, model="model/vision", session_id=SESSION_ID, tz_name="UTC"
    )
    assert result.content == inline
