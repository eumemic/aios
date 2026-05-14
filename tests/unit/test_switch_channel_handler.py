"""Unit tests for the ``switch_channel`` tool handler — focal-lock + dispatch.

The recap-builder pure function is covered in ``test_switch_channel.py``.
This file exercises the *handler* itself: the per_chat focal-lock check
that rejects switch attempts on sessions spawned for a single chat, and
the bound-channels validation path.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, patch

from aios.harness.channels import SWITCH_CHANNEL_METADATA_KEY
from aios.tools.registry import ToolResult
from aios.tools.switch_channel import switch_channel_handler
from tests.unit.conftest import fake_pool_yielding_conn


def _patch_pool(pool: Any) -> Any:
    """Patch ``runtime.require_pool`` to return our fake pool."""
    return patch("aios.tools.switch_channel.runtime.require_pool", return_value=pool)


def _patched_conn() -> Any:
    """Connection mock with a working ``conn.transaction()`` async-cm."""
    conn = MagicMock()
    txn = MagicMock()
    txn.__aenter__ = AsyncMock(return_value=None)
    txn.__aexit__ = AsyncMock(return_value=None)
    conn.transaction = MagicMock(return_value=txn)
    return conn


class TestPerChatLock:
    """Sessions with ``focal_locked = TRUE`` are bound to a single chat
    by construction (today set at per_chat-mode spawn time);
    ``switch_channel`` must reject any attempt to mutate focal on them.
    """

    async def test_rejects_when_session_was_per_chat_spawned(self) -> None:
        conn = _patched_conn()
        pool = fake_pool_yielding_conn(conn)
        with (
            _patch_pool(pool),
            patch(
                "aios.tools.switch_channel.queries.is_session_focal_locked",
                new_callable=AsyncMock,
            ) as locked,
            patch(
                "aios.tools.switch_channel.queries.get_session_focal_channel",
                new_callable=AsyncMock,
            ) as focal,
            patch(
                "aios.tools.switch_channel.queries.set_session_focal_channel",
                new_callable=AsyncMock,
            ) as set_focal,
            patch(
                "aios.tools.switch_channel.queries.list_session_channels",
                new_callable=AsyncMock,
            ) as list_channels,
        ):
            locked.return_value = True
            result = await switch_channel_handler(
                "sess_per_chat", {"channel_id": "signal/+1/chat-2"}
            )

        assert isinstance(result, ToolResult)
        assert result.is_error is True
        assert "focal channel is locked" in result.content
        marker = result.metadata[SWITCH_CHANNEL_METADATA_KEY]
        assert marker == {"target": "signal/+1/chat-2", "success": False}
        # Focal-lock short-circuits before any other DB read or write.
        focal.assert_not_awaited()
        set_focal.assert_not_awaited()
        list_channels.assert_not_awaited()

    async def test_rejects_per_chat_even_when_target_is_null(self) -> None:
        """Putting the phone down (target=None) is still focal mutation
        on a per_chat session and must be rejected.
        """
        conn = _patched_conn()
        pool = fake_pool_yielding_conn(conn)
        with (
            _patch_pool(pool),
            patch(
                "aios.tools.switch_channel.queries.is_session_focal_locked",
                new_callable=AsyncMock,
            ) as locked,
            patch(
                "aios.tools.switch_channel.queries.set_session_focal_channel",
                new_callable=AsyncMock,
            ) as set_focal,
        ):
            locked.return_value = True
            result = await switch_channel_handler("sess_per_chat", {"channel_id": None})

        assert result.is_error is True
        assert result.metadata[SWITCH_CHANNEL_METADATA_KEY] == {
            "target": None,
            "success": False,
        }
        set_focal.assert_not_awaited()


class TestNonPerChatPathStillWorks:
    """Sanity check that ordinary (non-locked) sessions reach the
    rest of the handler — the focal-lock guard short-circuits *only*
    when ``focal_locked`` is TRUE.
    """

    async def test_no_op_when_target_already_focal(self) -> None:
        conn = _patched_conn()
        pool = fake_pool_yielding_conn(conn)
        with (
            _patch_pool(pool),
            patch(
                "aios.tools.switch_channel.queries.is_session_focal_locked",
                new_callable=AsyncMock,
            ) as locked,
            patch(
                "aios.tools.switch_channel.queries.get_session_focal_channel",
                new_callable=AsyncMock,
            ) as focal,
            patch(
                "aios.tools.switch_channel.queries.set_session_focal_channel",
                new_callable=AsyncMock,
            ) as set_focal,
        ):
            locked.return_value = False
            focal.return_value = "signal/+1/chat-1"
            result = await switch_channel_handler("sess_normal", {"channel_id": "signal/+1/chat-1"})

        assert result.is_error is False
        assert result.metadata == {}
        assert "already" in result.content
        set_focal.assert_not_awaited()

    async def test_clear_focal_succeeds(self) -> None:
        conn = _patched_conn()
        pool = fake_pool_yielding_conn(conn)
        with (
            _patch_pool(pool),
            patch(
                "aios.tools.switch_channel.queries.is_session_focal_locked",
                new_callable=AsyncMock,
            ) as locked,
            patch(
                "aios.tools.switch_channel.queries.get_session_focal_channel",
                new_callable=AsyncMock,
            ) as focal,
            patch(
                "aios.tools.switch_channel.queries.set_session_focal_channel",
                new_callable=AsyncMock,
            ) as set_focal,
        ):
            locked.return_value = False
            focal.return_value = "signal/+1/chat-1"
            result = await switch_channel_handler("sess_normal", {"channel_id": None})

        assert result.is_error is False
        assert result.metadata[SWITCH_CHANNEL_METADATA_KEY] == {
            "target": None,
            "success": True,
        }
        set_focal.assert_awaited_once_with(conn, "sess_normal", None, account_id=ANY)
