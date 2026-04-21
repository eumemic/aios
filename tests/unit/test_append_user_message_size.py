"""Tests for the user-message size cap in append_user_message."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import asyncpg
import pytest
from pydantic import ValidationError

from aios.errors import PayloadTooLargeError
from aios.models.sessions import SessionCreate
from aios.services.sessions import MAX_USER_MESSAGE_CHARS, append_user_message


class TestAppendUserMessageSizeCap:
    async def test_over_cap_raises_payload_too_large(self) -> None:
        pool = cast("asyncpg.Pool[Any]", MagicMock())
        oversize = "x" * (MAX_USER_MESSAGE_CHARS + 1)

        with pytest.raises(PayloadTooLargeError) as excinfo:
            await append_user_message(pool, "sess_x", oversize)

        err = excinfo.value
        assert err.status_code == 413
        assert err.error_type == "payload_too_large"
        assert err.detail["max_chars"] == MAX_USER_MESSAGE_CHARS
        assert err.detail["got_chars"] == MAX_USER_MESSAGE_CHARS + 1

    async def test_over_cap_short_circuits_before_db(self) -> None:
        pool = MagicMock()
        oversize = "x" * (MAX_USER_MESSAGE_CHARS + 1)

        with pytest.raises(PayloadTooLargeError):
            await append_user_message(cast("asyncpg.Pool[Any]", pool), "sess_x", oversize)

        pool.acquire.assert_not_called()

    async def test_at_cap_does_not_raise(self) -> None:
        pool = MagicMock()
        pool.acquire.side_effect = RuntimeError("past the gate")
        at_cap = "x" * MAX_USER_MESSAGE_CHARS

        with pytest.raises(RuntimeError, match="past the gate"):
            await append_user_message(cast("asyncpg.Pool[Any]", pool), "sess_x", at_cap)


class TestSessionCreateInitialMessageCap:
    # POST /sessions creates the session row before appending initial_message;
    # validating the length on the Pydantic model keeps that path
    # correct-by-construction (no orphan session on 413).

    def test_over_cap_initial_message_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SessionCreate(
                agent_id="agent_x",
                environment_id="env_x",
                initial_message="x" * (MAX_USER_MESSAGE_CHARS + 1),
            )

    def test_at_cap_initial_message_accepted(self) -> None:
        model = SessionCreate(
            agent_id="agent_x",
            environment_id="env_x",
            initial_message="x" * MAX_USER_MESSAGE_CHARS,
        )
        assert model.initial_message is not None
        assert len(model.initial_message) == MAX_USER_MESSAGE_CHARS
