"""Correctness regression for the trigger broken-window fix (issue #1356).

``trigger_create`` / ``trigger_update`` used to call bare
``Model.model_validate(...)``, so a schema-valid-but-Pydantic-semantically-
invalid argument raised a raw ``ValidationError``. That exception is
neither a ``ToolBail`` nor an ``AiosError``, so ``_classify_tool_error``
hit its final arm and EVICTED the sandbox. Routing both handlers through
``tool_input`` converts that path onto the clean, model-visible
``ToolBail`` (``should_evict=False``) channel.

These feed each handler a cron ``source`` whose ``schedule`` passes the
JSON schema (a non-empty string) but fails the write-path
``_validate_cron_expression`` model_validator, and assert the handler now
raises ``ToolBail`` — not a raw ``ValidationError``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

import aios.tools  # noqa: F401 — registers the builtins
from aios.tools import trigger_create as tc
from aios.tools import trigger_update as tu
from aios.tools.invoke import ToolBail

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _stub_runtime(monkeypatch: Any) -> None:
    """Make ``require_pool`` + ``load_session_account_id`` cheap (no DB).

    The service is also stubbed so a *successful* parse cannot fall through
    to a real DB call — but the semantic-invalid cases bail before it.
    """
    monkeypatch.setattr("aios.harness.runtime.require_pool", lambda: object())
    monkeypatch.setattr(
        "aios.services.sessions.load_session_account_id", AsyncMock(return_value="acc_x")
    )
    monkeypatch.setattr("aios.services.triggers.add_trigger", AsyncMock())
    monkeypatch.setattr("aios.services.triggers.update_trigger", AsyncMock())


# A cron schedule that is a non-empty string (passes JSON schema) but is not a
# valid cron expression (fails the write-path model_validator) — the canonical
# schema-valid-but-Pydantic-semantically-invalid input.
_BAD_CRON_SOURCE = {"kind": "cron", "schedule": "definitely not a cron"}
_VALID_ACTION = {"kind": "wake_owner", "content": "wake up"}


async def test_trigger_create_semantic_failure_is_toolbail() -> None:
    with pytest.raises(ToolBail, match="invalid arguments"):
        await tc.trigger_create_handler(
            "ses_1",
            {"name": "t1", "source": _BAD_CRON_SOURCE, "action": _VALID_ACTION},
        )


async def test_trigger_update_semantic_failure_is_toolbail() -> None:
    with pytest.raises(ToolBail, match="invalid arguments"):
        await tu.trigger_update_handler(
            "ses_1",
            {"name": "t1", "source": _BAD_CRON_SOURCE},
        )


async def test_trigger_create_does_not_raise_raw_validation_error() -> None:
    # The whole point of the lift: the raw ValidationError no longer escapes.
    with pytest.raises(ToolBail):
        try:
            await tc.trigger_create_handler(
                "ses_1",
                {"name": "t1", "source": _BAD_CRON_SOURCE, "action": _VALID_ACTION},
            )
        except ValidationError:  # pragma: no cover — would mean the fix regressed
            pytest.fail("raw ValidationError escaped trigger_create_handler")
