"""Regression guard: a transient litellm 502 must not corrupt step-level state.

After a ``BadGatewayError`` on the first step, the next step must consume
the next scripted response and end the turn cleanly.  Pins the in-process
recovery path that hypothesis 3 of issue #289 questioned (and that the
proposed repro confirmed is intact).
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import litellm
import pytest

from tests.conftest import needs_docker
from tests.e2e.harness import Harness, assistant, last_assistant_content

pytestmark = pytest.mark.docker


def _make_bad_gateway() -> litellm.exceptions.BadGatewayError:
    return litellm.exceptions.BadGatewayError(
        message="Server error '502 Bad Gateway' for url 'https://api.example.com/v1/messages'",
        llm_provider="anthropic",
        model="anthropic/claude-opus-4-7",
    )


@needs_docker
class TestLiteLLMBadGatewayRecovery:
    async def test_second_step_runs_after_first_step_502(self, harness: Harness) -> None:
        harness.script_model([assistant("Hello after retry.")])
        first_call = True

        async def fake_acompletion(**kwargs: Any) -> Any:
            nonlocal first_call
            if first_call:
                first_call = False
                raise _make_bad_gateway()
            if kwargs.get("stream"):
                return harness._pop_streaming_response(**kwargs)
            return harness._pop_response(**kwargs)

        # ``loop.py`` binds ``defer_wake`` via ``from … import …``, so the
        # conftest's patch on ``aios.services.wake.defer_wake`` doesn't reach
        # the loop-level call site.  Patch the loop binding directly.
        async def _noop_defer_wake(*args: Any, **kwargs: Any) -> None:
            return None

        with (
            mock.patch("aios.harness.completion.litellm.acompletion", fake_acompletion),
            mock.patch("aios.harness.loop.defer_wake", _noop_defer_wake),
        ):
            session = await harness.start("hello")

            await harness.run_step(session.id)
            s_after_first = await harness.session(session.id)
            assert s_after_first.status == "rescheduling"

            await harness.run_step(session.id)
            s_after_second = await harness.session(session.id)
            assert s_after_second.status == "idle"
            assert s_after_second.stop_reason == {"type": "end_turn"}

            events = await harness.events(session.id)
            assert last_assistant_content(events) == "Hello after retry."
