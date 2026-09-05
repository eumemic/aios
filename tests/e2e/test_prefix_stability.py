"""E2E: the outbound message list is a prefix-stable function of the log.

Runs the REAL ``run_session_step`` against the testcontainer Postgres with the
scripted model and asserts the prompt-prefix-cache invariant on the captured
outbound payloads: the ``messages`` sent for step N must be a message-for-
message prefix of the ``messages`` sent for step N+1.

The mid-inference case is the one that broke: an inbound that lands while the
model is generating is blind to the assistant turn that inference produces.
``build_messages`` used to defer such a message to the tail of the window and
then, once a newer assistant existed, render it back at its seq position —
re-ordering the prefix on both providers.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import litellm

from tests.e2e.harness import Harness, assistant
from tests.support import assert_message_prefix

_AGENT_MODEL = "fake/test"


class TestPrefixStability:
    async def test_inbound_during_inference_keeps_prefix_across_three_steps(
        self, harness: Harness
    ) -> None:
        harness.script_model([assistant("a1"), assistant("a2"), assistant("a3")])
        session = await harness.start("hello")

        # Inject an inbound while the FIRST agent inference is in flight: wrap
        # the harness's fake completion so the append happens inside the model
        # phase of step 1 (after the window was read, before the assistant is
        # appended) — exactly the blind-spot shape.
        # The harness fixture already patched ``litellm.acompletion`` on the
        # shared module object; wrap whatever is installed there.
        original = litellm.acompletion
        injected = False

        async def hooked(**kwargs: Any) -> Any:
            nonlocal injected
            if not injected and kwargs.get("model") == _AGENT_MODEL:
                injected = True
                await harness.inject_message(session.id, "mid-inference note")
            return await original(**kwargs)

        with mock.patch("aios.harness.completion.litellm.acompletion", hooked):
            await harness.run_step(session.id)
        assert injected, "the hook never saw the agent's model call"
        # Step 2: the model replies to the blind-spot note.
        await harness.run_until_idle(session.id)
        # Step 3: a later inbound, after the note has been answered.
        await harness.inject_message(session.id, "follow-up")
        await harness.run_until_idle(session.id)

        calls = [c["messages"] for c in harness.model_calls]
        assert len(calls) == 3, f"expected 3 agent calls, got {len(calls)}"
        assert_message_prefix(calls[0], calls[1])
        assert_message_prefix(calls[1], calls[2])
