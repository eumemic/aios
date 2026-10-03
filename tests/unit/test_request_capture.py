"""Request capture's record and blobs (#2471)."""

from __future__ import annotations

from aios.harness.completion import LlmRequest
from aios.harness.request_capture import capture_request, encode
from aios.models.agents import AgentBinding


def test_an_inline_api_key_reaches_neither_a_blob_nor_the_hash() -> None:
    request = LlmRequest(
        messages=[{"role": "user", "content": "hi"}],
        params={"api_key": "sk-inline-secret", "temperature": 0.2},
    )
    capture = capture_request(
        request,
        system_prompt="sys",
        model="openrouter/x",
        capability_model="openrouter/x",
        binding=AgentBinding(agent_id="agt_1", version=1),
        after_seq=None,
        through_seq=3,
        omission=None,
        reminder_seqs=(),
        in_flight_tool_call_ids=frozenset(),
        tz_name="UTC",
        workspace_path=None,
    )
    assert all(b"sk-inline-secret" not in body for body in capture.blobs.values())
    assert encode({"temperature": 0.2}) in capture.blobs.values()
    # The request itself is untouched: the send path still has its params.
    assert request.params == {"api_key": "sk-inline-secret", "temperature": 0.2}
