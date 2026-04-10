"""Phase 1 end-to-end smoke test against testcontainer Postgres + a fake LiteLLM.

This test is the proof point for Phase 1: a real FastAPI app, a real Postgres,
and a fake LiteLLM that returns a canned assistant message. The flow is:

1. Create a credential
2. Create an environment
3. Create an agent that references the credential and uses a stub model
4. Create a session with `initial_message="hello"`
5. Verify the session goes idle with stop_reason="end_turn"
6. Verify the event log contains the user message + lifecycle events + the
   assistant message we faked

The test runs the harness loop INLINE inside the create-session request
handler (Phase 1 doesn't have a worker yet).
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from tests.conftest import needs_docker


def _build_fake_litellm_response(content: str = "Hello back!") -> dict[str, Any]:
    """Construct an object that quacks like a litellm.acompletion response."""

    class FakeMessage:
        def __init__(self, role: str, content: str) -> None:
            self.role = role
            self.content = content

        def model_dump(self) -> dict[str, Any]:
            return {"role": self.role, "content": self.content}

    return {
        "choices": [{"message": FakeMessage("assistant", content)}],
    }


@pytest.fixture
def test_client(aios_env: dict[str, str]) -> TestClient:
    from aios.api.app import create_app

    app = create_app()
    return TestClient(app)


@needs_docker
def test_phase1_end_to_end(test_client: TestClient, aios_env: dict[str, str]) -> None:
    headers = {"Authorization": f"Bearer {aios_env['AIOS_API_KEY']}"}

    # Health check
    r = test_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    # Create credential
    r = test_client.post(
        "/v1/credentials",
        headers=headers,
        json={"name": "test-key", "provider": "fake", "value": "sk-test-1234"},
    )
    assert r.status_code == 201, r.text
    cred = r.json()
    assert cred["name"] == "test-key"
    assert "value" not in cred  # secret never echoed
    cred_id = cred["id"]
    assert cred_id.startswith("cred_")

    # Create environment
    r = test_client.post(
        "/v1/environments",
        headers=headers,
        json={"name": "default"},
    )
    assert r.status_code == 201, r.text
    env = r.json()
    env_id = env["id"]
    assert env_id.startswith("env_")

    # Create agent
    r = test_client.post(
        "/v1/agents",
        headers=headers,
        json={
            "name": "test-agent",
            "model": "fake/test-model",
            "system": "You are a test assistant.",
            "tools": [],
            "credential_id": cred_id,
        },
    )
    assert r.status_code == 201, r.text
    agent = r.json()
    agent_id = agent["id"]
    assert agent_id.startswith("agent_")
    assert agent["credential_id"] == cred_id

    # Create session with an initial message — patches litellm.acompletion
    # to return our canned response without ever hitting a real model.
    fake_response = _build_fake_litellm_response("Hello, world!")

    async def fake_acompletion(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return fake_response

    with mock.patch("aios.harness.completion.litellm.acompletion", fake_acompletion):
        r = test_client.post(
            "/v1/sessions",
            headers=headers,
            json={
                "agent_id": agent_id,
                "environment_id": env_id,
                "initial_message": "Say hello",
            },
        )
    assert r.status_code == 201, r.text
    session = r.json()
    session_id = session["id"]
    assert session_id.startswith("sess_")
    # The harness ran inline; the session should be idle with stop_reason end_turn.
    assert session["status"] == "idle"
    assert session["stop_reason"] == "end_turn"

    # Fetch the events for this session
    r = test_client.get(f"/v1/sessions/{session_id}/events", headers=headers)
    assert r.status_code == 200
    events = r.json()["data"]
    # Expect at least: user message, turn_started, assistant message, turn_ended
    kinds = [(e["kind"], e["data"].get("role") or e["data"].get("event")) for e in events]
    assert ("message", "user") in kinds
    assert ("lifecycle", "turn_started") in kinds
    assert ("message", "assistant") in kinds
    assert ("lifecycle", "turn_ended") in kinds

    # Find the assistant message and check its content
    assistant_events = [
        e for e in events if e["kind"] == "message" and e["data"].get("role") == "assistant"
    ]
    assert len(assistant_events) == 1
    assert assistant_events[0]["data"]["content"] == "Hello, world!"

    # Seq is gapless and starts at 1
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs)
    assert seqs[0] == 1
    assert seqs == list(range(1, len(seqs) + 1))


@needs_docker
def test_unauthorized_without_bearer(test_client: TestClient) -> None:
    r = test_client.get("/v1/agents")
    assert r.status_code == 401
    body = r.json()
    assert body["error"]["type"] == "unauthorized"


@needs_docker
def test_unauthorized_with_wrong_bearer(test_client: TestClient) -> None:
    r = test_client.get(
        "/v1/agents",
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert r.status_code == 401
