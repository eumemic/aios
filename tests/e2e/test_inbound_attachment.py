"""E2E test for inbound attachment staging through ``/v1/connectors/inbound``.

Post-#328 the inbound endpoint takes ``multipart/form-data``: text
fields ride as ``Form`` parts, attachments ride as ``UploadFile``
parts. The api streams each file's bytes into
``<workspace>/_attachments/<session>/<connector>/<...>``; there's no
shared filesystem coupling (closes #322 P1).

This test pins the api-side staging contract:

* Bytes from each multipart ``UploadFile`` are streamed into per-session
  staging at ``<workspace>/_attachments/<session>/<connector>/<event-id>-<filename>``.
* The appended event records each attachment with the expected
  ``in_sandbox_path`` shape so the model sees a stable path.

It does NOT exercise the docker-level bind-mount plumbing — that's
covered by the manual smoke documented in PR 1's verification section.
"""

from __future__ import annotations

import httpx

from aios.ids import EVENT, make_id, split_id
from tests.conftest import needs_docker
from tests.e2e.harness import Harness
from tests.helpers.connections import bearer


def _new_event_id() -> str:
    return split_id(make_id(EVENT))[1]


async def _create_connection(http_client: httpx.AsyncClient, account: str) -> str:
    r = await http_client.post(
        "/v1/connections",
        json={"connector": "echo", "account": account},
    )
    assert r.status_code == 201, r.text
    return str(r.json()["id"])


async def _attach(harness: Harness, connection_id: str, session_id: str) -> None:
    from aios.services import connections as connections_service

    await connections_service.attach_connection(harness._pool, connection_id, session_id=session_id)


async def _issue_token(http_client: httpx.AsyncClient, connection_id: str) -> str:
    r = await http_client.post("/v1/connector-tokens", json={"connection_id": connection_id})
    assert r.status_code == 201, r.text
    return str(r.json()["plaintext"])


@needs_docker
class TestInboundAttachmentStaging:
    async def test_attachment_streamed_into_workspace(
        self, http_client: httpx.AsyncClient, harness: Harness
    ) -> None:
        import json as _json

        from aios.sandbox.volumes import session_attachments_dir
        from aios.services import agents as agents_service
        from aios.services import environments as env_svc
        from aios.services import sessions as sess_svc

        agent = await agents_service.create_agent(
            harness._pool,
            name=f"att-{id(self)}",
            model="fake/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await env_svc.create_environment(harness._pool, name=f"env-att-{id(self)}")
        session = await sess_svc.create_session(
            harness._pool,
            agent_id=agent.id,
            environment_id=env.id,
            title=None,
            metadata={},
        )
        connection_id = await _create_connection(http_client, f"att-{id(self)}")
        await _attach(harness, connection_id, session.id)
        token = await _issue_token(http_client, connection_id)

        event_id = _new_event_id()
        payload = b"\x89PNG\r\n\x1a\nfake-image-bytes"
        r = await http_client.post(
            "/v1/connectors/inbound",
            headers=bearer(token),
            data={
                "event_id": event_id,
                "chat_id": "chat-att",
                "sender": _json.dumps({"display_name": "Alice"}),
                "content": "here's a photo",
            },
            files=[("attachments", ("photo.png", payload, "image/png"))],
        )
        assert r.status_code == 201, r.text

        # Bytes ended up at the canonical per-session-per-connector path.
        staged_dir = session_attachments_dir(session.id) / "echo"
        staged_files = list(staged_dir.iterdir())
        assert len(staged_files) == 1
        target = staged_files[0]
        assert target.name == f"{event_id}-photo.png"
        assert target.read_bytes() == payload

        # Event log carries the attachment record with the in-sandbox path.
        events = await harness.events(session.id)
        user_events = [e for e in events if e.kind == "message" and e.data.get("role") == "user"]
        assert len(user_events) == 1
        attachments = user_events[0].data["metadata"]["attachments"]
        assert len(attachments) == 1
        assert attachments[0]["filename"] == "photo.png"
        assert attachments[0]["content_type"] == "image/png"
        assert attachments[0]["size"] == len(payload)
        assert attachments[0]["in_sandbox_path"] == f"/mnt/attachments/echo/{event_id}-photo.png"
