"""E2E test for inbound attachment staging through ``/v1/connectors/inbound``.

The new post-#318 architecture has connectors POST inbound user messages
with an ``attachments`` list of ``{host_path, filename, content_type,
size}`` records.  The api reads those bytes via ``host_path`` (assumed
reachable from inside the api container via a shared bind mount) and
stages them into ``<workspace>/_attachments/<session>/<connector>/<...>``.

This test pins the api-side staging contract:

* The api's ``handle_inbound`` service moves the source file via
  ``os.rename`` (or ``shutil.copy2`` + unlink on EXDEV).
* The appended event records each attachment with the expected
  ``in_sandbox_path`` shape so the model sees a stable path.
* The original ``host_path`` source file is gone after staging (single
  copy of the bytes lives in the staged location).

It does NOT exercise the docker-level cross-container bind-mount
plumbing — that's covered by the manual smoke documented in PR 1's
verification section.  This test catches regressions in the python-level
staging logic only.
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
    async def test_attachment_moved_into_workspace(
        self, http_client: httpx.AsyncClient, harness: Harness
    ) -> None:
        from aios.config import get_settings
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

        # Pre-stage a fake "downloaded by the connector" file in a sibling
        # location under workspace_root.  In a real deployment this would
        # be the connector's own state dir bind-mounted into the api
        # container at the same path; here we just pick a writable spot.
        workspace_root = get_settings().workspace_root
        workspace_root.mkdir(parents=True, exist_ok=True)
        connector_temp = workspace_root / "_inbound_temp"
        connector_temp.mkdir(parents=True, exist_ok=True)
        source = connector_temp / "photo.png"
        payload = b"\x89PNG\r\n\x1a\nfake-image-bytes"
        source.write_bytes(payload)

        event_id = _new_event_id()
        r = await http_client.post(
            "/v1/connectors/inbound",
            headers=bearer(token),
            json={
                "event_id": event_id,
                "chat_id": "chat-att",
                "sender": {"display_name": "Alice"},
                "content": "here's a photo",
                "attachments": [
                    {
                        "host_path": str(source),
                        "filename": "photo.png",
                        "content_type": "image/png",
                        "size": len(payload),
                    }
                ],
            },
        )
        assert r.status_code == 201, r.text

        # Source is gone — bytes moved into staging.
        assert not source.exists()

        # Staged at the canonical per-session-per-connector path.
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

    async def test_missing_host_path_drops_inbound(
        self, http_client: httpx.AsyncClient, harness: Harness
    ) -> None:
        """A connector that POSTs an unreachable ``host_path`` gets a clean
        5xx with ``drop_reason='attachment_staging_failed'`` — the api
        wraps the underlying ``FileNotFoundError`` as
        ``AttachmentStagingError`` rather than letting it 500 uncaught."""
        from aios.services import agents as agents_service
        from aios.services import environments as env_svc
        from aios.services import sessions as sess_svc

        agent = await agents_service.create_agent(
            harness._pool,
            name=f"attmiss-{id(self)}",
            model="fake/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
        )
        env = await env_svc.create_environment(harness._pool, name=f"env-attmiss-{id(self)}")
        session = await sess_svc.create_session(
            harness._pool,
            agent_id=agent.id,
            environment_id=env.id,
            title=None,
            metadata={},
        )
        connection_id = await _create_connection(http_client, f"attmiss-{id(self)}")
        await _attach(harness, connection_id, session.id)
        token = await _issue_token(http_client, connection_id)

        r = await http_client.post(
            "/v1/connectors/inbound",
            headers=bearer(token),
            json={
                "event_id": _new_event_id(),
                "chat_id": "chat-attmiss",
                "sender": {"display_name": "Alice"},
                "content": "broken pointer",
                "attachments": [
                    {
                        "host_path": "/tmp/does-not-exist-" + str(id(self)),
                        "filename": "missing.png",
                        "content_type": "image/png",
                        "size": 0,
                    }
                ],
            },
        )
        assert r.status_code >= 400, r.text
        body = r.json()
        # The error body must carry a machine-readable drop_reason so the
        # connector can branch on it (today's prod bug — the connector
        # crashed on raise_for_status, throwing the body away).
        assert body["error"]["detail"]["drop_reason"] == "attachment_staging_failed", body

        # No event was appended — staging failure aborts before the
        # append-with-dedup step.
        events = await harness.events(session.id)
        user_events = [e for e in events if e.kind == "message" and e.data.get("role") == "user"]
        assert user_events == []
