"""E2E test: the post-reparent inbound surfaces over HTTP as 422
``drop_reason=detached`` (not 404 ``session_missing``).

The sibling unit/integration coverage
(``tests/integration/test_resolver_reparent_cross_account.py``) drives
``handle_inbound`` directly at the service layer. This module drives the
**full HTTP path** — a multipart POST to ``/v1/connectors/runtime/inbound``
against a live uvicorn server — so the drop-reason → HTTP-status mapping in
``aios.api.routers.connectors._inbound_drop_error`` (DETACHED → 422,
SESSION_MISSING → 404) is exercised end-to-end against the real FastAPI
error envelope, not just the service-layer enum.

It is also the closest automated proxy to the operator-facing repro in the
bug report (reparent a connection with a pre-reparent ``chat_sessions``
ledger row, then send an inbound for that chat authenticated as the
destination account). The report's "daemon-cache caveat (v1)" step
(restart the connector on the destination key) is modelled here by issuing
a fresh runtime token scoped to the destination account and POSTing with
it — that token's ``account_id`` is the inbound route's authenticated scope.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from aios.config import get_settings
from aios.crypto.vault import CryptoBox
from aios.db import queries
from aios.services import agents as agents_service
from aios.services import connections as connections_service
from aios.services import environments as env_svc
from aios.services import sessions as sess_svc
from tests.conftest import needs_docker
from tests.e2e.conftest import live_aios_server
from tests.helpers.connections import (
    admit_inbound_all,
    asgi_client,
    bearer,
    create_connection,
    mint_runtime_token_via_db,
)
from tests.integration.conftest import seed_agent_env_session

pytestmark = pytest.mark.docker


@pytest.fixture
async def live_server(aios_env: dict[str, str]) -> AsyncIterator[str]:
    """Run uvicorn on a free port, serving the aios app (mirrors
    ``test_echo_http_connector.py``'s fixture)."""
    async with live_aios_server() as url:
        yield url


async def _mint_child(http_client: httpx.AsyncClient, root_key: str, name: str) -> tuple[str, str]:
    r = await http_client.post(
        "/v1/accounts/children",
        headers=bearer(root_key),
        json={"display_name": name, "can_mint_children": False},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    return body["account_id"], body["plaintext_key"]


async def _post_inbound(
    client: httpx.AsyncClient,
    token: str,
    *,
    connection_id: str,
    event_id: str,
    chat_id: str,
    content: str = "hello",
) -> httpx.Response:
    files: list[tuple[str, tuple[str | None, bytes, str]]] = [
        ("connection_id", (None, connection_id.encode(), "text/plain")),
        ("event_id", (None, event_id.encode(), "text/plain")),
        ("chat_id", (None, chat_id.encode(), "text/plain")),
        ("content", (None, content.encode(), "text/plain")),
        (
            "sender",
            (None, json.dumps({"id": chat_id, "display_name": "Alice"}).encode(), "text/plain"),
        ),
    ]
    return await client.post(
        "/v1/connectors/runtime/inbound",
        headers=bearer(token),
        files=files,
    )


@needs_docker
class TestReparentCrossAccountInboundHttp:
    """Drive the post-reparent inbound over a real HTTP socket."""

    async def test_post_reparent_inbound_is_422_detached_not_404_session_missing(
        self,
        harness: Any,
        live_server: str,
        aios_env: dict[str, str],
    ) -> None:
        """A previously-routed chat on a connection reparented from acc_a to
        acc_b must, when an inbound arrives authenticated as acc_b, surface
        as HTTP 422 with ``drop_reason=detached`` (not 404
        ``session_missing``). Pre-fix the resolver passed the cross-account
        ``session_id`` to ``append_event`` → ``NotFoundError`` →
        ``SESSION_MISSING`` → 404; the runner treats 404 as non-fatal and
        drops/acks, masking the reparent as the cause."""
        pool = harness._pool
        crypto_box = CryptoBox.from_base64(get_settings().vault_key.get_secret_value())
        root_key = aios_env["AIOS_API_KEY"]
        async with asgi_client(pool) as client:
            acc_a, _key_a = await _mint_child(client, root_key, "tenant-a")
            acc_b, _key_b = await _mint_child(client, root_key, "tenant-b")

        _agent, _env, session = await seed_agent_env_session(
            pool, account_id=acc_a, prefix="e2e-xacc"
        )
        connection = await connections_service.create_connection(
            pool,
            account_id=acc_a,
            connector="echo",
            external_account_id="e2e-xacc-1",
            metadata={},
            crypto_box=crypto_box,
        )
        chat_id = "chat-pre-reparent"
        async with pool.acquire() as conn:
            await queries.insert_chat_session(
                conn,
                connection_id=connection.id,
                chat_id=chat_id,
                session_id=session.id,
                account_id=acc_a,
            )
        await admit_inbound_all(pool, connection.id)

        # ``acc_test_stub`` is the bootstrapped root the e2e fixtures seed
        # (``parent_account_id IS NULL``); the reparent service's root gate
        # requires the requester to be root.
        await connections_service.reparent_connection(
            pool,
            connection.id,
            destination_account_id=acc_b,
            requester_account_id="acc_test_stub",
            crypto_box=crypto_box,
        )

        dest_token = await mint_runtime_token_via_db(pool, connector="echo", account_id=acc_b)

        async with httpx.AsyncClient(base_url=live_server, timeout=30.0) as http:
            r = await _post_inbound(
                http,
                dest_token,
                connection_id=connection.id,
                event_id="e2e-evt-cross-account-1",
                chat_id=chat_id,
            )
        assert r.status_code == 422, r.text
        body = r.json()
        assert body["error"]["detail"]["drop_reason"] == "detached", r.text

    async def test_live_same_account_inbound_delivers_over_http(
        self,
        harness: Any,
        live_server: str,
        aios_env: dict[str, str],
    ) -> None:
        """Control / regression guard over HTTP: a LIVE session owned by the
        SAME account as the inbound's auth must deliver (201 + an
        ``appended_event_id``), confirming the resolver fix did not flip the
        live-same-account case over the HTTP path."""
        pool = harness._pool
        account_id = "acc_test_stub"

        agent = await agents_service.create_agent(
            pool,
            name=f"e2e-live-{id(self)}",
            model="fake/test",
            system="",
            tools=[],
            description=None,
            metadata={},
            window_min=50_000,
            window_max=150_000,
            account_id=account_id,
        )
        env = await env_svc.create_environment(
            pool, name=f"env-live-{id(self)}", account_id=account_id
        )
        session = await sess_svc.create_session(
            pool,
            agent_id=agent.id,
            environment_id=env.id,
            title=None,
            metadata={},
            account_id=account_id,
        )
        root_key = aios_env["AIOS_API_KEY"]
        connection_id = await create_connection(root_key, live_server, f"acct-live-{id(self)}")
        await connections_service.attach_connection(
            pool, connection_id, session_id=session.id, account_id=account_id
        )
        await admit_inbound_all(pool, connection_id)

        token = await mint_runtime_token_via_db(pool, connector="echo", account_id=account_id)

        async with httpx.AsyncClient(base_url=live_server, timeout=30.0) as http:
            r = await _post_inbound(
                http,
                token,
                connection_id=connection_id,
                event_id="e2e-evt-live-1",
                chat_id="chat-live",
            )
        assert r.status_code == 201, r.text
        assert r.json()["appended_event_id"] is not None, r.text
