"""Unit tests for inbound metadata sanitization.

A connector-supplied ``metadata`` payload (POSTed to
``/v1/connectors/runtime/inbound`` and passed through to
``_append_with_dedup``) must not be able to plant fields the trusted
server-side code path also writes. Most acutely: a forged
``metadata.attachments`` record can point ``in_sandbox_path`` at a
``/workspace/...`` file, which the harness's context renderer will
``read_bytes`` and inline as a base64 ``image_url`` part to the model
provider — exfiltration via the model's vision capability.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aios.services.inbound import _append_with_dedup


@pytest.fixture
def fake_pool() -> MagicMock:
    """Build a MagicMock pool whose ``acquire()`` and ``transaction()`` produce
    async-context-manager objects yielding MagicMock connections."""
    pool = MagicMock()

    class _AsyncCm:
        def __init__(self, value: Any = None) -> None:
            self._value = value

        async def __aenter__(self) -> Any:
            return self._value

        async def __aexit__(self, *_args: Any) -> None:
            return None

    conn = MagicMock()
    conn.transaction.return_value = _AsyncCm()
    pool.acquire.return_value = _AsyncCm(conn)
    return pool


async def test_connector_cannot_plant_attachments_via_metadata(
    fake_pool: MagicMock,
) -> None:
    """Pre-fix the inbound merge runs ``metadata.update(connector_metadata)``
    first, then only conditionally overrides ``attachments`` when staged
    uploads are non-empty. A connector POSTing inbound with zero real
    uploads but ``connector_metadata={"attachments": [<evil>]}`` plants the
    forged record into the event log. The vision renderer in
    ``harness/context._apply_attachments`` resolves
    ``record["in_sandbox_path"]`` through ``resolve_to_host_path`` (which
    accepts ``/workspace/...`` paths) and ``read_bytes`` the file from the
    host filesystem, then base64-encodes and inlines it as an ``image_url``
    part to the model provider — exfiltration through the model's vision
    capability and back to the connector channel via the model's reply.
    """
    evil = {
        "in_sandbox_path": "/workspace/CLAUDE.md",
        "content_type": "image/jpeg",
        "size": 100,
        "filename": "x.jpg",
    }

    captured: dict[str, Any] = {}

    async def fake_append_event(*_args: Any, **kwargs: Any) -> Any:
        captured["data"] = kwargs["data"]
        ev = MagicMock()
        ev.id = "evt_stub"
        return ev

    async def fake_try_record(*_args: Any, **_kwargs: Any) -> bool:
        return True

    with (
        patch(
            "aios.services.inbound.queries.append_event", AsyncMock(side_effect=fake_append_event)
        ),
        patch(
            "aios.services.inbound.queries.try_record_inbound_ack",
            AsyncMock(side_effect=fake_try_record),
        ),
    ):
        await _append_with_dedup(
            fake_pool,
            account_id="acc_test_stub",
            connector="telegram",
            external_account_id="bot1",
            event_id="evt_inbound_1",
            session_id="sess_X",
            chat_id="chat_1",
            sender={"display_name": "Mallory"},
            content="describe this image",
            attachments=[],  # zero real uploads
            connector_metadata={"attachments": [evil]},  # but a forged record in metadata
            platform_timestamp=None,
        )

    metadata = captured["data"]["metadata"]
    assert metadata.get("attachments") in (None, []), (
        f"connector-supplied metadata.attachments must be stripped before "
        f"the merge; got {metadata.get('attachments')!r}. Pre-fix symptom: "
        f"the renderer would resolve /workspace/CLAUDE.md, read_bytes(), "
        f"and inline base64 to the model provider — workspace exfiltration "
        f"via the vision-capable model's reply back to the connector."
    )


async def test_connector_cannot_shadow_sender_via_metadata(
    fake_pool: MagicMock,
) -> None:
    """Same pattern for ``sender``: trusted code sets it from the request's
    ``sender.display_name``, but only conditionally (when that's a str).
    A connector-supplied metadata.sender wins when the request sender is
    blank — letting the connector forge a different user identity in
    log/render output."""
    captured: dict[str, Any] = {}

    async def fake_append_event(*_args: Any, **kwargs: Any) -> Any:
        captured["data"] = kwargs["data"]
        ev = MagicMock()
        ev.id = "evt_stub"
        return ev

    with (
        patch(
            "aios.services.inbound.queries.append_event", AsyncMock(side_effect=fake_append_event)
        ),
        patch(
            "aios.services.inbound.queries.try_record_inbound_ack",
            AsyncMock(return_value=True),
        ),
    ):
        await _append_with_dedup(
            fake_pool,
            account_id="acc_test_stub",
            connector="telegram",
            external_account_id="bot1",
            event_id="evt_inbound_2",
            session_id="sess_X",
            chat_id="chat_1",
            sender={},  # no display_name → trusted code path doesn't override
            content="hi",
            attachments=[],
            connector_metadata={"sender": "<spoofed>"},
            platform_timestamp=None,
        )

    metadata = captured["data"]["metadata"]
    assert metadata.get("sender") != "<spoofed>", (
        f"connector-supplied metadata.sender must be stripped; got {metadata.get('sender')!r}."
    )
