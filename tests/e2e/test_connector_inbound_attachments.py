"""E2E coverage for inbound attachment staging.

Drives ``ConnectorSubprocessRegistry._handle_inbound`` directly (no
real subprocess) and verifies:

* a connector temp file is staged into
  ``<workspace_root>/_attachments/<session>/<connector>/<event-ulid>-<filename>``
* the appended event carries structured ``metadata.attachments``
  records with the model-facing ``in_sandbox_path``
* a redelivered inbound (same ``event_id``) does not double-stage
* the worker startup orphan GC sweep removes stranded files but
  leaves referenced ones alone
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from aios.config import get_settings
from aios.harness.attachment_gc import sweep_orphan_attachments
from aios.ids import make_id
from aios.sandbox.volumes import session_attachments_dir
from aios.services import (
    connections as connections_service,
)
from aios.services import (
    sessions as sessions_service,
)
from tests.conftest import needs_docker
from tests.e2e._connector_helpers import (
    make_agent_and_env,
    make_registry,
    patch_send_ack,
    unique_account,
)
from tests.e2e.harness import Harness


def _make_connector_temp(tmp_path: Path, name: str, payload: bytes) -> dict[str, Any]:
    """Write a fake connector-side temp file and return the wire-shaped record."""
    src = tmp_path / name
    src.write_bytes(payload)
    return {
        "host_path": str(src),
        "filename": name,
        "content_type": "image/jpeg",
        "size": len(payload),
    }


@needs_docker
class TestInboundAttachmentStaging:
    async def test_attachment_lands_at_staged_path_with_metadata(
        self, harness: Harness, tmp_path: Path
    ) -> None:
        """Happy path: inbound with one attachment.

        Asserts (a) the host file exists at the expected staged path,
        (b) the event row carries structured metadata.attachments with
        the in_sandbox_path the renderer will expose to the model.
        """
        agent_id, env_id = await make_agent_and_env(harness, prefix="attach")
        account = unique_account()
        session = await sessions_service.create_session(
            harness._pool,
            agent_id=agent_id,
            environment_id=env_id,
            title="attach-1",
            metadata={},
        )
        conn_row = await connections_service.create_connection(
            harness._pool, connector="echo", account=account, metadata={}
        )
        await connections_service.attach_connection(
            harness._pool, conn_row.id, session_id=session.id
        )

        connector_tmp = tmp_path / "connector-temp"
        connector_tmp.mkdir()
        att = _make_connector_temp(connector_tmp, "photo.jpg", b"jpegbytes")
        event_id = make_id("evt")

        registry = make_registry()
        patch_send_ack(registry)
        with mock.patch(
            "aios.harness.connector_supervisor.defer_wake",
            new=mock.AsyncMock(return_value=None),
        ):
            await registry._handle_inbound(
                ("echo", "echo"),
                {
                    "event_id": event_id,
                    "account": account,
                    "chat_id": "chat-x",
                    "sender": {"display_name": "Alice"},
                    "content": "look at this",
                    "attachments": [att],
                },
            )

        # File staged with ULID prefix; connector temp consumed.
        staged = session_attachments_dir(session.id) / "echo" / f"{event_id}-photo.jpg"
        assert staged.exists()
        assert staged.read_bytes() == b"jpegbytes"
        # Connector temp dir is empty: rename consumed the source file.
        assert list(connector_tmp.iterdir()) == []

        # Event row carries the structured record.
        events = await harness.events(session.id)
        user_messages = [e for e in events if e.kind == "message" and e.data.get("role") == "user"]
        latest = user_messages[-1]
        attachments = latest.data["metadata"]["attachments"]
        assert attachments == [
            {
                "filename": "photo.jpg",
                "content_type": "image/jpeg",
                "size": len(b"jpegbytes"),
                "in_sandbox_path": f"/mnt/attachments/echo/{event_id}-photo.jpg",
            }
        ]
        # No drop counter bumped.
        assert dict(registry._states[("echo", "echo")].drops) == {}

    async def test_dedup_replay_does_not_double_stage(
        self, harness: Harness, tmp_path: Path
    ) -> None:
        """Replaying the same event_id twice produces one event row and no
        duplicate file. The second delivery's staging is a no-op (target
        already exists from the first attempt) and the dedup ledger
        rolls the second event back."""
        agent_id, env_id = await make_agent_and_env(harness, prefix="attach")
        account = unique_account()
        session = await sessions_service.create_session(
            harness._pool,
            agent_id=agent_id,
            environment_id=env_id,
            title="attach-dedup",
            metadata={},
        )
        conn_row = await connections_service.create_connection(
            harness._pool, connector="echo", account=account, metadata={}
        )
        await connections_service.attach_connection(
            harness._pool, conn_row.id, session_id=session.id
        )

        connector_tmp = tmp_path / "connector-temp"
        connector_tmp.mkdir()
        att1 = _make_connector_temp(connector_tmp, "photo.jpg", b"first-bytes")
        event_id = make_id("evt")

        registry = make_registry()
        patch_send_ack(registry)
        with mock.patch(
            "aios.harness.connector_supervisor.defer_wake",
            new=mock.AsyncMock(return_value=None),
        ):
            params = {
                "event_id": event_id,
                "account": account,
                "chat_id": "chat-dedup",
                "sender": {"display_name": "Bob"},
                "content": "hello",
                "attachments": [att1],
            }
            await registry._handle_inbound(("echo", "echo"), params)

            # Second delivery: temp file is gone (consumed by first
            # rename), but the params still reference it. The replay
            # path must short-circuit on the existing target file
            # rather than crashing on a missing temp.
            params2 = dict(params)
            params2["attachments"] = [dict(att1)]  # same fields, same host_path
            await registry._handle_inbound(("echo", "echo"), params2)

        # Exactly one user event with this content.
        events = await harness.events(session.id)
        matching = [e for e in events if e.kind == "message" and e.data.get("content") == "hello"]
        assert len(matching) == 1

        # Exactly one staged file with the expected payload.
        connector_stage = session_attachments_dir(session.id) / "echo"
        files = list(connector_stage.iterdir())
        assert len(files) == 1
        assert files[0].name == f"{event_id}-photo.jpg"
        assert files[0].read_bytes() == b"first-bytes"

    async def test_missing_temp_path_drops_inbound(self, harness: Harness, tmp_path: Path) -> None:
        """A connector that lies about its host path doesn't crash the
        supervisor — drop with the canonical reason and ack the spool."""
        agent_id, env_id = await make_agent_and_env(harness, prefix="attach")
        account = unique_account()
        session = await sessions_service.create_session(
            harness._pool,
            agent_id=agent_id,
            environment_id=env_id,
            title="attach-bogus",
            metadata={},
        )
        conn_row = await connections_service.create_connection(
            harness._pool, connector="echo", account=account, metadata={}
        )
        await connections_service.attach_connection(
            harness._pool, conn_row.id, session_id=session.id
        )

        bogus = {
            "host_path": str(tmp_path / "definitely-not-here.jpg"),
            "filename": "ghost.jpg",
            "content_type": "image/jpeg",
            "size": 99,
        }

        registry = make_registry()
        patch_send_ack(registry)
        with mock.patch(
            "aios.harness.connector_supervisor.defer_wake",
            new=mock.AsyncMock(return_value=None),
        ):
            await registry._handle_inbound(
                ("echo", "echo"),
                {
                    "event_id": make_id("evt"),
                    "account": account,
                    "chat_id": "chat-bogus",
                    "sender": {"display_name": "Carol"},
                    "content": "ghost message",
                    "attachments": [bogus],
                },
            )

        # No event row appended (drop happens before transaction).
        events = await harness.events(session.id)
        ghost_messages = [
            e for e in events if e.kind == "message" and e.data.get("content") == "ghost message"
        ]
        assert ghost_messages == []
        # Drop counter bumped.
        assert dict(registry._states[("echo", "echo")].drops).get("attachment_staging_failed") == 1


@needs_docker
class TestOrphanAttachmentGc:
    async def test_sweep_removes_stranded_files_keeps_referenced(
        self, harness: Harness, tmp_path: Path
    ) -> None:
        """An orphan file (no event row referencing its in_sandbox_path)
        is deleted; a referenced file is left alone."""
        agent_id, env_id = await make_agent_and_env(harness, prefix="attach")
        account = unique_account()
        session = await sessions_service.create_session(
            harness._pool,
            agent_id=agent_id,
            environment_id=env_id,
            title="attach-gc",
            metadata={},
        )
        conn_row = await connections_service.create_connection(
            harness._pool, connector="echo", account=account, metadata={}
        )
        await connections_service.attach_connection(
            harness._pool, conn_row.id, session_id=session.id
        )

        # First, drive a real inbound with attachment so we have a
        # referenced file on disk + matching event row.
        connector_tmp = tmp_path / "connector-temp"
        connector_tmp.mkdir()
        att = _make_connector_temp(connector_tmp, "kept.jpg", b"keep-me")
        event_id = make_id("evt")

        registry = make_registry()
        patch_send_ack(registry)
        with mock.patch(
            "aios.harness.connector_supervisor.defer_wake",
            new=mock.AsyncMock(return_value=None),
        ):
            await registry._handle_inbound(
                ("echo", "echo"),
                {
                    "event_id": event_id,
                    "account": account,
                    "chat_id": "chat-gc",
                    "sender": {"display_name": "Dave"},
                    "content": "msg",
                    "attachments": [att],
                },
            )

        # Manually drop an orphan into the same dir, with a fake event-id
        # prefix that has no corresponding event row.
        orphan_path = session_attachments_dir(session.id) / "echo" / f"{make_id('evt')}-orphan.jpg"
        orphan_path.write_bytes(b"orphan-bytes")
        kept_path = session_attachments_dir(session.id) / "echo" / f"{event_id}-kept.jpg"
        assert kept_path.exists()
        assert orphan_path.exists()

        deleted = await sweep_orphan_attachments(harness._pool)

        assert deleted == 1
        assert not orphan_path.exists()
        assert kept_path.exists(), "referenced file must survive the sweep"

    async def test_sweep_no_root_dir_returns_zero(
        self, harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When _attachments/ doesn't exist yet, the sweep is a no-op."""
        settings = get_settings()
        # Point at a dir that doesn't exist.
        monkeypatch.setattr(settings, "workspace_root", tmp_path / "nonexistent")
        deleted = await sweep_orphan_attachments(harness._pool)
        assert deleted == 0
