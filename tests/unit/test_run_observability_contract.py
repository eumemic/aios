"""Run-observability contract guards (issue #1140).

These assert the documented contract for three #1068 dogfood papercuts:

(a) The run ``/wait`` terminal state is carried by ``run_status``/``done`` —
    there is NO ``state`` field, and the model/schema must say so explicitly so
    a watcher does not key on a nonexistent ``.state`` and wait forever.
(b) ``GET /v1/sessions/{id}/events`` (and the run-events twin) can return an
    empty page transiently; the endpoint docs must tell consumers an empty list
    is a paging artifact, not a "session reset."
(c) The two coexisting event schemas (run ``{type,payload,seq}`` vs
    child-session ``{kind,data}``) must be documented.

Pure in-memory: no Postgres, no Docker. Schema assertions walk OpenAPI
metadata, which is exactly what consumers (SDK/MCP) read.
"""

from __future__ import annotations

from typing import Any


def _openapi() -> dict[str, Any]:
    # Deferred import: app import runs get_settings() at import time; the
    # conftest env fixture must fire first (see test_openapi_snapshot.py).
    from aios.api.app import create_app

    return create_app().openapi()


class TestWaitTerminalStateField:
    def test_wait_response_has_no_state_field(self) -> None:
        from aios.models.workflows import WfRunWaitResponse

        # The watcher in #1068 keyed on ``.state`` and waited forever. Guard
        # that the field genuinely does not exist (so the fix is to document
        # run_status/done, not to silently add a state alias).
        assert "state" not in WfRunWaitResponse.model_fields
        assert "run_status" in WfRunWaitResponse.model_fields
        assert "done" in WfRunWaitResponse.model_fields

    def test_wait_response_fields_describe_terminal_polling(self) -> None:
        from aios.models.workflows import WfRunWaitResponse

        run_status_desc = (WfRunWaitResponse.model_fields["run_status"].description or "").lower()
        done_desc = (WfRunWaitResponse.model_fields["done"].description or "").lower()
        # The terminal-state field must be self-describing in the schema.
        assert "terminal" in run_status_desc or "status" in run_status_desc
        assert "no" in done_desc and "state" in done_desc

    def test_wfrun_status_field_is_documented(self) -> None:
        from aios.models.workflows import WfRun

        desc = (WfRun.model_fields["status"].description or "").lower()
        assert "status" in desc
        assert "state" in desc  # explicitly disambiguates status-vs-state

    def test_openapi_wait_schema_documents_status_not_state(self) -> None:
        schema = _openapi()["components"]["schemas"]["WfRunWaitResponse"]
        props = schema["properties"]
        assert "state" not in props
        assert "run_status" in props and props["run_status"].get("description")


class TestTransientEmptyEventsDocumented:
    def test_session_events_op_warns_about_transient_empty(self) -> None:
        op = _openapi()["paths"]["/v1/sessions/{session_id}/events"]["get"]
        desc = (op.get("description", "") + op.get("summary", "")).lower()
        assert "empty" in desc
        assert "reset" in desc

    def test_run_events_op_warns_about_transient_empty(self) -> None:
        op = _openapi()["paths"]["/v1/runs/{run_id}/events"]["get"]
        desc = (op.get("description", "") + op.get("summary", "")).lower()
        assert "empty" in desc
        assert "reset" in desc


class TestDualEventSchemasDocumented:
    def test_event_model_docstring_names_kind_data_shape(self) -> None:
        from aios.models.events import Event

        doc = (Event.__doc__ or "").lower()
        assert "kind" in doc and "data" in doc

    def test_wfrunevent_model_docstring_names_type_payload_shape(self) -> None:
        from aios.models.workflows import WfRunEvent

        doc = (WfRunEvent.__doc__ or "").lower()
        assert "type" in doc and "payload" in doc

    def test_reference_doc_exists(self) -> None:
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        doc = repo_root / "docs" / "reference" / "run-observability.md"
        assert doc.exists(), "expected docs/reference/run-observability.md"
        text = doc.read_text().lower()
        # Must cover all three papercuts.
        assert "run_status" in text and "state" in text
        assert "empty" in text
        assert "{type" in text or "type, payload" in text or "type,payload" in text
