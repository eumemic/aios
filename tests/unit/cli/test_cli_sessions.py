"""Tests for ``aios sessions ...`` destructive verbs via the typer app.

Focuses on the ``delete`` guard (hard-delete requires ``--yes``) introduced
to line ``sessions delete`` up with the other hard-delete commands. Happy-
path CRUD is exercised elsewhere.
"""

from __future__ import annotations

import httpx
from typer.testing import CliRunner

from aios.cli.app import app

runner = CliRunner()


def test_delete_refuses_without_yes_and_makes_no_request(mocked_cli):
    result = runner.invoke(app, ["sessions", "delete", "sess_1"])
    assert result.exit_code == 2
    assert "--yes" in result.output
    assert "archive" in result.output  # remind about the soft alternative
    assert mocked_cli.captured.method == ""  # no HTTP call was made


def test_delete_is_hard_delete_with_yes(mocked_cli):
    mocked_cli.queue_response(httpx.Response(204))
    result = runner.invoke(app, ["sessions", "delete", "sess_1", "--yes"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "DELETE"
    assert mocked_cli.captured.path == "/v1/sessions/sess_1"
    # Confirmation line on stdout — "deleted <id>" gives scripts something to
    # grep and humans a visible ack (the server returns 204 with no body).
    assert "deleted" in result.output
    assert "sess_1" in result.output


def _profile_events_page() -> dict:
    """Minimal events envelope with a single complete step, enough for the
    profiler to emit every section."""
    return {
        "data": [
            {
                "id": "ev_1",
                "seq": 1,
                "kind": "span",
                "data": {"event": "step_start", "cause": "message"},
                "created_at": "2026-04-21T12:00:00.000000",
            },
            {
                "id": "ev_2",
                "seq": 2,
                "kind": "span",
                "data": {"event": "sweep_start", "site": "entry"},
                "created_at": "2026-04-21T12:00:00.010000",
            },
            {
                "id": "ev_3",
                "seq": 3,
                "kind": "span",
                "data": {
                    "event": "sweep_end",
                    "sweep_start_id": "ev_2",
                    "repaired_ghosts": 0,
                    "woken_sessions": 1,
                },
                "created_at": "2026-04-21T12:00:00.012000",
            },
            {
                "id": "ev_4",
                "seq": 4,
                "kind": "span",
                "data": {"event": "context_build_start"},
                "created_at": "2026-04-21T12:00:00.015000",
            },
            {
                "id": "ev_5",
                "seq": 5,
                "kind": "span",
                "data": {"event": "context_build_end", "context_build_start_id": "ev_4"},
                "created_at": "2026-04-21T12:00:00.017000",
            },
            {
                "id": "ev_6",
                "seq": 6,
                "kind": "span",
                "data": {"event": "model_request_start"},
                "created_at": "2026-04-21T12:00:00.020000",
            },
            {
                "id": "ev_7",
                "seq": 7,
                "kind": "span",
                "data": {"event": "model_request_end", "model_request_start_id": "ev_6"},
                "created_at": "2026-04-21T12:00:01.020000",
            },
            {
                "id": "ev_8",
                "seq": 8,
                "kind": "span",
                "data": {"event": "step_end", "step_start_id": "ev_1"},
                "created_at": "2026-04-21T12:00:01.025000",
            },
        ],
        "has_more": False,
        "next_after": None,
    }


def test_profile_table_output(mocked_cli):
    mocked_cli.queue_response(httpx.Response(200, json=_profile_events_page()))
    result = runner.invoke(app, ["sessions", "profile", "sess_1"])
    assert result.exit_code == 0, result.output
    assert mocked_cli.captured.method == "GET"
    assert mocked_cli.captured.path == "/v1/sessions/sess_1/events"
    assert mocked_cli.captured.query.get("kind") == ["span"]
    # Table output must have the three headline sections the user will scan for.
    assert "INSIDE-STEP" in result.output
    assert "WALL-CLOCK" in result.output
    assert "model_request" in result.output


def test_profile_json_output_is_machine_readable(mocked_cli):
    import json

    mocked_cli.queue_response(httpx.Response(200, json=_profile_events_page()))
    result = runner.invoke(app, ["--format", "json", "sessions", "profile", "sess_1"])
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert parsed["n_steps"] == 1
    assert any(p["phase"] == "model_request" for p in parsed["inside"])


def test_profile_empty_session_gracefully(mocked_cli):
    mocked_cli.queue_response(
        httpx.Response(200, json={"data": [], "has_more": False, "next_after": None})
    )
    result = runner.invoke(app, ["sessions", "profile", "sess_empty"])
    assert result.exit_code == 0, result.output
    assert "no steps" in result.output
