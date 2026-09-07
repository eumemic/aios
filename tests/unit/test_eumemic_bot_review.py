"""The eumemic-bot review launcher: wait → read artifact → post → verify → archive.

The bug class these pin down is "the Action goes green and no `### Code review`
comment exists": every one of these tests fails loudly rather than silently
skipping publication.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

_SCRIPT = Path(__file__).parents[2] / "scripts" / "eumemic_bot_review.py"
_SPEC = importlib.util.spec_from_file_location("eumemic_bot_review", _SCRIPT)
assert _SPEC and _SPEC.loader
reviewer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reviewer)

_ARTIFACT = "### Code review\n\nLooks good."

# Checkout, token mint, the GitHub POST and the archive all live outside the
# review budget; the job timeout must cover them on top of the worst-case wait.
_JOB_OVERHEAD_SECONDS = 120


def _assistant(content: Any) -> dict[str, Any]:
    return {"kind": "message", "data": {"role": "assistant", "content": content}}


def test_review_from_events_reads_the_list_envelope(monkeypatch: Any) -> None:
    """``GET /events`` returns ``{"data": [...]}`` — an ``items`` read finds nothing."""
    # dir=backward pages newest-first: the artifact turn precedes the tool turn.
    events = {
        "data": [
            _assistant("  ### Code review\n\nPass.  "),
            _assistant("not artifact"),
        ],
        "has_more": False,
    }
    monkeypatch.setattr(reviewer, "_request", lambda *args, **kwargs: events)

    assert reviewer._review_from_events("https://aios.test", "key", "sess_1") == (
        "### Code review\n\nPass."
    )


def test_review_from_events_handles_content_blocks_and_a_preamble(monkeypatch: Any) -> None:
    """Content-part blocks and a chatty lead-in still yield a heading-first artifact."""
    events = {
        "data": [
            _assistant([{"type": "text", "text": "Here it is:\n\n### Code review\n\nPass."}]),
        ]
    }
    monkeypatch.setattr(reviewer, "_request", lambda *args, **kwargs: events)

    assert reviewer._review_from_events("https://aios.test", "key", "sess_1") == (
        "### Code review\n\nPass."
    )


def test_review_from_events_ignores_user_messages(monkeypatch: Any) -> None:
    """The launcher's own corrective prompt quotes the heading — never echo it back."""
    events = {"data": [{"kind": "message", "data": {"role": "user", "content": _ARTIFACT}}]}
    monkeypatch.setattr(reviewer, "_request", lambda *args, **kwargs: events)

    assert reviewer._review_from_events("https://aios.test", "key", "sess_1") is None


def test_wait_polls_until_the_session_stops_working(monkeypatch: Any) -> None:
    """An assistant tool-call turn is not the end of the review: keep waiting."""
    polls = iter(
        [
            {"session_status": "active", "next_after": 4},
            {"session_status": "active", "next_after": 9},
            {"session_status": "idle", "next_after": 12},
        ]
    )
    urls: list[str] = []

    def request(method: str, url: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        urls.append(url)
        return next(polls)

    monkeypatch.setattr(reviewer, "_request", request)

    status = reviewer._wait_until_working_stops(
        "https://aios.test", "key", "sess_1", reviewer.time.monotonic() + 60
    )
    assert status == "idle"
    # The cursor advances, so a busy session never re-reads the same event page.
    assert [u.split("?")[1] for u in urls] == [
        "after=0&timeout=30",
        "after=4&timeout=30",
        "after=9&timeout=30",
    ]


def test_launcher_matches_the_committed_api_contract() -> None:
    """Pin the launcher's request/response shapes to ``openapi.json``.

    The launcher is a plain-urllib client with no generated types, so nothing
    else notices when it reads a field the API does not emit — the whole class
    of "green Action, no comment" bug. ``openapi.json`` is CI-regenerated from
    FastAPI introspection, which makes it the honest contract to pin against.
    """
    spec = json.loads((_SCRIPT.parents[1] / "openapi.json").read_text())
    schemas = spec["components"]["schemas"]

    events = spec["paths"]["/v1/sessions/{session_id}/events"]["get"]
    assert {"dir", "kind", "limit"} <= {p["name"] for p in events["parameters"]}
    # ``_review_from_events`` reads the envelope's list: it is ``data``, never ``items``.
    assert "data" in schemas["ListResponse_Event_"]["properties"]

    wait = spec["paths"]["/v1/sessions/{session_id}/wait"]["get"]
    wait_params = {p["name"]: p["schema"] for p in wait["parameters"]}
    assert {"after", "timeout"} <= set(wait_params)
    # The server caps its long-poll; the socket deadline must outlive it.
    assert wait_params["timeout"]["maximum"] >= reviewer._WAIT_SECONDS
    assert reviewer._WAIT_HTTP_TIMEOUT > reviewer._WAIT_SECONDS
    assert {"session_status", "next_after"} <= set(schemas["WaitResponse"]["properties"])
    assert "active" in schemas["WaitResponse"]["properties"]["session_status"]["enum"]

    assert "archive_when_idle" in schemas["SessionCreate"]["properties"]
    assert "content" in schemas["SessionUserMessage"]["properties"]
    assert "/v1/sessions/{session_id}/archive" in spec["paths"]


def test_review_budget_fits_under_the_job_timeout() -> None:
    """The launcher must FATAL on its own budget BEFORE the runner kills the job.

    No test can tell that a budget is too SMALL -- that is what raising 1200 to
    2700 was for. What a test can hold is the ordering between the two clocks,
    which is precisely what raising one of them can break. Assert the property,
    not the literals: both numbers are expected to be re-tuned again.

    Order matters. The review step is ``continue-on-error``, so a launcher FATAL
    leaves the job green and still runs the "did not post" summary step. A job
    that hits ``timeout-minutes`` instead is killed outright: red check, and the
    summary step is skipped, which is exactly the silent failure this launcher
    exists to prevent. So the worst-case launcher wall clock -- the budget plus
    one final long-poll, which is issued just under the deadline and can hold
    the socket for ``_WAIT_HTTP_TIMEOUT`` -- must land clear of the job timeout
    with room for checkout, token mint, the GitHub POST, and the archive.
    """
    workflow = yaml.safe_load(
        (_SCRIPT.parents[1] / ".github/workflows/eumemic-bot-review.yml").read_text()
    )
    job = workflow["jobs"]["review"]
    step = next(s for s in job["steps"] if s.get("id") == "review")
    budget = int(step["env"]["REVIEW_TIMEOUT_SECONDS"])
    job_seconds = int(job["timeout-minutes"]) * 60

    worst_case = budget + reviewer._WAIT_HTTP_TIMEOUT
    assert worst_case + _JOB_OVERHEAD_SECONDS <= job_seconds, (
        f"REVIEW_TIMEOUT_SECONDS={budget} plus a {reviewer._WAIT_HTTP_TIMEOUT}s final long-poll "
        f"leaves under {_JOB_OVERHEAD_SECONDS}s of the {job_seconds}s job for checkout, token "
        f"mint and the GitHub POST: the runner would kill the job before the launcher can "
        f"report why"
    )

    # Two copies of one number: the workflow's env and the launcher's default. A
    # manual `python3 scripts/eumemic_bot_review.py` must get the CI budget too.
    assert budget == reviewer._REVIEW_SECONDS, (
        f"launcher default {reviewer._REVIEW_SECONDS}s and workflow env {budget}s "
        f"disagree on the review budget"
    )


def test_review_scope_avoids_repeating_ci_and_exhaustive_work() -> None:
    """The launcher must constrain the costly tool loops seen in live reviews."""
    guidance = reviewer.REVIEW_SCOPE
    assert "focused tests" in guidance
    assert "repository-wide test, lint, format, or type-check suites" in guidance
    assert "exhaustive ad hoc benchmarks" in guidance
    assert "substantive PR diff is unchanged" in guidance


def test_missing_artifact_gets_one_corrective_turn(monkeypatch: Any) -> None:
    reviews = iter([None, "### Code review\n\nFound on retry."])
    posts: list[tuple[Any, ...]] = []
    monkeypatch.setattr(reviewer, "_review_from_events", lambda *args: next(reviews))
    monkeypatch.setattr(reviewer, "_wait_until_working_stops", lambda *args: "idle")

    def request(*args: Any, **kwargs: Any) -> dict[str, Any]:
        posts.append(args)
        return {"seq": 17}

    monkeypatch.setattr(reviewer, "_request", request)

    assert reviewer._ask_for_review_artifact("https://aios.test", "key", "sess_1") == (
        "### Code review\n\nFound on retry."
    )
    assert [p[0:2] for p in posts] == [("POST", "https://aios.test/v1/sessions/sess_1/messages")]


def test_second_miss_is_fatal(monkeypatch: Any) -> None:
    monkeypatch.setattr(reviewer, "_review_from_events", lambda *args: None)
    monkeypatch.setattr(reviewer, "_wait_until_working_stops", lambda *args: "idle")
    monkeypatch.setattr(reviewer, "_request", lambda *args, **kwargs: {})

    with pytest.raises(SystemExit) as exc:
        reviewer._ask_for_review_artifact("https://aios.test", "key", "sess_1")
    assert exc.value.code == 1


class _Api:
    """Stand-in for the aios API + GitHub, recording the call order."""

    def __init__(self, events: dict[str, Any]) -> None:
        self.events = events
        self.calls: list[str] = []
        self.posted_body: str | None = None

    def request(
        self, method: str, url: str, api_key: str, body: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        path = url.split("/v1/")[1].split("?")[0]
        self.calls.append(f"{method} /v1/{path}")
        if path == "sessions":
            return {"id": "sess_1"}
        if path.endswith("/wait"):
            return {"session_status": "idle", "next_after": 3}
        if path.endswith("/events"):
            return self.events
        return {}

    def github(
        self, method: str, url: str, token: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        assert body is not None
        self.calls.append(f"{method} {url}")
        self.posted_body = body["body"]
        return {"html_url": "https://github.com/eumemic/aios/pull/1#issuecomment-1", **body}


@pytest.fixture
def launcher_env(monkeypatch: Any) -> None:
    for key, value in {
        "AIOS_URL": "https://aios.test",
        "AIOS_API_KEY": "key",
        "GH_TOKEN": "ghs_token",
        "REPO": "eumemic/aios",
        "PR_NUMBER": "2384",
        "HEAD_SHA": "9143ff54",
        "CLONE_URL": "https://github.com/eumemic/aios.git",
        "AGENT_ID": "agt_1",
        "ENVIRONMENT_ID": "env_1",
    }.items():
        monkeypatch.setenv(key, value)


def test_main_posts_the_artifact_then_archives(
    monkeypatch: Any, launcher_env: None, capsys: Any
) -> None:
    api = _Api({"data": [_assistant(_ARTIFACT)]})
    monkeypatch.setattr(reviewer, "_request", api.request)
    monkeypatch.setattr(reviewer, "_github_request", api.github)

    reviewer.main()

    assert api.calls == [
        "POST /v1/sessions",
        "GET /v1/sessions/sess_1/wait",
        "GET /v1/sessions/sess_1/events",
        "POST https://api.github.com/repos/eumemic/aios/issues/2384/comments",
        "POST /v1/sessions/sess_1/archive",
    ]
    assert api.posted_body is not None
    assert api.posted_body.startswith("### Code review")
    # The marker is what the verification round-trip asserts on.
    assert "<!-- eumemic-bot-review:9143ff54 -->" in api.posted_body
    assert "posted and verified ### Code review: https://github.com/" in capsys.readouterr().out


def test_main_archives_even_when_the_artifact_never_arrives(
    monkeypatch: Any, launcher_env: None
) -> None:
    """A failed review must not leak a live session — archive_when_idle is off now."""
    api = _Api({"data": [_assistant("I could not review this.")]})
    monkeypatch.setattr(reviewer, "_request", api.request)
    monkeypatch.setattr(reviewer, "_github_request", api.github)

    with pytest.raises(SystemExit) as exc:
        reviewer.main()

    assert exc.value.code == 1
    assert api.posted_body is None
    assert api.calls[-1] == "POST /v1/sessions/sess_1/archive"


def test_main_fails_when_github_does_not_confirm(monkeypatch: Any, launcher_env: None) -> None:
    """A comment GitHub did not store must never read as a successful review."""
    api = _Api({"data": [_assistant(_ARTIFACT)]})
    monkeypatch.setattr(reviewer, "_request", api.request)
    monkeypatch.setattr(reviewer, "_github_request", lambda *args, **kwargs: {"message": "Bad"})

    with pytest.raises(SystemExit) as exc:
        reviewer.main()

    assert exc.value.code == 1
    assert api.calls[-1] == "POST /v1/sessions/sess_1/archive"
