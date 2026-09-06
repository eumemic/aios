from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

_SCRIPT = Path(__file__).parents[2] / "scripts" / "eumemic_bot_review.py"
_SPEC = importlib.util.spec_from_file_location("eumemic_bot_review", _SCRIPT)
assert _SPEC and _SPEC.loader
reviewer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reviewer)


def test_review_from_events_selects_code_review_assistant(monkeypatch: Any) -> None:
    events = {
        "items": [
            {"kind": "message", "data": {"role": "assistant", "content": "not artifact"}},
            {
                "kind": "message",
                "data": {"role": "assistant", "content": "  ### Code review\n\nPass.  "},
            },
        ]
    }
    monkeypatch.setattr(reviewer, "_request", lambda *args, **kwargs: events)

    assert reviewer._review_from_events("https://aios.test", "key", "sess_1") == (
        "### Code review\n\nPass."
    )


def test_missing_artifact_gets_one_corrective_turn(monkeypatch: Any) -> None:
    reviews = iter([None, "### Code review\n\nFound on retry."])
    requests: list[tuple[Any, ...]] = []
    waits: list[int | None] = []
    monkeypatch.setattr(reviewer, "_review_from_events", lambda *args: next(reviews))
    monkeypatch.setattr(
        reviewer, "_await_turn", lambda *args, watermark=None: waits.append(watermark)
    )

    def request(*args: Any, **kwargs: Any) -> dict[str, Any]:
        requests.append(args)
        return {"seq": 17}

    monkeypatch.setattr(reviewer, "_request", request)

    assert reviewer._ask_for_review_artifact("https://aios.test", "key", "sess_1") == (
        "### Code review\n\nFound on retry."
    )
    assert waits == [None, 17]
    assert requests[0][0:2] == ("POST", "https://aios.test/v1/sessions/sess_1/messages")


def test_launcher_contract_keeps_session_live_until_verified() -> None:
    source = _SCRIPT.read_text()
    assert '"archive_when_idle": False' in source
    assert "posted and verified Code review" in source
    assert source.index("posted and verified Code review") < source.index(
        'f"{base}/v1/sessions/{sid}/archive"'
    )
