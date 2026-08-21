from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


def _load_module() -> Any:
    path = Path(__file__).parents[2] / "scripts" / "detect_pr_conflicts.py"
    spec = importlib.util.spec_from_file_location("detect_pr_conflicts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flags_remaining_pr_after_competing_pr_merges() -> None:
    detector = _load_module()
    comments: list[tuple[int, str]] = []

    class GitHub:
        def get(self, path: str) -> Any:
            if path == "/repos/eumemic/aios/pulls?state=open&base=master&per_page=100&page=1":
                return [{"number": 2, "draft": False}]
            if path == "/repos/eumemic/aios/pulls?state=open&base=master&per_page=100&page=2":
                return []
            if path == "/repos/eumemic/aios/pulls/2":
                return {"number": 2, "draft": False, "mergeable": False, "mergeable_state": "dirty"}
            raise AssertionError(path)

        def post(self, path: str, body: dict[str, str]) -> Any:
            assert path == "/repos/eumemic/aios/issues/2/comments"
            comments.append((2, body["body"]))
            return {}

    conflicted = detector.detect_conflicts(GitHub(), "eumemic/aios", "master", "abc123")

    assert conflicted == [2]
    assert comments == [
        (
            2,
            "<!-- aios-conflict-notice:abc123 -->\n"
            "CONFLICTED: master `abc123` cannot be merged into this branch cleanly. "
            "Rebase the branch before review or fix-round work continues.",
        )
    ]


def test_persistently_unknown_mergeability_is_an_error() -> None:
    detector = _load_module()
    posted: list[str] = []

    class GitHub:
        def get(self, path: str) -> Any:
            if "pulls?" in path:
                return (
                    [
                        {"number": 3, "draft": False},
                        {"number": 4, "draft": False},
                        {"number": 5, "draft": True},
                    ]
                    if "page=1" in path
                    else []
                )
            states: dict[str, dict[str, object]] = {
                "3": {"mergeable": True, "mergeable_state": "behind"},
                "4": {"mergeable": None, "mergeable_state": "unknown"},
            }
            number = path.rsplit("/", 1)[-1]
            return {"number": int(number), "draft": False, **states[number]}

        def post(self, path: str, body: dict[str, str]) -> Any:
            posted.append(path)
            return {}

    with pytest.raises(RuntimeError, match="could not determine mergeability for pull #4"):
        detector.detect_conflicts(
            GitHub(), "eumemic/aios", "master", "def456", sleep=lambda _seconds: None
        )
    assert posted == []


def test_notification_failures_do_not_block_other_conflicted_prs() -> None:
    detector = _load_module()
    attempted: list[int] = []

    class GitHub:
        def get(self, path: str) -> Any:
            if "pulls?" in path:
                return (
                    [{"number": 1, "draft": False}, {"number": 2, "draft": False}]
                    if "page=1" in path
                    else []
                )
            number = int(path.rsplit("/", 1)[-1])
            return {
                "number": number,
                "draft": False,
                "mergeable": False,
                "mergeable_state": "dirty",
            }

        def post(self, path: str, body: dict[str, str]) -> Any:
            number = int(path.split("/issues/", 1)[1].split("/", 1)[0])
            attempted.append(number)
            raise RuntimeError(f"injected failure {number}")

    with pytest.raises(RuntimeError) as caught:
        detector.detect_conflicts(GitHub(), "eumemic/aios", "master", "abc123")

    assert attempted == [1, 2]
    assert "pull #1: RuntimeError: injected failure 1" in str(caught.value)
    assert "pull #2: RuntimeError: injected failure 2" in str(caught.value)
