#!/usr/bin/env python3
"""Notify open pull requests that became conflicted after a push to the base branch."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, Protocol

_API = "https://api.github.com"


class GitHubClient(Protocol):
    def get(self, path: str) -> Any: ...

    def post(self, path: str, body: dict[str, str]) -> Any: ...


class GitHub:
    def __init__(self, token: str) -> None:
        self._token = token

    def _request(self, method: str, path: str, body: dict[str, str] | None = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(f"{_API}{path}", data=data, method=method)
        request.add_header("Authorization", f"Bearer {self._token}")
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("X-GitHub-Api-Version", "2022-11-28")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"GitHub {method} {path} returned {exc.code}: {detail}") from exc

    def get(self, path: str) -> Any:
        return self._request("GET", path)

    def post(self, path: str, body: dict[str, str]) -> Any:
        return self._request("POST", path, body)


def _open_pull_requests(github: GitHubClient, repo: str, base: str) -> list[dict[str, Any]]:
    pulls: list[dict[str, Any]] = []
    page = 1
    while True:
        batch = github.get(f"/repos/{repo}/pulls?state=open&base={base}&per_page=100&page={page}")
        if not isinstance(batch, list):
            raise RuntimeError("GitHub pull list response was not a list")
        pulls.extend(pull for pull in batch if isinstance(pull, dict))
        if len(batch) < 100:
            return pulls
        page += 1


def _mergeability(
    github: GitHubClient,
    repo: str,
    number: int,
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    """Wait briefly for GitHub's asynchronous mergeability calculation."""
    pull: dict[str, Any] = {}
    for attempt in range(6):
        candidate = github.get(f"/repos/{repo}/pulls/{number}")
        if not isinstance(candidate, dict):
            raise RuntimeError(f"GitHub pull #{number} response was not an object")
        pull = candidate
        if pull.get("mergeable") is not None or pull.get("mergeable_state") != "unknown":
            break
        if attempt < 5:
            sleep(2**attempt)
    if pull.get("mergeable") is None and pull.get("mergeable_state") == "unknown":
        raise RuntimeError(
            f"GitHub could not determine mergeability for pull #{number} after 6 attempts"
        )
    return pull


def detect_conflicts(
    github: GitHubClient,
    repo: str,
    base: str,
    sha: str,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> list[int]:
    """Comment on each non-draft open PR GitHub classifies as ``dirty``."""
    conflicted: list[int] = []
    per_pull_failures: list[tuple[int | None, str, Exception]] = []
    short_sha = sha[:12]
    for summary in _open_pull_requests(github, repo, base):
        if summary.get("draft") is True:
            continue
        number = summary.get("number")
        if not isinstance(number, int):
            per_pull_failures.append(
                (None, "classify", RuntimeError("GitHub pull response had no integer number"))
            )
            continue
        try:
            pull = _mergeability(github, repo, number, sleep)
        except Exception as exc:
            per_pull_failures.append((number, "classify", exc))
            continue
        if pull.get("draft") is True or pull.get("mergeable_state") != "dirty":
            continue

        message = (
            f"<!-- aios-conflict-notice:{short_sha} -->\n"
            f"CONFLICTED: {base} `{short_sha}` cannot be merged into this branch cleanly. "
            "Rebase the branch before review or fix-round work continues."
        )
        conflicted.append(number)
        try:
            github.post(f"/repos/{repo}/issues/{number}/comments", {"body": message})
        except Exception as exc:
            per_pull_failures.append((number, "notify", exc))

    if per_pull_failures:
        details = "; ".join(
            f"pull #{number if number is not None else '<unknown>'}: "
            f"{type(exc).__name__}: {exc} ({stage})"
            for number, stage, exc in per_pull_failures
        )
        raise RuntimeError(
            f"Failed to process {len(per_pull_failures)} pull request(s): {details}"
        )
    return conflicted


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    sha = os.environ.get("GITHUB_SHA")
    base = os.environ.get("GITHUB_REF_NAME", "master")
    if not token or not repo or not sha:
        print("GITHUB_TOKEN, GITHUB_REPOSITORY, and GITHUB_SHA are required", file=sys.stderr)
        return 2
    conflicted = detect_conflicts(GitHub(token), repo, base, sha)
    print(json.dumps({"classification": "CONFLICTED", "pull_requests": conflicted}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
