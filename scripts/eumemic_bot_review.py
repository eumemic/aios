#!/usr/bin/env python3
"""Run an aios dev-review session and publish its GitHub review artifact.

Used by .github/workflows/eumemic-bot-review.yml. The workflow hands in a
short-lived eumemic-bot installation token as GH_TOKEN.

The launcher owns publication: it waits for the session to stop working, reads
the `### Code review` artifact off the event log, POSTs it as eumemic-bot,
verifies GitHub stored it, and only then archives the session. The session
itself never posts — a review that never reached GitHub now fails loudly here
instead of vanishing with a self-archiving session.

Resolution order for the reviewer agent:
  1. AGENT_ID if set
  2. exact name match for AGENT_NAME (default: dev-review)
  3. fail with the names visible to this API key (account-scoped)

Resolution order for the sandbox environment (required by POST /v1/sessions):
  1. ENVIRONMENT_ID if set
  2. exact name match for ENVIRONMENT_NAME (default: dev-pipeline-real)
  3. fail with the names visible to this API key (account-scoped)

Env:
  AIOS_URL, AIOS_API_KEY, GH_TOKEN, REPO, PR_NUMBER, HEAD_SHA, CLONE_URL
  AGENT_NAME (default: dev-review), AGENT_ID (optional)
  ENVIRONMENT_NAME (default: dev-pipeline-real), ENVIRONMENT_ID (optional)
  REVIEW_TIMEOUT_SECONDS (default: 1200) — whole-review budget, shared by the
    first turn and the corrective turn. Keep it under the job's timeout-minutes.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, NoReturn

AGENT_NAME = os.environ.get("AGENT_NAME", "dev-review")
ENVIRONMENT_NAME = os.environ.get("ENVIRONMENT_NAME", "dev-pipeline-real")

ARTIFACT_HEADING = "### Code review"

# Long-poll window for GET /v1/sessions/{id}/wait (server caps it at 60). The
# socket deadline must OUTLIVE it, or every poll dies on a client read timeout
# before the server ever answers.
_WAIT_SECONDS = 30
_WAIT_HTTP_TIMEOUT = _WAIT_SECONDS * 2


def _die(msg: str, code: int = 1) -> NoReturn:
    print(f"FATAL: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _skip(msg: str) -> NoReturn:
    print(f"SKIP: {msg}", file=sys.stderr)
    raise SystemExit(0)


def _env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        _die(f"{name} is not set")
    return val


def _request(
    method: str, url: str, api_key: str, body: dict | None = None, timeout: float = 30
) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Accept", "application/json")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:800]
        _die(f"{method} {url} returned {exc.code}: {detail}")
    except (urllib.error.URLError, TimeoutError) as exc:
        # A read timeout surfaces as a bare TimeoutError, not a URLError.
        _die(f"{method} {url} failed: {exc}")


def _github_request(method: str, url: str, token: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:800]
        _die(f"{method} {url} returned {exc.code}: {detail}")
    except (urllib.error.URLError, TimeoutError) as exc:
        _die(f"{method} {url} failed: {exc}")


def _wait_until_working_stops(base: str, api_key: str, session_id: str, deadline: float) -> str:
    """Block until the session is no longer ``active``; return its status.

    ``GET /wait`` is the right primitive: it long-polls, returns the moment new
    events land (so the status read happens milliseconds after the final
    assistant message), and reports the derived session status. ``GET /await``
    is NOT — it resolves on ``last_reacted_seq >= watermark``, and the model's
    very first tool-call turn already satisfies that, long before the review
    exists.
    """
    after = 0
    while time.monotonic() < deadline:
        query = urllib.parse.urlencode({"after": after, "timeout": _WAIT_SECONDS})
        payload = _request(
            "GET",
            f"{base}/v1/sessions/{session_id}/wait?{query}",
            api_key,
            timeout=_WAIT_HTTP_TIMEOUT,
        )
        after = payload.get("next_after", after)
        status = payload.get("session_status")
        if status != "active":
            return str(status)
    _die(f"session {session_id} was still working after the review timeout")


def _message_text(content: Any) -> str:
    """Assistant content is a plain string, or content-part blocks on providers
    that emit them (mirrors ``aios.cli.tail_format._as_text``)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return ""


def _artifact_in(text: str) -> str | None:
    """The artifact is the heading line and everything after it, or None."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith(ARTIFACT_HEADING):
            # The comment body must OPEN with the heading, so drop any lead-in
            # lines and the heading line's own indent; the rest is verbatim.
            return "\n".join([line.lstrip(), *lines[i + 1 :]]).strip()
    return None


def _review_from_events(base: str, api_key: str, session_id: str) -> str | None:
    query = urllib.parse.urlencode({"dir": "backward", "kind": "message", "limit": "100"})
    payload = _request("GET", f"{base}/v1/sessions/{session_id}/events?{query}", api_key)
    # ``dir=backward`` pages newest-first, so the first hit is the latest artifact.
    for event in payload.get("data", []):
        data = event.get("data", {})
        if data.get("role") != "assistant":
            continue
        artifact = _artifact_in(_message_text(data.get("content")))
        if artifact is not None:
            return artifact
    return None


def _ask_for_review_artifact(base: str, api_key: str, session_id: str) -> str:
    deadline = time.monotonic() + int(os.environ.get("REVIEW_TIMEOUT_SECONDS", "1200"))
    status = _wait_until_working_stops(base, api_key, session_id, deadline)
    review = _review_from_events(base, api_key, session_id)
    if review is not None:
        return review
    if status == "archived":
        _die(f"session {session_id} was archived without a `{ARTIFACT_HEADING}` artifact")

    # One corrective turn handles a model that followed the dev-review workflow-child
    # contract and attempted the unavailable `return` tool in this foreground session.
    _request(
        "POST",
        f"{base}/v1/sessions/{session_id}/messages",
        api_key,
        {
            "content": (
                "The GitHub publisher needs your review as a normal assistant message now. "
                "Do not call `return` or any posting tool. Reply with the complete artifact, "
                f"starting exactly with `{ARTIFACT_HEADING}`."
            )
        },
    )
    status = _wait_until_working_stops(base, api_key, session_id, deadline)
    review = _review_from_events(base, api_key, session_id)
    if review is None:
        _die(f"session {session_id} went {status} without a `{ARTIFACT_HEADING}` artifact")
    return review


def _archive(base: str, api_key: str, session_id: str) -> None:
    """Reclaim the session on every exit path.

    ``archive_when_idle`` can no longer do it — the session must outlive its own
    idleness so the launcher can read the artifact — so the launcher owns the
    reclaim, including when publishing failed. Never masks the original failure.
    """
    try:
        _request("POST", f"{base}/v1/sessions/{session_id}/archive", api_key)
    except SystemExit:
        print(f"WARN: session {session_id} was left unarchived", file=sys.stderr)


def _list_agents(base: str, api_key: str, name: str | None = None) -> list[dict]:
    q = {"limit": "50"}
    if name:
        q["name"] = name
    url = f"{base}/v1/agents?{urllib.parse.urlencode(q)}"
    payload = _request("GET", url, api_key)
    rows = payload.get("data")
    if not isinstance(rows, list):
        _die(f"GET /v1/agents missing data: {json.dumps(payload)[:400]}")
    return rows


def _list_environments(base: str, api_key: str) -> list[dict]:
    url = f"{base}/v1/environments?{urllib.parse.urlencode({'limit': '100'})}"
    payload = _request("GET", url, api_key)
    rows = payload.get("data")
    if not isinstance(rows, list):
        _die(f"GET /v1/environments missing data: {json.dumps(payload)[:400]}")
    return rows


def resolve_agent(base: str, api_key: str) -> str:
    pinned = os.environ.get("AGENT_ID", "").strip()
    if pinned:
        return pinned
    exact = [r for r in _list_agents(base, api_key, AGENT_NAME) if r.get("name") == AGENT_NAME]
    if len(exact) == 1:
        return str(exact[0]["id"])
    visible = [f"{r.get('name')}:{r.get('id')}" for r in _list_agents(base, api_key)]
    _skip(
        f"no live agent named {AGENT_NAME!r} on this API key's account "
        f"(visible: {visible or 'none'}). Set DEV_REVIEW_AGENT_ID to a reviewer on this account."
    )


def resolve_environment(base: str, api_key: str) -> str:
    pinned = os.environ.get("ENVIRONMENT_ID", "").strip()
    if pinned:
        return pinned
    exact = [r for r in _list_environments(base, api_key) if r.get("name") == ENVIRONMENT_NAME]
    if len(exact) == 1:
        return str(exact[0]["id"])
    visible = [f"{r.get('name')}:{r.get('id')}" for r in _list_environments(base, api_key)]
    _die(
        f"no environment named {ENVIRONMENT_NAME!r} on this API key's account "
        f"(visible: {visible or 'none'}). Set ENVIRONMENT_ID."
    )


def main() -> None:
    base = _env("AIOS_URL").rstrip("/")
    api_key = _env("AIOS_API_KEY")
    token = _env("GH_TOKEN")
    repo = _env("REPO")
    pr_number = _env("PR_NUMBER")
    head_sha = _env("HEAD_SHA")
    clone_url = _env("CLONE_URL")

    agent_id = resolve_agent(base, api_key)
    environment_id = resolve_environment(base, api_key)
    prompt = (
        f"Review pull request {repo}#{pr_number} at {head_sha}. "
        f"The repository is cloned at /mnt/review. "
        f"Fetch the PR diff via the github http_request server "
        f"(GET /repos/{repo}/pulls/{pr_number} and /repos/{repo}/pulls/{pr_number}/files). "
        f"If http_request is unauthorized, use GH_TOKEN from the environment with gh or curl. "
        f"This is a foreground session, so the `return` tool is unavailable. Do not post to "
        f"GitHub yourself. Reply as a normal assistant message with the complete review artifact; "
        f"its first line must be exactly `{ARTIFACT_HEADING}`. The launcher will post and verify "
        f"it."
    )
    body = {
        "agent_id": agent_id,
        "environment_id": environment_id,
        "title": f"eumemic-bot review {repo}#{pr_number}",
        # The launcher archives — see _archive. Self-reclaim would race the read
        # of the artifact the launcher is about to publish.
        "archive_when_idle": False,
        "initial_message": prompt,
        "env": {"GH_TOKEN": token, "GH_REPO": repo, "PR_NUMBER": pr_number},
        "resources": [
            {
                "type": "github_repository",
                "url": clone_url,
                "mount_path": "/mnt/review",
                "authorization_token": token,
                "git_user_name": "eumemic-bot[bot]",
                "git_user_email": "4752589+eumemic-bot[bot]@users.noreply.github.com",
            }
        ],
        "metadata": {
            "source": "eumemic-bot-review",
            "repo": repo,
            "pr_number": pr_number,
            "head_sha": head_sha,
        },
    }
    session = _request("POST", f"{base}/v1/sessions", api_key, body)
    sid = session.get("id")
    if not sid:
        _die(f"create session returned no id: {json.dumps(session)[:400]}")
    print(
        f"started session {sid} on agent {agent_id} env {environment_id} "
        f"for {repo}#{pr_number}@{head_sha}"
    )
    try:
        review = _ask_for_review_artifact(base, api_key, str(sid))
        marker = f"<!-- eumemic-bot-review:{head_sha} -->"
        if marker not in review:
            review = f"{review}\n\n{marker}"
        comment = _github_request(
            "POST",
            f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments",
            token,
            {"body": review},
        )
        comment_url = comment.get("html_url")
        # The marker round-trip proves GitHub stored THIS run's artifact; exact
        # body equality would also fail on any server-side normalization.
        if not comment_url or marker not in _message_text(comment.get("body")):
            _die(f"GitHub did not confirm the review comment: {json.dumps(comment)[:400]}")
        print(f"posted and verified {ARTIFACT_HEADING}: {comment_url}")
    finally:
        _archive(base, api_key, str(sid))


if __name__ == "__main__":
    main()
