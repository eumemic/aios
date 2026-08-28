#!/usr/bin/env python3
"""Start an aios dev-review session for a GitHub pull request.

Used by .github/workflows/eumemic-bot-review.yml. The workflow hands in a
short-lived eumemic-bot installation token as GH_TOKEN.

Resolution order for the reviewer agent:
  1. AGENT_ID if set
  2. exact name match for AGENT_NAME (default: dev-review)
  3. fail with the names visible to this API key (account-scoped)

Env:
  AIOS_URL, AIOS_API_KEY, GH_TOKEN, REPO, PR_NUMBER, HEAD_SHA, CLONE_URL
  AGENT_NAME (default: dev-review), AGENT_ID (optional)
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

AGENT_NAME = os.environ.get("AGENT_NAME", "dev-review")


def _die(msg: str, code: int = 1) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _skip(msg: str) -> None:
    print(f"SKIP: {msg}", file=sys.stderr)
    raise SystemExit(0)


def _env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        _die(f"{name} is not set")
    return val


def _request(method: str, url: str, api_key: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Accept", "application/json")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:800]
        _die(f"{method} {url} returned {exc.code}: {detail}")
    except urllib.error.URLError as exc:
        _die(f"{method} {url} failed: {exc}")


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


def main() -> None:
    base = _env("AIOS_URL").rstrip("/")
    api_key = _env("AIOS_API_KEY")
    token = _env("GH_TOKEN")
    repo = _env("REPO")
    pr_number = _env("PR_NUMBER")
    head_sha = _env("HEAD_SHA")
    clone_url = _env("CLONE_URL")

    agent_id = resolve_agent(base, api_key)
    prompt = (
        f"Review pull request {repo}#{pr_number} at {head_sha}. "
        f"The repository is cloned at /mnt/review. "
        f"Fetch the PR diff via the github http_request server "
        f"(GET /repos/{repo}/pulls/{pr_number} and /repos/{repo}/pulls/{pr_number}/files). "
        f"If http_request is unauthorized, use GH_TOKEN from the environment with gh or curl. "
        f"Post a review-artifact comment whose body begins with the line `### Code review` "
        f"(POST /repos/{repo}/issues/{pr_number}/comments). "
        f"Return ONLY via return a value conforming to "
        f"{{verdict:'pass'|'fail', issues:[...], artifact_posted:true}}."
    )
    body = {
        "agent_id": agent_id,
        "title": f"eumemic-bot review {repo}#{pr_number}",
        "archive_when_idle": True,
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
    print(f"started session {sid} on agent {agent_id} for {repo}#{pr_number}@{head_sha}")


if __name__ == "__main__":
    main()
