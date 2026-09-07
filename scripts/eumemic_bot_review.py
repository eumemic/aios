#!/usr/bin/env python3
"""Run a proxy-backed coding agent locally and publish its review as eumemic-bot."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import NoReturn

ARTIFACT_HEADING = "### Code review"
DEFAULT_MODEL = "gpt-5.6-sol"
_REVIEW_SECONDS = 900
REVIEW_SCOPE = (
    "Keep verification proportional to the changed code. Use focused tests for affected "
    "behavior, but do not run repository-wide test, lint, format, or type-check suites; CI "
    "already runs those. Do not modify the checkout or post to GitHub."
)


def _die(msg: str, code: int = 1) -> NoReturn:
    print(f"FATAL: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        _die(f"{name} is not set")
    return value


def _artifact_in(text: str) -> str | None:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == ARTIFACT_HEADING:
            return "\n".join([line.lstrip(), *lines[index + 1 :]]).strip()
    return None


def model_kind(model: str) -> str:
    if model.startswith("gpt-"):
        return "codex"
    if model.startswith("claude-"):
        return "claude"
    if model.startswith("grok-"):
        return "pi"
    _die(f"unsupported REVIEW_MODEL {model!r}; expected gpt-*, claude-*, or grok-*")


def _agent_command(model: str, artifact_path: Path) -> tuple[list[str], dict[str, str]]:
    """Build the harness command and its proxy environment."""
    kind = model_kind(model)
    env = os.environ.copy()
    if kind == "codex":
        key = env.get("OAI_PROXY_API_KEY") or env.get("OPENAI_API_KEY")
        if not key:
            _die("OAI_PROXY_API_KEY (or OPENAI_API_KEY) is not set")
        env.update(OPENAI_API_KEY=key, OPENAI_BASE_URL="https://oai-proxy.eumemic.ai/v1")
        return (
            [
                "codex",
                "exec",
                "--model",
                model,
                "--sandbox",
                "read-only",
                "--ephemeral",
                "--output-last-message",
                str(artifact_path),
                "-",
            ],
            env,
        )
    if kind == "claude":
        key = env.get("ANT_PROXY_API_KEY") or env.get("ANTHROPIC_API_KEY")
        if not key:
            _die("ANT_PROXY_API_KEY (or ANTHROPIC_API_KEY) is not set")
        env.update(ANTHROPIC_API_KEY=key, ANTHROPIC_BASE_URL="https://ant-proxy.eumemic.ai")
        return (
            [
                "claude",
                "--print",
                "--model",
                model,
                "--output-format",
                "text",
                "--no-session-persistence",
                "--allowedTools",
                "Read,Glob,Grep,Bash",
                "-",
            ],
            env,
        )

    key = env.get("XAI_PROXY_API_KEY") or env.get("XAI_API_KEY")
    if not key:
        _die("XAI_PROXY_API_KEY (or XAI_API_KEY) is not set")
    config_dir = artifact_path.parent / "pi-config"
    config_dir.mkdir()
    (config_dir / "models.json").write_text(
        json.dumps(
            {
                "providers": {
                    "xai-proxy": {
                        "name": "xAI (eumemic proxy)",
                        "baseUrl": "https://xai-proxy.eumemic.ai/v1",
                        "api": "openai-responses",
                        "apiKey": key,
                        "authHeader": True,
                        "models": [
                            {
                                "id": model,
                                "name": model,
                                "reasoning": True,
                                "input": ["text"],
                                "contextWindow": 256000,
                                "maxTokens": 16384,
                            }
                        ],
                    }
                }
            }
        )
    )
    env.update(XAI_API_KEY=key, PI_CODING_AGENT_DIR=str(config_dir))
    return (
        [
            "pi",
            "--print",
            "--no-session",
            "--no-extensions",
            "--no-skills",
            "--provider",
            "xai-proxy",
            "--model",
            model,
            "--tools",
            "read,grep,find,ls,bash",
            "--",
        ],
        env,
    )


def _prompt(repo: str, pr_number: str, head_sha: str) -> str:
    return (
        f"Review pull request {repo}#{pr_number}. The current checkout is pinned to PR head "
        f"{head_sha}. Review the changes against the PR base using local git history and inspect "
        f"the affected source. Report only actionable correctness, security, or regression "
        f"findings, with file and line references. If there are none, say so briefly. "
        f"Your final response must start exactly with `{ARTIFACT_HEADING}`. {REVIEW_SCOPE}"
    )


def run_agent(model: str, prompt: str, timeout: int) -> str:
    with tempfile.TemporaryDirectory(prefix="eumemic-review-") as temp:
        artifact_path = Path(temp) / "last-message.md"
        command, env = _agent_command(model, artifact_path)
        try:
            result = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                env=env,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError:
            _die(f"{command[0]} is not installed")
        except subprocess.TimeoutExpired:
            _die(f"{model} review exceeded {timeout} seconds")
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
        if result.returncode:
            _die(f"{command[0]} exited with status {result.returncode}")
        output = artifact_path.read_text() if artifact_path.exists() else result.stdout
        artifact = _artifact_in(output)
        if artifact is None:
            _die(f"{model} returned no `{ARTIFACT_HEADING}` artifact")
        return artifact


def _github_request(method: str, url: str, token: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    if body is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:800]
        _die(f"{method} {url} returned {exc.code}: {detail}")
    except (urllib.error.URLError, TimeoutError) as exc:
        _die(f"{method} {url} failed: {exc}")


def main() -> None:
    token = _env("GH_TOKEN")
    repo = _env("REPO")
    pr_number = _env("PR_NUMBER")
    head_sha = _env("HEAD_SHA")
    model = os.environ.get("REVIEW_MODEL", "").strip() or DEFAULT_MODEL
    timeout = int(os.environ.get("REVIEW_TIMEOUT_SECONDS") or _REVIEW_SECONDS)
    actual_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    ).stdout.strip()
    if not actual_head.startswith(head_sha):
        _die(f"checkout HEAD {actual_head or 'unknown'} does not match PR head {head_sha}")
    print(f"reviewing {repo}#{pr_number}@{head_sha} with {model} ({model_kind(model)})")
    review = run_agent(model, _prompt(repo, pr_number, head_sha), timeout)
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
    if not comment_url or marker not in str(comment.get("body", "")):
        _die(f"GitHub did not confirm the review comment: {json.dumps(comment)[:400]}")
    print(f"posted and verified {ARTIFACT_HEADING}: {comment_url}")


if __name__ == "__main__":
    main()
