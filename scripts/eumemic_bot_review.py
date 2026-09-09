#!/usr/bin/env python3
"""Run a proxy-backed coding agent, then separately publish as eumemic-bot.

Used by .github/workflows/eumemic-bot-review.yml. The workflow checks out the PR
head and installs the harness for the routed model. The workflow invokes this
launcher in two separate phases: ``agent`` writes the review artifact without
an installation token anywhere in its process tree, then ``publish`` receives
the freshly minted token after the agent has exited.

The publish phase POSTs the artifact as eumemic-bot and verifies GitHub stored
the run-specific marker. A review that never reached GitHub fails loudly here
rather than vanishing.

Env:
  REVIEW_ARTIFACT_PATH, REPO, PR_NUMBER, HEAD_SHA
  BASE_SHA (agent phase only), GH_TOKEN (publish phase only)
  REVIEW_MODEL (default: DEFAULT_MODEL below) — routed by prefix to a harness
  REVIEW_TIMEOUT_SECONDS (default: _REVIEW_SECONDS below) — agent wall clock. It
    must run out before the job's timeout-minutes: this script's FATAL leaves the
    step's continue-on-error to keep the check green and still write the
    "did not post" job summary, whereas a runner kill takes both away.
  One proxy key for the routed family (see _agent_command).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import IO, Any, NoReturn

ARTIFACT_HEADING = "### Code review"
DEFAULT_MODEL = "gpt-5.6-sol"
_REVIEW_SECONDS = 900

OAI_PROXY_URL = "https://oai-proxy.eumemic.ai/v1"
ANT_PROXY_URL = "https://ant-proxy.eumemic.ai"
XAI_PROXY_URL = "https://xai-proxy.eumemic.ai/v1"

# The agent reads PR-authored files (source, AGENTS.md, CLAUDE.md) and can run
# shell commands, so it must not inherit anything that grants write access. The
# installation token in particular can comment and push as eumemic-bot. Each
# harness gets back exactly the one proxy key it needs and nothing else.
#
# This list is defence in depth, never the guarantee: an agent that has a shell
# as this user can read /proc/$PPID/environ, which still holds every variable
# this process was exec'd with (unsetenv does not rewrite that mapping). Any
# secret the agent must not reach therefore has to be absent from *this*
# process — which is why the workflow mints the App token only after the agent
# has exited and passes only the routed family's proxy key. Credentials that
# live in files rather than the environment are handled separately by
# _drop_persisted_git_credentials.
_STRIPPED_ENV = (
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "ACTIONS_RUNTIME_TOKEN",
    "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
    "OAI_PROXY_API_KEY",
    "ANT_PROXY_API_KEY",
    "XAI_PROXY_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "XAI_API_KEY",
)

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


def _emit(text: str | None, stream: IO[str]) -> None:
    if text:
        print(text, file=stream, end="" if text.endswith("\n") else "\n")


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], text=True, capture_output=True, check=False)


def _artifact_in(text: str) -> str | None:
    """Return the artifact starting at the LAST `### Code review` heading.

    The heading is a contract on the agent's *final* message. Codex hands that
    message over in its own file, but the Claude Code and Pi paths read stdout,
    which also carries tool activity — an earlier mention (a grep hit on this
    file, a quoted prior review) must not become the comment body.
    """
    lines = text.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].strip() == ARTIFACT_HEADING:
            return "\n".join([lines[index].lstrip(), *lines[index + 1 :]]).strip()
    return None


def model_kind(model: str) -> str:
    if model.startswith("gpt-"):
        return "codex"
    if model.startswith("claude-"):
        return "claude"
    if model.startswith("grok-"):
        return "pi"
    _die(f"unsupported REVIEW_MODEL {model!r}; expected gpt-*, claude-*, or grok-*")


def _proxy_key(primary: str, fallback: str) -> str:
    value = os.environ.get(primary, "").strip() or os.environ.get(fallback, "").strip()
    if not value:
        _die(f"{primary} (or {fallback}) is not set")
    return value


def _agent_command(model: str, artifact_path: Path) -> tuple[list[str], dict[str, str]]:
    """Build the harness command and its proxy environment."""
    kind = model_kind(model)
    env = {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV}
    if kind == "codex":
        key = _proxy_key("OAI_PROXY_API_KEY", "OPENAI_API_KEY")
        env["OPENAI_API_KEY"] = key
        # Codex ignores OPENAI_BASE_URL: the built-in `openai` provider pins
        # api.openai.com and its own auth, so an env-var-only setup silently
        # 401s against the real OpenAI. Routing through the proxy requires
        # declaring a provider and selecting it.
        provider = "eumemic_oai_proxy"
        return (
            [
                "codex",
                "exec",
                "--model",
                model,
                "--sandbox",
                # GitHub-hosted runners do not permit bubblewrap to configure
                # its loopback interface. The job is ephemeral and trusted;
                # credentials are still stripped from the agent environment.
                "danger-full-access",
                "--ephemeral",
                "-c",
                f"model_provider={provider}",
                "-c",
                f'model_providers.{provider}={{name="eumemic oai-proxy",'
                f'base_url="{OAI_PROXY_URL}",env_key="OPENAI_API_KEY",wire_api="responses"}}',
                "--output-last-message",
                str(artifact_path),
                "-",
            ],
            env,
        )
    if kind == "claude":
        key = _proxy_key("ANT_PROXY_API_KEY", "ANTHROPIC_API_KEY")
        env.update(ANTHROPIC_API_KEY=key, ANTHROPIC_BASE_URL=ANT_PROXY_URL)
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

    key = _proxy_key("XAI_PROXY_API_KEY", "XAI_API_KEY")
    config_dir = artifact_path.parent / "pi-config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "models.json").write_text(
        json.dumps(
            {
                "providers": {
                    "xai-proxy": {
                        "name": "xAI (eumemic proxy)",
                        "baseUrl": XAI_PROXY_URL,
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


def _prompt(repo: str, pr_number: str, head_sha: str, base_sha: str) -> str:
    return (
        f"Review pull request {repo}#{pr_number}. The checkout is pinned to PR head "
        f"{head_sha} and the PR base is {base_sha}, both present locally. The changes under "
        f"review are exactly `git diff {base_sha}...{head_sha}` — read that range first and "
        f"do not review code outside it except as context. Report only actionable "
        f"correctness, security, or regression findings, with file and line references. If "
        f"there are none, say so briefly. Your final response must start exactly with "
        f"`{ARTIFACT_HEADING}`. {REVIEW_SCOPE}"
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
        except subprocess.TimeoutExpired as exc:
            # capture_output buffers everything until the process ends, so a
            # timeout is exactly the run whose log would otherwise be empty.
            _emit(exc.stdout if isinstance(exc.stdout, str) else None, sys.stdout)
            _emit(exc.stderr if isinstance(exc.stderr, str) else None, sys.stderr)
            _die(f"{model} review exceeded {timeout} seconds")
        _emit(result.stdout, sys.stdout)
        _emit(result.stderr, sys.stderr)
        if result.returncode:
            _die(f"{command[0]} exited with status {result.returncode}")
        output = artifact_path.read_text() if artifact_path.exists() else result.stdout
        artifact = _artifact_in(output)
        if artifact is None:
            _die(f"{model} returned no `{ARTIFACT_HEADING}` artifact")
        return artifact


def _github_request(
    method: str, url: str, token: str, body: dict[str, Any] | None = None
) -> dict[str, Any]:
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


def _pin_checkout(head_sha: str, base_sha: str) -> None:
    """Fail unless the working tree is the PR head and the base is reachable.

    Both halves are load-bearing. A review of the wrong tree is worse than no
    review, and the prompt names an explicit `base...head` range, so the base
    commit has to be an object the agent can actually diff against.
    """
    actual_head = _git("rev-parse", "HEAD").stdout.strip()
    if not actual_head.startswith(head_sha):
        _die(f"checkout HEAD {actual_head or 'unknown'} does not match PR head {head_sha}")
    if _git("cat-file", "-e", f"{base_sha}^{{commit}}").returncode:
        fetched = _git("fetch", "--no-tags", "--quiet", "origin", base_sha)
        if fetched.returncode or _git("cat-file", "-e", f"{base_sha}^{{commit}}").returncode:
            _die(f"PR base {base_sha} is missing from the checkout: {fetched.stderr.strip()[:300]}")


def _drop_persisted_git_credentials() -> None:
    """Remove the checkout's stored push credential before the agent runs.

    `actions/checkout` persists the workflow token as an `http.*.extraheader`
    in .git/config. It lives in a file, so stripping GH_TOKEN and friends from
    the agent's environment does not reach it, and the agent has a shell and
    (on the codex route, which runs unsandboxed) a network. `_pin_checkout` is
    the only thing that needs the credential, so it goes as soon as that
    returns.
    """
    listed = _git("config", "--local", "--name-only", "--get-regexp", r"http\..+\.extraheader")
    for key in listed.stdout.split():
        _git("config", "--local", "--unset-all", key)


def run_agent_phase() -> None:
    """Run and wait for the untrusted agent, then persist its final artifact."""
    if os.environ.get("GH_TOKEN", "").strip():
        _die("GH_TOKEN must not be set during the agent phase")
    repo = _env("REPO")
    pr_number = _env("PR_NUMBER")
    head_sha = _env("HEAD_SHA")
    base_sha = _env("BASE_SHA")
    artifact_path = Path(_env("REVIEW_ARTIFACT_PATH"))
    model = os.environ.get("REVIEW_MODEL", "").strip() or DEFAULT_MODEL
    timeout = int(os.environ.get("REVIEW_TIMEOUT_SECONDS") or _REVIEW_SECONDS)
    _pin_checkout(head_sha, base_sha)
    _drop_persisted_git_credentials()
    print(
        f"reviewing {repo}#{pr_number}@{head_sha} against {base_sha} with {model} "
        f"({model_kind(model)})"
    )
    review = run_agent(model, _prompt(repo, pr_number, head_sha, base_sha), timeout)
    artifact_path.write_text(review + "\n")
    print(f"wrote {ARTIFACT_HEADING} artifact to {artifact_path}")


def run_publish_phase() -> None:
    """Publish a completed artifact; this process never launches an agent."""
    token = _env("GH_TOKEN")
    repo = _env("REPO")
    pr_number = _env("PR_NUMBER")
    head_sha = _env("HEAD_SHA")
    artifact_path = Path(_env("REVIEW_ARTIFACT_PATH"))
    try:
        review = artifact_path.read_text()
    except OSError as exc:
        _die(f"cannot read review artifact {artifact_path}: {exc}")
    review = _artifact_in(review) or _die(
        f"review artifact contains no `{ARTIFACT_HEADING}` heading"
    )
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


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"agent", "publish"}:
        _die(f"usage: {Path(sys.argv[0]).name} agent|publish", code=2)
    if sys.argv[1] == "agent":
        run_agent_phase()
    else:
        run_publish_phase()


if __name__ == "__main__":
    main()
