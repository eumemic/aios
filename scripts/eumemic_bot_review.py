#!/usr/bin/env python3
"""Run a proxy-backed coding agent locally and publish its review as eumemic-bot.

Used by .github/workflows/eumemic-bot-review.yml. The workflow checks out the PR
head, installs the harness for the routed model, and hands in a short-lived
eumemic-bot installation token as GH_TOKEN.

The launcher owns publication: it runs the agent against the pinned checkout,
extracts the final `### Code review` artifact, POSTs it as eumemic-bot, and
verifies GitHub stored the run-specific marker. A review that never reached
GitHub fails loudly here rather than vanishing.

Publication is gated on EVIDENCE OF INSPECTION, not on the agent's exit status.
`codex` exits 0 when its sandbox blocks every command, so a zero exit says only
that the process ended — not that the agent read a byte of the diff. The agent
must echo the full sha256 of `git diff base...head`, which the launcher
recomputes; a mismatch or a missing line is a distinct, loud, never-publishable
state (NO_EVIDENCE_EXIT_CODE), because the alternative is an
authoritative-sounding "LGTM" from a reviewer that inspected nothing.

The digest is the ONLY accepting channel. The line count is echoed for legible
diagnostics but cannot authorise publication: it is public at the PR's `.diff`
URL, so it never distinguished a real read from a network fetch.

Because the sandbox is off (the only mode that executes on a hosted runner),
the agent's environment is also stripped of the Actions control files
(GITHUB_OUTPUT/ENV/PATH), not just of credentials — otherwise the agent could
forge `published=true` and silence the workflow's own missing-review detector.

Env:
  GH_TOKEN, REPO, PR_NUMBER, HEAD_SHA, BASE_SHA
  REVIEW_MODEL (default: DEFAULT_MODEL below) — routed by prefix to a harness
  REVIEW_SANDBOX_MODE (default: DEFAULT_SANDBOX below) — codex sandbox policy
  REVIEW_TIMEOUT_SECONDS (default: _REVIEW_SECONDS below) — agent wall clock. It
    must run out before the job's timeout-minutes: this script's FATAL leaves the
    step's continue-on-error to keep the check green and still write the
    "did not post" job summary, whereas a runner kill takes both away.
  One proxy key for the routed family (see _agent_command).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import IO, NoReturn

ARTIFACT_HEADING = "### Code review"
DEFAULT_MODEL = "gpt-5.6-sol"
_REVIEW_SECONDS = 900

# A verdict is only publishable when the agent proved it read the diff. The
# proof is a line the agent can only produce by running the diff command in the
# checkout: the launcher computes the same digest itself and compares. Nothing
# derivable from the prompt alone counts — the expected values are deliberately
# NOT in the prompt, only the recipe for deriving them.
EVIDENCE_TEMPLATE = "<!-- inspected: lines=<N> sha256=<HEX> -->"
# The FULL 64-hex digest, not a prefix. A prefix shortens the only
# high-entropy channel, and the width itself is load-bearing: a mutant
# widening this bound survived the suite once already.
_EVIDENCE_RE = re.compile(
    r"<!--\s*inspected:\s*lines=(\d+)\s+sha256=([0-9a-fA-F]{64})\s*-->", re.IGNORECASE
)
# "The agent inspected nothing" is NOT an ordinary failure: codex exits 0 when
# its sandbox blocks every command, so this is the state that used to publish an
# authoritative-sounding verdict off a zero-byte read. Distinct exit code,
# distinct banner, never publishable.
NO_EVIDENCE_EXIT_CODE = 3
NO_EVIDENCE_BANNER = "NO EVIDENCE OF INSPECTION — refusing to publish a verdict"

SANDBOX_MODES = ("read-only", "workspace-write", "danger-full-access")
DEFAULT_SANDBOX = "read-only"

# This channel is advisory by construction: the workflow sets continue-on-error
# on every step, so the PR check is green whether the verdict is LGTM or
# "BLOCKING". The banner says so in the comment, because an unqualified
# "### Code review" reads as an authoritative gate that does not exist.
ADVISORY_BANNER = (
    "**Advisory automated review — non-blocking.** This check never fails the PR; "
    "a green check means the harness ran, not that the code is approved. Findings below "
    "are input to a human reviewer, not a merge gate."
)

OAI_PROXY_URL = "https://oai-proxy.eumemic.ai/v1"
ANT_PROXY_URL = "https://ant-proxy.eumemic.ai"
XAI_PROXY_URL = "https://xai-proxy.eumemic.ai/v1"

# The agent reads PR-authored files (source, AGENTS.md, CLAUDE.md) and can run
# shell commands, so it must not inherit anything that grants write access. The
# installation token in particular can comment and push as eumemic-bot. Each
# harness gets back exactly the one proxy key it needs and nothing else.
#
# This list is not only about SECRETS. Since the sandbox is off
# (danger-full-access — the only mode that executes on a hosted runner), the
# agent has a real shell, so the Actions CONTROL PLANE is attack surface too:
#
#   GITHUB_OUTPUT — the input to the workflow's "did not post" safety net. An
#     agent that appends `published=true` and then emits no evidence gets the
#     launcher to refuse publication (exit 3, nothing posted) while the net
#     reads `published == 'true'` and stays silent: green run, no review, no
#     warning. That is precisely the silent failure this launcher exists to
#     close, so leaving the path in the child env hands the agent the ability
#     to forge the detector's own input.
#   GITHUB_ENV / GITHUB_PATH — writes there mutate later steps of this job.
#
# Stripping these from the CHILD is functionally free: _record_published reads
# GITHUB_OUTPUT from the LAUNCHER's own os.environ, which is untouched.
_STRIPPED_ENV = (
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "ACTIONS_RUNTIME_TOKEN",
    "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
    "GITHUB_OUTPUT",
    "GITHUB_ENV",
    "GITHUB_PATH",
    "OAI_PROXY_API_KEY",
    "ANT_PROXY_API_KEY",
    "XAI_PROXY_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "XAI_API_KEY",
)

# GitHub Actions exposes additional control-plane paths (for example
# ``_runner_file_commands/step_summary_*``) under unrelated environment keys.
# Since the agent runs with a real shell, deny those paths wherever they occur,
# not only when carried by the well-known GITHUB_* variable names.
_CONTROL_PATH_MARKERS = ("file_commands", "_runner_file_commands")

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
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in _STRIPPED_ENV
        and not any(marker in v.lower() for marker in _CONTROL_PATH_MARKERS)
    }
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
                _sandbox_mode(),
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


def _sandbox_mode() -> str:
    """Sandbox policy for the codex harness.

    Codex ships its own bubblewrap and `read-only` needs namespaces the GitHub
    runner refuses: every command dies with
    `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted` — and codex
    still exits 0, which is how a review that read nothing got published. The
    runner is already an ephemeral single-use VM, and the launcher strips every
    writable credential from the agent env before exec, so the second sandbox
    layer buys little and cost us the entire review. Overridable for hosts where
    bwrap does work.
    """
    mode = os.environ.get("REVIEW_SANDBOX_MODE", "").strip() or DEFAULT_SANDBOX
    if mode not in SANDBOX_MODES:
        _die(
            f"unsupported REVIEW_SANDBOX_MODE {mode!r}; expected one of {', '.join(SANDBOX_MODES)}"
        )
    return mode


def diff_evidence(base_sha: str, head_sha: str) -> tuple[int, str]:
    """Return (line count, sha256) of `git diff base...head`, computed locally.

    This is the value the agent has to reproduce. It is deliberately never put
    in the prompt — only the recipe for deriving it — so an agent whose shell is
    dead cannot emit it, and neither can one that guessed.
    """
    result = subprocess.run(
        ["git", "--no-pager", "diff", f"{base_sha}...{head_sha}"],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        _die(
            f"could not compute the diff for {base_sha}...{head_sha}: {result.stderr.decode(errors='replace')[:300]}"
        )
    raw = result.stdout
    if not raw.strip():
        _die(f"`git diff {base_sha}...{head_sha}` is empty; there is nothing to review")
    return raw.count(b"\n"), hashlib.sha256(raw).hexdigest()


def _die_without_evidence(detail: str) -> NoReturn:
    """The loud, distinct, never-publishable state.

    Separate exit code and banner from an ordinary FATAL because this is the
    exact condition that used to sail through as a green, authoritative verdict.
    """
    print(f"FATAL: {NO_EVIDENCE_BANNER}: {detail}", file=sys.stderr)
    print(f"::error title={NO_EVIDENCE_BANNER}::{detail}", file=sys.stdout)
    raise SystemExit(NO_EVIDENCE_EXIT_CODE)


def require_inspection_evidence(artifact: str, expected: tuple[int, str]) -> None:
    """Refuse to publish unless the artifact proves the agent read the diff.

    THE DIGEST IS THE ONLY ACCEPTING CHANNEL, and it must match in full.

    An earlier revision also accepted a matching line count on its own,
    documented as "derivable only by running the diff in the checkout". That
    was FALSE: `https://github.com/<org>/<repo>/pull/<n>.diff` is public and
    carries the identical line count with no checkout access at all (measured
    on this PR: 2243 on both, different bytes so different digests). An agent
    whose shell is dead but whose network is live — the exact blocked-agent
    state this gate exists to catch — could obtain the count and pair it with
    64 arbitrary hex characters. The low-entropy half of an OR is the strength
    of the whole OR.

    The line count is still parsed and still reported on a mismatch, because it
    makes the diagnostic legible; it just cannot authorise publication.

    Requiring the full digest does not reintroduce the brittleness the OR was
    guarding against: `sha256sum` output is a stable 64-hex string, whereas
    `wc -l` was the channel prone to benign formatting drift.

    Note for future mutation runs: replacing the `==` below with
    `expected_digest.startswith(claimed_digest)` is an EQUIVALENT MUTANT and no
    test can kill it. `_EVIDENCE_RE` admits exactly 64 hex characters and
    `expected_digest` is a sha256 hexdigest, so both operands are always the
    same length, where `startswith` and `==` coincide. `==` is kept because it
    states the intent directly and does not depend on the regex for its
    safety — but a survivor there is expected, not a gap.
    """
    expected_lines, expected_digest = expected
    match = _EVIDENCE_RE.search(artifact)
    if match is None:
        _die_without_evidence(
            f"the agent's `{ARTIFACT_HEADING}` carries no well-formed `{EVIDENCE_TEMPLATE}` line "
            "(the sha256 must be all 64 hex characters), so nothing shows it read the diff. "
            "Its verdict is not publishable."
        )
    claimed_lines = int(match.group(1))
    claimed_digest = match.group(2).lower()
    if claimed_digest == expected_digest.lower():
        return
    _die_without_evidence(
        f"inspection evidence does not match the diff: agent claimed lines={claimed_lines} "
        f"sha256={claimed_digest}, launcher computed lines={expected_lines} "
        f"sha256={expected_digest}. Its verdict is not publishable."
    )


def _prompt(repo: str, pr_number: str, head_sha: str, base_sha: str) -> str:
    return (
        f"Review pull request {repo}#{pr_number}. The checkout is pinned to PR head "
        f"{head_sha} and the PR base is {base_sha}, both present locally. The changes under "
        f"review are exactly `git diff {base_sha}...{head_sha}` — read that range first and "
        f"do not review code outside it except as context. Report only actionable "
        f"correctness, security, or regression findings, with file and line references. If "
        f"there are none, say so briefly. Your final response must start exactly with "
        f"`{ARTIFACT_HEADING}`.\n\n"
        f"MANDATORY PROOF OF INSPECTION. Run exactly:\n"
        f"  git --no-pager diff {base_sha}...{head_sha} | wc -l\n"
        f"  git --no-pager diff {base_sha}...{head_sha} | sha256sum\n"
        f"and end your final response with a line of the form\n"
        f"  {EVIDENCE_TEMPLATE}\n"
        f"substituting the real values you observed. Quote the sha256 IN FULL — all 64 hex "
        f"characters, not an abbreviation: the digest is the channel that authorises "
        f"publication, and an abbreviated or malformed one is refused. Do NOT guess, infer, or "
        f"fabricate the values: the launcher recomputes the digest and refuses to publish any "
        f"review whose digest does not match exactly. The line count alone will NOT do — it is "
        f"published at the PR's .diff URL and so proves nothing about what you read. "
        f"If your shell cannot run those commands, say so plainly and DO NOT emit an "
        f"evidence line and DO NOT render a verdict — an unverifiable review is worse than "
        f"none. {REVIEW_SCOPE}"
    )


def run_agent(
    model: str, prompt: str, timeout: int, evidence: tuple[int, str] | None = None
) -> str:
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
        # Exit 0 proves only that the harness process ended. It does NOT prove the
        # agent could read anything: codex exits 0 when bubblewrap blocks every
        # command. The evidence check is what separates a review from a fluent
        # guess, so it gates publication independently of the exit status.
        if evidence is not None:
            require_inspection_evidence(artifact, evidence)
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


def _record_published(comment_url: str) -> None:
    """Emit `published=true` on GITHUB_OUTPUT, only on a verified publication."""
    output_path = os.environ.get("GITHUB_OUTPUT", "").strip()
    if not output_path:
        return
    try:
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write("published=true\n")
            handle.write(f"comment_url={comment_url}\n")
    except OSError as exc:  # pragma: no cover - runner filesystem fault
        print(f"warning: could not record publication: {exc}", file=sys.stderr)


def main() -> None:
    token = _env("GH_TOKEN")
    repo = _env("REPO")
    pr_number = _env("PR_NUMBER")
    head_sha = _env("HEAD_SHA")
    base_sha = _env("BASE_SHA")
    model = os.environ.get("REVIEW_MODEL", "").strip() or DEFAULT_MODEL
    timeout = int(os.environ.get("REVIEW_TIMEOUT_SECONDS") or _REVIEW_SECONDS)
    _pin_checkout(head_sha, base_sha)
    evidence = diff_evidence(base_sha, head_sha)
    print(
        f"reviewing {repo}#{pr_number}@{head_sha} against {base_sha} with {model} "
        f"({model_kind(model)}); diff is {evidence[0]} lines, sha256 {evidence[1]}"
    )
    review = run_agent(model, _prompt(repo, pr_number, head_sha, base_sha), timeout, evidence)
    marker = f"<!-- eumemic-bot-review:{head_sha} -->"
    body = f"{ADVISORY_BANNER}\n\n{review}"
    if marker not in body:
        body = f"{body}\n\n{marker}"
    comment = _github_request(
        "POST",
        f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments",
        token,
        {"body": body},
    )
    comment_url = comment.get("html_url")
    if not comment_url or marker not in str(comment.get("body", "")):
        _die(f"GitHub did not confirm the review comment: {json.dumps(comment)[:400]}")
    # Positive publication signal for the workflow's safety net. The net must key
    # on "did a genuine review land?" — the real miss exited 0 with the step
    # outcome 'success', so a net keyed on step outcome was skipped precisely
    # when it was needed. This line is written only after GitHub echoed the
    # marker back, so its ABSENCE is the fact the net reads.
    _record_published(comment_url)
    print(f"posted and verified {ARTIFACT_HEADING}: {comment_url}")


if __name__ == "__main__":
    main()
