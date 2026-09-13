"""Unit tests for the local coding-agent review launcher."""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).parents[2]
_SCRIPT = _ROOT / "scripts/eumemic_bot_review.py"
_WORKFLOW = _ROOT / ".github/workflows/eumemic-bot-review.yml"
_SPEC = importlib.util.spec_from_file_location("eumemic_bot_review", _SCRIPT)
assert _SPEC and _SPEC.loader
reviewer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reviewer)


def _ok(stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], 0, stdout, "")


@pytest.fixture
def clean_env(monkeypatch: Any) -> None:
    """Start from an environment with no harness credentials of any kind."""
    for name in reviewer._STRIPPED_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "model,kind", [("gpt-5.6-sol", "codex"), ("claude-opus-5", "claude"), ("grok-4.6", "pi")]
)
def test_model_routing(model: str, kind: str) -> None:
    assert reviewer.model_kind(model) == kind


def test_unknown_model_is_fatal() -> None:
    with pytest.raises(SystemExit):
        reviewer.model_kind("gemini-pro")


def test_codex_routes_through_an_explicit_provider_not_openai_base_url(
    monkeypatch: Any, clean_env: None, tmp_path: Path
) -> None:
    """Codex ignores OPENAI_BASE_URL, so the proxy must arrive as provider config.

    An env-var-only setup silently authenticates against api.openai.com instead
    of the proxy, which is why this asserts on the -c overrides specifically.
    """
    monkeypatch.setenv("OAI_PROXY_API_KEY", "secret")
    command, env = reviewer._agent_command("gpt-5.6-sol", tmp_path / "review.md")
    assert command[:4] == ["codex", "exec", "--model", "gpt-5.6-sol"]
    assert command[command.index("--output-last-message") + 1] == str(tmp_path / "review.md")
    overrides = [command[i + 1] for i, arg in enumerate(command) if arg == "-c"]
    provider = next(o.split("=", 1)[1] for o in overrides if o.startswith("model_provider="))
    table = next(o for o in overrides if o.startswith(f"model_providers.{provider}="))
    assert f'base_url="{reviewer.OAI_PROXY_URL}"' in table
    assert 'wire_api="responses"' in table
    assert 'env_key="OPENAI_API_KEY"' in table
    assert env["OPENAI_API_KEY"] == "secret"
    assert "OPENAI_BASE_URL" not in env


def test_claude_command_uses_anthropic_proxy(
    monkeypatch: Any, clean_env: None, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANT_PROXY_API_KEY", "secret")
    command, env = reviewer._agent_command("claude-opus-5", tmp_path / "review.md")
    assert command[0] == "claude"
    assert "--print" in command
    assert env["ANTHROPIC_BASE_URL"] == reviewer.ANT_PROXY_URL
    assert env["ANTHROPIC_API_KEY"] == "secret"


def test_pi_command_writes_xai_provider(monkeypatch: Any, clean_env: None, tmp_path: Path) -> None:
    monkeypatch.setenv("XAI_PROXY_API_KEY", "secret")
    command, env = reviewer._agent_command("grok-4.6", tmp_path / "review.md")
    assert command[0] == "pi"
    assert command[command.index("--provider") + 1] == "xai-proxy"
    config = Path(env["PI_CODING_AGENT_DIR"]).joinpath("models.json").read_text()
    assert reviewer.XAI_PROXY_URL in config
    assert "grok-4.6" in config


def test_missing_proxy_key_for_routed_family_is_fatal(monkeypatch: Any, clean_env: None) -> None:
    monkeypatch.setenv("ANT_PROXY_API_KEY", "wrong-family")
    with pytest.raises(SystemExit):
        reviewer._agent_command("gpt-5.6-sol", Path("/tmp/unused.md"))


@pytest.mark.parametrize(
    "model,kept", [("gpt-5.6-sol", "OPENAI_API_KEY"), ("claude-opus-5", "ANTHROPIC_API_KEY")]
)
def test_agent_env_drops_the_install_token_and_unrouted_keys(
    monkeypatch: Any, clean_env: None, tmp_path: Path, model: str, kept: str
) -> None:
    """The agent runs PR-authored code; it must not inherit a writable token."""
    monkeypatch.setenv("GH_TOKEN", "ghs_installation")
    monkeypatch.setenv("GITHUB_TOKEN", "ghs_actions")
    monkeypatch.setenv("ACTIONS_RUNTIME_TOKEN", "runtime")
    monkeypatch.setenv("OAI_PROXY_API_KEY", "oai")
    monkeypatch.setenv("ANT_PROXY_API_KEY", "ant")
    monkeypatch.setenv("XAI_PROXY_API_KEY", "xai")
    _, env = reviewer._agent_command(model, tmp_path / "review.md")
    assert "ghs_installation" not in env.values()
    assert not {"GH_TOKEN", "GITHUB_TOKEN", "ACTIONS_RUNTIME_TOKEN"} & set(env)
    assert not {"OAI_PROXY_API_KEY", "ANT_PROXY_API_KEY", "XAI_PROXY_API_KEY"} & set(env)
    assert kept in env


@pytest.mark.parametrize("model", ["gpt-5.6-sol", "claude-opus-5", "grok-4.6"])
def test_agent_cannot_reach_the_actions_control_files(
    monkeypatch: Any, clean_env: None, tmp_path: Path, model: str
) -> None:
    """The agent must not inherit the Actions control plane.

    `$GITHUB_OUTPUT` is the INPUT to the safety net that detects a missing
    review. Under `danger-full-access` the agent has a real shell, so an
    inherited path lets it `echo published=true >> $GITHUB_OUTPUT`: the
    launcher then refuses to publish (exit 3, nothing posted) but the
    workflow's `steps.review.outputs.published != 'true'` net does NOT fire.
    Green run, no review, no warning — the exact silent failure this whole
    mechanism exists to close, reachable BY the reviewed code's agent.

    `$GITHUB_ENV`, `$GITHUB_PATH`, `$GITHUB_STATE` and `$GITHUB_STEP_SUMMARY`
    are the same class: writes there mutate later steps of this job, or the
    job summary a human reads when the net does fire.

    The runner also exposes `_runner_file_commands/*` paths under keys outside
    the documented GITHUB_* set, so the value strip is asserted too — but only
    over an EXPLICIT runner env. Asserting `"file_commands" in v` over the
    ambient environment false-fails on a real runner, whose own control paths
    are present regardless of what this function does.

    Stripping them from the CHILD costs nothing: `_record_published` reads
    GITHUB_OUTPUT from the LAUNCHER's own os.environ, which is untouched —
    asserted directly in
    test_stripping_github_output_does_not_break_the_publication_signal.
    """
    runner_env = {
        "OAI_PROXY_API_KEY": "oai",
        "ANT_PROXY_API_KEY": "ant",
        "XAI_PROXY_API_KEY": "xai",
        "GITHUB_OUTPUT": "/runner/_temp/_runner_file_commands/set_output_abc",
        "GITHUB_ENV": "/runner/_temp/_runner_file_commands/set_env_abc",
        "GITHUB_PATH": "/runner/_temp/_runner_file_commands/add_path_abc",
        "GITHUB_STEP_SUMMARY": "/runner/_temp/_runner_file_commands/step_summary_abc",
        "GITHUB_STATE": "/runner/_temp/_runner_file_commands/save_state_abc",
        # Actions also exposes a control path under keys outside the GITHUB_*
        # set; the name list alone does not cover those.
        "RUNNER_TEMP_SUMMARY": "/runner/_temp/_runner_file_commands/artifacts_abc",
        # An ordinary inherited variable, to pin the filter's blast radius:
        # the child still needs PATH to find its harness binary at all.
        "PATH": "/usr/local/bin:/usr/bin",
    }
    monkeypatch.setattr(reviewer.os, "environ", runner_env)
    _, env = reviewer._agent_command(model, tmp_path / "review.md")
    assert not {
        "GITHUB_OUTPUT",
        "GITHUB_ENV",
        "GITHUB_PATH",
        "GITHUB_STEP_SUMMARY",
        "GITHUB_STATE",
    } & set(env)
    # Not merely absent by name — no key may carry the path. Safe to assert
    # over every value here because os.environ IS runner_env for this call.
    assert not [v for v in env.values() if "file_commands" in v]
    assert env["PATH"] == "/usr/local/bin:/usr/bin"


@pytest.mark.parametrize(
    "name",
    ["GITHUB_OUTPUT", "GITHUB_ENV", "GITHUB_PATH", "GITHUB_STEP_SUMMARY", "GITHUB_STATE"],
)
def test_control_variables_are_stripped_by_name_not_only_by_path(
    monkeypatch: Any, clean_env: None, tmp_path: Path, name: str
) -> None:
    """Each control variable is covered by the NAME list on its own.

    The value strip keys off `_runner_file_commands`, an undocumented internal
    of the runner's layout. If that directory is ever renamed the name list is
    the only remaining cover, so it is pinned here with a value the value strip
    cannot match — drop a name from `_STRIPPED_ENV` and this fails.
    """
    monkeypatch.setattr(
        reviewer.os,
        "environ",
        {"OAI_PROXY_API_KEY": "oai", name: "/runner/_temp/control-plane-abc"},
    )
    _, env = reviewer._agent_command("gpt-5.6-sol", tmp_path / "review.md")
    assert name not in env


def test_stripping_github_output_does_not_break_the_publication_signal(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Verified, not assumed: the launcher reads GITHUB_OUTPUT from its OWN env.

    This is the functional check behind the strip. `_record_published` never
    consults the child env, so removing the var from the agent's environment
    cannot break the net's positive signal.
    """
    output = tmp_path / "gh-output"
    output.write_text("")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("OAI_PROXY_API_KEY", "oai")
    _, env = reviewer._agent_command("gpt-5.6-sol", tmp_path / "review.md")
    assert "GITHUB_OUTPUT" not in env  # gone from the child
    reviewer._record_published("https://github.test/c/1")  # still works in the parent
    assert "published=true" in output.read_text()


def test_checkout_does_not_leave_a_git_credential_on_disk() -> None:
    """Env stripping does not reach `.git/config`.

    actions/checkout defaults `persist-credentials: true`, which writes
    `http.https://github.com/.extraheader` — HTTP basic auth carrying the
    workflow GITHUB_TOKEN — into the checkout on disk. The launcher strips
    credentials from the agent's ENV, but an unsandboxed agent just runs
    `cat .git/config`. Bounded here by `contents: read` on a public repo, but
    "strips every writable credential" is only true if this is off.
    """
    job = yaml.safe_load(_WORKFLOW.read_text())["jobs"]["review"]
    checkout = next(
        step for step in job["steps"] if str(step.get("uses", "")).startswith("actions/checkout")
    )
    assert checkout["with"]["persist-credentials"] is False


def test_prompt_pins_the_reviewed_range_to_base_and_head() -> None:
    prompt = reviewer._prompt("eumemic/aios", "7", "headsha", "basesha")
    assert "git diff basesha...headsha" in prompt
    assert reviewer.ARTIFACT_HEADING in prompt


def test_run_agent_extracts_heading_from_stdout(monkeypatch: Any, clean_env: None) -> None:
    monkeypatch.setenv("ANT_PROXY_API_KEY", "secret")
    completed = subprocess.CompletedProcess([], 0, "preamble\n### Code review\n\nFinding.", "")
    monkeypatch.setattr(reviewer.subprocess, "run", lambda *args, **kwargs: completed)
    assert reviewer.run_agent("claude-opus-5", "prompt", 10) == "### Code review\n\nFinding."


def test_run_agent_takes_the_final_heading_not_an_echoed_one(
    monkeypatch: Any, clean_env: None
) -> None:
    """Pi and Claude stdout carries tool activity, which can quote the heading."""
    monkeypatch.setenv("XAI_PROXY_API_KEY", "secret")
    noisy = (
        'grep "### Code review" scripts/eumemic_bot_review.py\n'
        "### Code review\nARTIFACT_HEADING = ...\n"
        "### Code review\n\nThe real finding.\n"
    )
    monkeypatch.setattr(
        reviewer.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 0, noisy, "")
    )
    assert reviewer.run_agent("grok-4.6", "prompt", 10) == "### Code review\n\nThe real finding."


def test_run_agent_prefers_codex_last_message(monkeypatch: Any, clean_env: None) -> None:
    monkeypatch.setenv("OAI_PROXY_API_KEY", "secret")

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        path = Path(command[command.index("--output-last-message") + 1])
        path.write_text("chatty\n### Code review\n\nLooks good.")
        return subprocess.CompletedProcess(command, 0, "event output", "")

    monkeypatch.setattr(reviewer.subprocess, "run", run)
    assert reviewer.run_agent("gpt-5.6-sol", "prompt", 10).endswith("Looks good.")


def test_run_agent_reports_partial_output_on_timeout(
    monkeypatch: Any, clean_env: None, capsys: Any
) -> None:
    """A 15-minute timeout is the likeliest failure; its log must not be empty."""
    monkeypatch.setenv("ANT_PROXY_API_KEY", "secret")

    def run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired("claude", 10, output="got this far", stderr="warned")

    monkeypatch.setattr(reviewer.subprocess, "run", run)
    with pytest.raises(SystemExit):
        reviewer.run_agent("claude-opus-5", "prompt", 10)
    captured = capsys.readouterr()
    assert "got this far" in captured.out
    assert "warned" in captured.err


def test_missing_artifact_heading_is_fatal(monkeypatch: Any, clean_env: None) -> None:
    monkeypatch.setenv("ANT_PROXY_API_KEY", "secret")
    monkeypatch.setattr(
        reviewer.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, "I refuse to follow format", ""),
    )
    with pytest.raises(SystemExit):
        reviewer.run_agent("claude-opus-5", "prompt", 10)


def test_pin_checkout_rejects_a_tree_that_is_not_the_pr_head(monkeypatch: Any) -> None:
    monkeypatch.setattr(reviewer, "_git", lambda *args: _ok("deadbeefotherhead\n"))
    with pytest.raises(SystemExit):
        reviewer._pin_checkout("abc123", "base456")


def test_pin_checkout_fetches_a_missing_base(monkeypatch: Any) -> None:
    calls: list[tuple[str, ...]] = []

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[0] == "rev-parse":
            return _ok("abc123full\n")
        if args[0] == "cat-file":
            # Absent before the fetch, present after it.
            missing = not any(call[0] == "fetch" for call in calls)
            return subprocess.CompletedProcess([], 1 if missing else 0, "", "")
        return _ok()

    monkeypatch.setattr(reviewer, "_git", git)
    reviewer._pin_checkout("abc123", "base456")
    assert ("fetch", "--no-tags", "--quiet", "origin", "base456") in calls


def test_pin_checkout_fails_when_the_base_cannot_be_fetched(monkeypatch: Any) -> None:
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        if args[0] == "rev-parse":
            return _ok("abc123full\n")
        return subprocess.CompletedProcess([], 1, "", "no such object")

    monkeypatch.setattr(reviewer, "_git", git)
    with pytest.raises(SystemExit):
        reviewer._pin_checkout("abc123", "base456")


_DIFF_EVIDENCE = (137, "a" * 64)


def _review_env(monkeypatch: Any) -> None:
    for key, value in {
        "GH_TOKEN": "token",
        "REPO": "eumemic/aios",
        "PR_NUMBER": "1",
        "HEAD_SHA": "abc123",
        "BASE_SHA": "base456",
        "REVIEW_MODEL": "gpt-5.6-sol",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(reviewer, "_git", lambda *args: _ok("abc123full\n"))
    # The real `git diff base...head` cannot run against fixture SHAs. Stubbing
    # it keeps these tests about publication, not about diff computation —
    # which test_diff_evidence_is_computed_from_the_real_diff covers directly.
    monkeypatch.setattr(reviewer, "diff_evidence", lambda *args: _DIFF_EVIDENCE)


def test_main_posts_and_verifies_marker(monkeypatch: Any, capsys: Any) -> None:
    _review_env(monkeypatch)
    monkeypatch.setattr(reviewer, "run_agent", lambda *a, **k: "### Code review\n\nPass.")
    posted: dict[str, str] = {}

    def github(method: str, url: str, token: str, body: dict[str, str]) -> dict[str, str]:
        posted.update(body)
        assert url.endswith("/repos/eumemic/aios/issues/1/comments")
        return {"html_url": "https://github.test/comment/1", "body": body["body"]}

    monkeypatch.setattr(reviewer, "_github_request", github)
    reviewer.main()
    assert "<!-- eumemic-bot-review:abc123 -->" in posted["body"]
    assert "posted and verified" in capsys.readouterr().out


def test_main_pins_the_checkout_before_reviewing_anything(monkeypatch: Any) -> None:
    """Reviewing the WRONG TREE is the one failure worse than no review.

    The reviewer's mutant that deleted the `_pin_checkout(...)` call from
    `main()` SURVIVED all 44 tests: every pin test called the function
    directly, none asserted main invokes it. A verdict rendered against an
    unpinned tree is authoritative and about different code.

    Pinning must also happen BEFORE the agent runs, not after — a review of the
    wrong tree that is later detected has already burned the run.
    """
    _review_env(monkeypatch)
    calls: list[tuple[str, str]] = []
    order: list[str] = []

    def pin(head: str, base: str) -> None:
        calls.append((head, base))
        order.append("pin")

    monkeypatch.setattr(reviewer, "_pin_checkout", pin)

    def agent(*a: Any, **k: Any) -> str:
        order.append("agent")
        return "### Code review\n\nPass."

    monkeypatch.setattr(reviewer, "run_agent", agent)
    monkeypatch.setattr(
        reviewer,
        "_github_request",
        lambda method, url, token, body: {
            "html_url": "https://github.test/c/1",
            "body": body["body"],
        },
    )
    reviewer.main()
    assert calls == [("abc123", "base456")], "main() did not pin the checkout to the PR head"
    assert order == ["pin", "agent"], "the tree must be pinned before the agent reviews it"


def test_main_fails_when_github_does_not_echo_the_marker(monkeypatch: Any) -> None:
    """An unverified post is the silent-miss failure this launcher exists to catch."""
    _review_env(monkeypatch)
    monkeypatch.setattr(reviewer, "run_agent", lambda *a, **k: "### Code review\n\nPass.")
    monkeypatch.setattr(
        reviewer,
        "_github_request",
        lambda *args, **kwargs: {"html_url": "https://github.test/c/1", "body": "truncated"},
    )
    with pytest.raises(SystemExit):
        reviewer.main()


def test_workflow_pins_head_and_base_and_keeps_no_aios_session_config() -> None:
    text = _WORKFLOW.read_text()
    workflow = yaml.safe_load(text)
    job = workflow["jobs"]["review"]
    checkout = job["steps"][0]["with"]
    review = next(step for step in job["steps"] if step.get("id") == "review")
    assert checkout["ref"] == "${{ github.event.pull_request.head.sha }}"
    assert checkout["fetch-depth"] == 0
    assert review["env"]["HEAD_SHA"] == "${{ github.event.pull_request.head.sha }}"
    assert review["env"]["BASE_SHA"] == "${{ github.event.pull_request.base.sha }}"
    # The launcher's own timeout has to fire before the runner kills the job,
    # or continue-on-error and the "did not post" summary are both lost.
    assert int(review["env"]["REVIEW_TIMEOUT_SECONDS"]) < job["timeout-minutes"] * 60
    assert job["timeout-minutes"] <= 20
    assert "AIOS_API_KEY" not in text
    assert "DEV_REVIEW_AGENT_ID" not in text
    assert "/v1/sessions" not in _SCRIPT.read_text()
    assert reviewer.DEFAULT_MODEL in workflow["env"]["REVIEW_MODEL"]


def test_workflow_never_fails_the_pr_check_on_an_ops_miss() -> None:
    """The channel stays advisory: an ops miss must not block a merge.

    Retained deliberately. What changed is the WORDING, not the policy — since
    the check cannot go red on a bad verdict, the comment now says so in its
    first line instead of rendering an unqualified authoritative heading.
    """
    job = yaml.safe_load(_WORKFLOW.read_text())["jobs"]["review"]
    steps = {step["id"]: step for step in job["steps"] if "id" in step}
    assert all(steps[name]["continue-on-error"] for name in ("harness", "app", "review"))
    summary = next(step for step in job["steps"] if "GITHUB_STEP_SUMMARY" in step.get("run", ""))
    assert summary["if"].startswith("always()")


def test_published_comment_is_labelled_advisory_not_an_authoritative_gate(
    monkeypatch: Any,
) -> None:
    """An unqualified `### Code review` reads as a merge gate that does not exist.

    continue-on-error means green regardless of verdict, so the body must say
    non-blocking. This is the wording half of the advisory decision; the gate
    half is the inspection-evidence refusal.
    """
    _review_env(monkeypatch)
    monkeypatch.setattr(reviewer, "run_agent", lambda *a, **k: "### Code review\n\nPass.")
    posted: dict[str, str] = {}
    monkeypatch.setattr(
        reviewer,
        "_github_request",
        lambda method, url, token, body: (
            posted.update(body),
            {"html_url": "https://github.test/c/1", "body": body["body"]},
        )[1],
    )
    reviewer.main()
    body = posted["body"]
    assert reviewer.ADVISORY_BANNER in body
    assert "non-blocking" in body.lower()
    # The banner has to precede the verdict, or a reader skims the heading first.
    assert body.index(reviewer.ADVISORY_BANNER) < body.index("### Code review")


# --------------------------------------------------------------------------
# Publication gating: exit status is NOT evidence the agent reviewed anything.
# `codex` exits 0 when bubblewrap blocks every command (run 34196306433), so
# these are the two states that must never reach GitHub.
# --------------------------------------------------------------------------


def _agent_returning(stdout: str, returncode: int = 0) -> Any:
    def run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], returncode, stdout, "")

    return run


def _good_artifact(lines: int = 137, digest: str = "a" * 64) -> str:
    return (
        f"### Code review\n\nA real finding.\n\n<!-- inspected: lines={lines} sha256={digest} -->"
    )


def test_agent_exiting_nonzero_must_not_post(monkeypatch: Any, clean_env: None) -> None:
    """A crashed agent's output is not a review, however well-formed it looks."""
    monkeypatch.setenv("ANT_PROXY_API_KEY", "secret")
    monkeypatch.setattr(reviewer.subprocess, "run", _agent_returning(_good_artifact(), 1))
    with pytest.raises(SystemExit) as exc:
        reviewer.run_agent("claude-opus-5", "prompt", 10, _DIFF_EVIDENCE)
    assert exc.value.code != 0


def test_agent_exiting_zero_having_run_no_commands_must_not_post(
    monkeypatch: Any, clean_env: None
) -> None:
    """The production failure, exactly: exit 0, polished verdict, zero bytes read.

    The real run emitted a confession; the dangerous variant is the polite one,
    so this asserts on the LGTM shape. Nothing about exit status distinguishes
    them — only the absent evidence line does.
    """
    monkeypatch.setenv("OAI_PROXY_API_KEY", "secret")
    lgtm = "### Code review\n\nNo actionable findings in this range. LGTM."
    monkeypatch.setattr(reviewer.subprocess, "run", _agent_returning(lgtm, 0))
    with pytest.raises(SystemExit) as exc:
        reviewer.run_agent("gpt-5.6-sol", "prompt", 10, _DIFF_EVIDENCE)
    assert exc.value.code == reviewer.NO_EVIDENCE_EXIT_CODE


def test_no_evidence_is_a_distinct_loud_state_not_an_ordinary_failure(
    monkeypatch: Any, clean_env: None, capsys: Any
) -> None:
    monkeypatch.setenv("OAI_PROXY_API_KEY", "secret")
    bwrap = (
        "### Code review\n\nUnable to complete the review: every read-only shell command "
        "failed with `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`."
    )
    monkeypatch.setattr(reviewer.subprocess, "run", _agent_returning(bwrap, 0))
    with pytest.raises(SystemExit) as exc:
        reviewer.run_agent("gpt-5.6-sol", "prompt", 10, _DIFF_EVIDENCE)
    assert exc.value.code == reviewer.NO_EVIDENCE_EXIT_CODE
    assert exc.value.code != 1  # distinct from every other FATAL
    captured = capsys.readouterr()
    assert reviewer.NO_EVIDENCE_BANNER in captured.err
    assert "::error" in captured.out  # surfaces in the Actions UI, not just stderr


def test_fabricated_inspection_evidence_must_not_post(monkeypatch: Any, clean_env: None) -> None:
    """A guessed evidence line is not evidence; the launcher recomputes both."""
    monkeypatch.setenv("OAI_PROXY_API_KEY", "secret")
    monkeypatch.setattr(
        reviewer.subprocess, "run", _agent_returning(_good_artifact(999, "b" * 64), 0)
    )
    with pytest.raises(SystemExit) as exc:
        reviewer.run_agent("gpt-5.6-sol", "prompt", 10, _DIFF_EVIDENCE)
    assert exc.value.code == reviewer.NO_EVIDENCE_EXIT_CODE


@pytest.mark.parametrize(
    "verdict",
    [
        "No actionable findings. LGTM.",
        "BLOCKING: this must not merge — unguarded SQL interpolation at src/x.py:12.",
    ],
)
def test_a_genuinely_inspected_review_publishes_whatever_its_verdict(
    monkeypatch: Any, clean_env: None, verdict: str
) -> None:
    """Three-way discrimination: the gate is on INSPECTION, never on the verdict.

    A guard only ever seen refusing is indistinguishable from one that refuses
    everything, so PASS and FAIL verdicts both have to get through.
    """
    monkeypatch.setenv("OAI_PROXY_API_KEY", "secret")
    artifact = f"### Code review\n\n{verdict}\n\n<!-- inspected: lines=137 sha256={'a' * 64} -->"
    monkeypatch.setattr(reviewer.subprocess, "run", _agent_returning(artifact, 0))
    assert verdict in reviewer.run_agent("gpt-5.6-sol", "prompt", 10, _DIFF_EVIDENCE)


def test_a_matching_line_count_alone_is_NOT_evidence(monkeypatch: Any, clean_env: None) -> None:
    """The line count is public: it does NOT prove the agent read the checkout.

    Measured on this very PR: `https://github.com/eumemic/aios/pull/2404.diff`
    has the identical 2243 line count with zero checkout access (different
    bytes, so a different digest). An agent with network but no working shell —
    exactly the blocked-agent state this gate exists to catch — can obtain it.
    So the count is a low-entropy, independently-obtainable integer and cannot
    be an accepting channel on its own. Only the digest is sound.

    This test previously asserted the OPPOSITE (`..._accepts_a_matching_line_
    count_alone`); it codified the weakness as intended behaviour.
    """
    monkeypatch.setenv("OAI_PROXY_API_KEY", "secret")
    artifact = "### Code review\n\nOK.\n\n<!-- inspected: lines=137 sha256=" + "c" * 64 + " -->"
    monkeypatch.setattr(reviewer.subprocess, "run", _agent_returning(artifact, 0))
    with pytest.raises(SystemExit) as exc:
        reviewer.run_agent("gpt-5.6-sol", "prompt", 10, _DIFF_EVIDENCE)
    assert exc.value.code == reviewer.NO_EVIDENCE_EXIT_CODE


def test_the_published_diff_line_count_paired_with_a_forged_digest_is_refused() -> None:
    """The concrete attack, with this PR's real numbers."""
    public_line_count = 2243  # from pull/2404.diff, fetched without any checkout
    real = (2243, "085b04f85930cebf3d476a4905763f2c6d1ede92e9d2c1a060fe254d499dfd3b")
    artifact = f"### Code review\n\nLGTM.\n\n<!-- inspected: lines={public_line_count} sha256={'f' * 64} -->"
    with pytest.raises(SystemExit) as exc:
        reviewer.require_inspection_evidence(artifact, real)
    assert exc.value.code == reviewer.NO_EVIDENCE_EXIT_CODE


def test_an_abbreviated_digest_is_refused(monkeypatch: Any, clean_env: None) -> None:
    """A 16-hex prefix is guessable-ish and was accepted; require all 64."""
    with pytest.raises(SystemExit) as exc:
        reviewer.require_inspection_evidence(
            "### Code review\n\n<!-- inspected: lines=137 sha256=aaaaaaaaaaaaaaaa -->",
            _DIFF_EVIDENCE,
        )
    assert exc.value.code == reviewer.NO_EVIDENCE_EXIT_CODE


def test_the_full_digest_is_what_publishes(monkeypatch: Any, clean_env: None) -> None:
    """The permit half: the sound channel still lets a real review through.

    A guard only ever seen refusing is indistinguishable from one that refuses
    everything, so the accepting case is asserted alongside every refusal.
    """
    monkeypatch.setenv("OAI_PROXY_API_KEY", "secret")
    # Deliberately a WRONG line count with the RIGHT digest: the digest alone
    # must suffice, so a benign `wc -l` formatting difference cannot suppress a
    # genuine review.
    artifact = (
        "### Code review\n\nReal finding.\n\n<!-- inspected: lines=999 sha256=" + "a" * 64 + " -->"
    )
    monkeypatch.setattr(reviewer.subprocess, "run", _agent_returning(artifact, 0))
    assert "Real finding." in reviewer.run_agent("gpt-5.6-sol", "prompt", 10, _DIFF_EVIDENCE)


def test_evidence_regex_will_not_even_match_a_short_digest() -> None:
    """COUPLING GUARD for the `==` / `startswith` equivalent-mutant argument.

    Not merely defence in depth. `require_inspection_evidence` documents that
    mutating `claimed_digest == expected_digest` into
    `expected_digest.startswith(claimed_digest)` is an EQUIVALENT mutant and
    may be left unkilled. That argument is TRUE ONLY WHILE THIS TEST PASSES:
    the equivalence rests entirely on `_EVIDENCE_RE` pinning exactly 64 hex, so
    both operands are the same length. Widen the bound and `startswith` becomes
    a live prefix-acceptance vulnerability that would admit a 1-character
    digest as proof of inspection.

    So this is the test that licenses ignoring that survivor. Deleting it as
    redundant silently converts an accepted equivalent mutant into an unguarded
    fail-open. The reviewer's mutant widening this (64 -> 1) SURVIVED the old
    suite, which is why the width is asserted directly.
    """
    assert reviewer._EVIDENCE_RE.search("<!-- inspected: lines=1 sha256=" + "a" * 64 + " -->")
    for short in ("a" * 16, "a" * 63, "a" * 8):
        assert not reviewer._EVIDENCE_RE.search(f"<!-- inspected: lines=1 sha256={short} -->"), (
            f"regex matched a {len(short)}-char digest; the width is the guard"
        )
    # The equivalence argument in require_inspection_evidence's docstring is
    # only sound while the regex cannot admit a SHORTER operand than the real
    # digest. Pin that directly: a prefix must never be accepted as the whole.
    expected = hashlib.sha256(b"some diff").hexdigest()
    artifact = f"### Code review\n\n<!-- inspected: lines=1 sha256={expected[:8]} -->"
    with pytest.raises(SystemExit) as exc:
        reviewer.require_inspection_evidence(artifact, (1, expected))
    assert exc.value.code == reviewer.NO_EVIDENCE_EXIT_CODE


def test_main_refuses_to_post_when_the_agent_shows_no_inspection(monkeypatch: Any) -> None:
    """End to end: the refusal reaches main, so nothing is POSTed."""
    _review_env(monkeypatch)
    monkeypatch.setattr(reviewer, "diff_evidence", lambda *a: _DIFF_EVIDENCE)
    monkeypatch.setattr(
        reviewer.subprocess,
        "run",
        _agent_returning("### Code review\n\nLGTM, nothing to flag.", 0),
    )
    monkeypatch.setenv("OAI_PROXY_API_KEY", "secret")

    def must_not_post(*args: Any, **kwargs: Any) -> dict[str, str]:
        raise AssertionError("posted a review with no evidence of inspection")

    monkeypatch.setattr(reviewer, "_github_request", must_not_post)
    with pytest.raises(SystemExit) as exc:
        reviewer.main()
    assert exc.value.code == reviewer.NO_EVIDENCE_EXIT_CODE


def test_prompt_demands_evidence_derivable_only_from_the_diff() -> None:
    prompt = reviewer._prompt("eumemic/aios", "7", "headsha", "basesha")
    assert "sha256sum" in prompt
    assert "wc -l" in prompt
    assert reviewer.EVIDENCE_TEMPLATE in prompt
    # The expected values must NOT be in the prompt, or a blocked agent can
    # parrot them back and the evidence proves nothing.
    assert "137" not in prompt


def test_diff_evidence_is_computed_from_the_real_diff(monkeypatch: Any) -> None:
    """The launcher-side half must be derived, not trusted from the agent."""
    payload = b"diff --git a/x b/x\n+one\n+two\n"

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        assert command[:3] == ["git", "--no-pager", "diff"]
        assert command[3] == "base456...abc123"
        return subprocess.CompletedProcess(command, 0, payload, b"")

    monkeypatch.setattr(reviewer.subprocess, "run", run)
    lines, digest = reviewer.diff_evidence("base456", "abc123")
    assert lines == 3
    assert digest == hashlib.sha256(payload).hexdigest()


def test_diff_evidence_refuses_an_empty_diff(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        reviewer.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, b"   \n", b""),
    )
    with pytest.raises(SystemExit):
        reviewer.diff_evidence("base456", "abc123")


def test_workflow_safety_net_keys_on_publication_not_step_outcome() -> None:
    """The net was SKIPPED in the one real miss: the step outcome was 'success'.

    So it must not key on step outcome at all — a review that exits 0 having
    read nothing is exactly the case where outcome-keying fails.
    """
    job = yaml.safe_load(_WORKFLOW.read_text())["jobs"]["review"]
    summary = next(step for step in job["steps"] if "GITHUB_STEP_SUMMARY" in step.get("run", ""))
    condition = summary["if"]
    assert condition.startswith("always()")
    assert "steps.review.outputs.published != 'true'" in condition
    assert "outcome == 'failure'" not in condition


def test_safety_net_also_fires_when_the_review_step_never_ran() -> None:
    """Harness/token failure skips the review step; outputs are then empty.

    `!= 'true'` covers skipped, failed and "ran but refused" alike — the net
    must not depend on the step having produced any outcome at all.
    """
    job = yaml.safe_load(_WORKFLOW.read_text())["jobs"]["review"]
    summary = next(step for step in job["steps"] if "GITHUB_STEP_SUMMARY" in step.get("run", ""))
    review = next(step for step in job["steps"] if step.get("id") == "review")
    # The review step is conditional, so "skipped" is a reachable state.
    assert "if" in review
    assert summary["if"] == "always() && steps.review.outputs.published != 'true'"


def test_launcher_signals_publication_only_after_github_confirms(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """`published=true` is the net's input, so it must mean a VERIFIED post."""
    output = tmp_path / "gh-output"
    output.write_text("")
    _review_env(monkeypatch)
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(reviewer, "run_agent", lambda *a, **k: "### Code review\n\nPass.")
    monkeypatch.setattr(
        reviewer,
        "_github_request",
        lambda method, url, token, body: {
            "html_url": "https://github.test/c/1",
            "body": body["body"],
        },
    )
    reviewer.main()
    assert "published=true" in output.read_text()


def test_launcher_does_not_signal_publication_when_the_marker_is_not_echoed(
    monkeypatch: Any, tmp_path: Path
) -> None:
    output = tmp_path / "gh-output"
    output.write_text("")
    _review_env(monkeypatch)
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(reviewer, "run_agent", lambda *a, **k: "### Code review\n\nPass.")
    monkeypatch.setattr(
        reviewer,
        "_github_request",
        lambda *a, **k: {"html_url": "https://github.test/c/1", "body": "truncated"},
    )
    with pytest.raises(SystemExit):
        reviewer.main()
    assert "published=true" not in output.read_text()


def test_workflow_selects_a_sandbox_mode_that_can_actually_execute() -> None:
    """The environment fix must be a mode that RUNS, not merely a mode that is set.

    Measured against codex-cli 0.154.0 with the `codex sandbox` probe: both
    `read-only` and `workspace-write` route through codex's vendored bwrap and
    fail with "No permissions to create a new namespace" — zero commands
    execute and codex still exits 0. Only `danger-full-access` executes. An
    earlier draft of this fix selected `workspace-write`; this assertion is
    what caught that it would have changed nothing.

    Deliberately NOT satisfied by `apt-get install bubblewrap`: codex invokes
    its own vendored bwrap and never the one on PATH (verified by planting a
    fake bwrap first in PATH — it is never executed), so the distro package is
    inert.
    """
    job = yaml.safe_load(_WORKFLOW.read_text())["jobs"]["review"]
    review = next(step for step in job["steps"] if step.get("id") == "review")
    mode_expr = review["env"].get("REVIEW_SANDBOX_MODE", "")
    assert mode_expr, "no sandbox mode selected; the default read-only executes nothing"
    # The effective default (the `||` fallback) is what runs when the repo
    # variable is unset, which is the state that shipped the broken run.
    assert "danger-full-access" in mode_expr, (
        f"sandbox mode {mode_expr!r} routes through the vendored bwrap and executes "
        "nothing on a hosted runner"
    )
    assert "workspace-write" not in mode_expr


def test_workflow_does_not_rely_on_apt_installed_bubblewrap() -> None:
    """Pins the measured fact that the obvious remedy is inert.

    codex uses its vendored bwrap regardless of PATH, so installing the distro
    package would look like a fix and change nothing. If someone adds it back
    believing it fixes the sandbox, this fails and points at the measurement.
    """
    job = yaml.safe_load(_WORKFLOW.read_text())["jobs"]["review"]
    install = next(step for step in job["steps"] if step.get("id") == "harness")["run"]
    installs = [
        line
        for line in install.splitlines()
        if "bubblewrap" in line and not line.strip().startswith("#")
    ]
    assert not installs, f"apt-installed bubblewrap is inert for codex: {installs}"


def test_sandbox_mode_is_configurable_and_validated(monkeypatch: Any, clean_env: None) -> None:
    monkeypatch.setenv("OAI_PROXY_API_KEY", "secret")
    monkeypatch.delenv("REVIEW_SANDBOX_MODE", raising=False)
    command, _ = reviewer._agent_command("gpt-5.6-sol", Path("/tmp/a.md"))
    assert command[command.index("--sandbox") + 1] == reviewer.DEFAULT_SANDBOX
    monkeypatch.setenv("REVIEW_SANDBOX_MODE", "workspace-write")
    command, _ = reviewer._agent_command("gpt-5.6-sol", Path("/tmp/a.md"))
    assert command[command.index("--sandbox") + 1] == "workspace-write"
    monkeypatch.setenv("REVIEW_SANDBOX_MODE", "no-such-mode")
    with pytest.raises(SystemExit):
        reviewer._agent_command("gpt-5.6-sol", Path("/tmp/a.md"))


def test_workflow_installs_the_harness_for_every_routed_prefix() -> None:
    job = yaml.safe_load(_WORKFLOW.read_text())["jobs"]["review"]
    install = next(step for step in job["steps"] if step.get("id") == "harness")["run"]
    assert "@openai/codex" in install
    assert "@anthropic-ai/claude-code" in install
    assert "@mariozechner/pi-coding-agent" in install
    # Routed, not all three: switching model must change what gets installed.
    for prefix in ("gpt-*)", "claude-*)", "grok-*)"):
        assert prefix in install
