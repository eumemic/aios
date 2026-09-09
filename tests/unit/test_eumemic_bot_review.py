"""Unit tests for the local coding-agent review launcher."""

from __future__ import annotations

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
    assert command[command.index("--sandbox") + 1] == "danger-full-access"
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


def test_persisted_push_credential_is_unset_before_the_agent_runs(monkeypatch: Any) -> None:
    """actions/checkout stores the workflow token in .git/config, out of env reach."""
    calls: list[tuple[str, ...]] = []

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if "--get-regexp" in args:
            return _ok("http.https://github.com/.extraheader\n")
        return _ok()

    monkeypatch.setattr(reviewer, "_git", git)
    reviewer._drop_persisted_git_credentials()
    assert ("config", "--local", "--unset-all", "http.https://github.com/.extraheader") in calls


def test_main_scrubs_the_git_credential_before_handing_the_tree_to_the_agent(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Ordering is the point: _pin_checkout may fetch, the agent must not be able to."""
    _agent_env(monkeypatch, tmp_path)
    order: list[str] = []

    def scrub() -> None:
        order.append("scrub")

    def agent(*args: Any) -> str:
        order.append("agent")
        return "### Code review\n\nPass."

    monkeypatch.setattr(reviewer, "_drop_persisted_git_credentials", scrub)
    monkeypatch.setattr(reviewer, "run_agent", agent)
    reviewer.run_agent_phase()
    assert order == ["scrub", "agent"]


def _agent_env(monkeypatch: Any, tmp_path: Path) -> Path:
    artifact = tmp_path / "review.md"
    for key, value in {
        "REPO": "eumemic/aios",
        "PR_NUMBER": "1",
        "HEAD_SHA": "abc123",
        "BASE_SHA": "base456",
        "REVIEW_MODEL": "gpt-5.6-sol",
        "REVIEW_ARTIFACT_PATH": str(artifact),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        reviewer, "_git", lambda *args: _ok("abc123full\n" if args[0] == "rev-parse" else "")
    )
    return artifact


def _publish_env(monkeypatch: Any, tmp_path: Path) -> Path:
    artifact = tmp_path / "review.md"
    artifact.write_text("### Code review\n\nPass.\n")
    for key, value in {
        "GH_TOKEN": "token",
        "REPO": "eumemic/aios",
        "PR_NUMBER": "1",
        "HEAD_SHA": "abc123",
        "REVIEW_ARTIFACT_PATH": str(artifact),
    }.items():
        monkeypatch.setenv(key, value)
    return artifact


def test_agent_phase_rejects_gh_token_before_launch(monkeypatch: Any, tmp_path: Path) -> None:
    _agent_env(monkeypatch, tmp_path)
    monkeypatch.setenv("GH_TOKEN", "must-not-exist")
    monkeypatch.setattr(
        reviewer, "run_agent", lambda *args: pytest.fail("agent must not be launched")
    )
    with pytest.raises(SystemExit):
        reviewer.run_agent_phase()


def test_agent_phase_writes_artifact_after_agent_returns(monkeypatch: Any, tmp_path: Path) -> None:
    artifact = _agent_env(monkeypatch, tmp_path)

    def agent(*args: Any) -> str:
        assert not artifact.exists()
        return "### Code review\n\nPass."

    monkeypatch.setattr(reviewer, "run_agent", agent)
    reviewer.run_agent_phase()
    assert artifact.read_text() == "### Code review\n\nPass.\n"


def test_publish_phase_posts_and_verifies_marker(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    _publish_env(monkeypatch, tmp_path)
    posted: dict[str, str] = {}

    def github(method: str, url: str, token: str, body: dict[str, str]) -> dict[str, str]:
        posted.update(body)
        assert url.endswith("/repos/eumemic/aios/issues/1/comments")
        return {"html_url": "https://github.test/comment/1", "body": body["body"]}

    monkeypatch.setattr(reviewer, "_github_request", github)
    reviewer.run_publish_phase()
    assert "<!-- eumemic-bot-review:abc123 -->" in posted["body"]
    assert "posted and verified" in capsys.readouterr().out


def test_publish_phase_fails_when_github_does_not_echo_the_marker(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """An unverified post is the silent-miss failure this launcher exists to catch."""
    _publish_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        reviewer,
        "_github_request",
        lambda *args, **kwargs: {"html_url": "https://github.test/c/1", "body": "truncated"},
    )
    with pytest.raises(SystemExit):
        reviewer.run_publish_phase()


def test_workflow_pins_head_and_base_and_keeps_no_aios_session_config() -> None:
    text = _WORKFLOW.read_text()
    workflow = yaml.safe_load(text)
    job = workflow["jobs"]["review"]
    checkout = job["steps"][0]["with"]
    agent = next(step for step in job["steps"] if step.get("id") == "agent")
    assert checkout["ref"] == "${{ github.event.pull_request.head.sha }}"
    assert checkout["fetch-depth"] == 0
    assert agent["env"]["HEAD_SHA"] == "${{ github.event.pull_request.head.sha }}"
    assert agent["env"]["BASE_SHA"] == "${{ github.event.pull_request.base.sha }}"
    # The launcher's own timeout has to fire before the runner kills the job,
    # or continue-on-error and the "did not post" summary are both lost.
    assert int(agent["env"]["REVIEW_TIMEOUT_SECONDS"]) < job["timeout-minutes"] * 60
    assert job["timeout-minutes"] <= 20
    assert "AIOS_API_KEY" not in text
    assert "DEV_REVIEW_AGENT_ID" not in text
    assert "/v1/sessions" not in _SCRIPT.read_text()
    assert reviewer.DEFAULT_MODEL in workflow["env"]["REVIEW_MODEL"]


def test_workflow_never_fails_the_pr_check_on_an_ops_miss() -> None:
    job = yaml.safe_load(_WORKFLOW.read_text())["jobs"]["review"]
    steps = {step["id"]: step for step in job["steps"] if "id" in step}
    assert all(steps[name]["continue-on-error"] for name in ("harness", "agent", "app", "publish"))
    summary = next(step for step in job["steps"] if "GITHUB_STEP_SUMMARY" in step.get("run", ""))
    assert summary["if"].startswith("always()")
    for name in ("harness", "agent", "app", "publish"):
        assert f"steps.{name}.outcome == 'failure'" in summary["if"]


def test_workflow_mints_only_after_agent_exits_and_never_gives_agent_gh_token() -> None:
    steps = yaml.safe_load(_WORKFLOW.read_text())["jobs"]["review"]["steps"]
    positions = {step.get("id"): index for index, step in enumerate(steps)}
    assert positions["agent"] < positions["app"] < positions["publish"]
    agent = steps[positions["agent"]]
    publish = steps[positions["publish"]]
    assert "GH_TOKEN" not in agent.get("env", {})
    assert agent["run"].endswith(" agent")
    assert publish["env"]["GH_TOKEN"] == "${{ steps.app.outputs.token }}"
    assert publish["run"].endswith(" publish")


def test_workflow_installs_the_harness_for_every_routed_prefix() -> None:
    job = yaml.safe_load(_WORKFLOW.read_text())["jobs"]["review"]
    install = next(step for step in job["steps"] if step.get("id") == "harness")["run"]
    assert "@openai/codex" in install
    assert "@anthropic-ai/claude-code" in install
    assert "@mariozechner/pi-coding-agent" in install
    # Routed, not all three: switching model must change what gets installed.
    for prefix in ("gpt-*)", "claude-*)", "grok-*)"):
        assert prefix in install
