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
_SPEC = importlib.util.spec_from_file_location("eumemic_bot_review", _SCRIPT)
assert _SPEC and _SPEC.loader
reviewer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reviewer)


@pytest.mark.parametrize(
    "model,kind", [("gpt-5.6-sol", "codex"), ("claude-opus-5", "claude"), ("grok-4.6", "pi")]
)
def test_model_routing(model: str, kind: str) -> None:
    assert reviewer.model_kind(model) == kind


def test_unknown_model_is_fatal() -> None:
    with pytest.raises(SystemExit):
        reviewer.model_kind("gemini-pro")


def test_codex_command_uses_responses_proxy(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setenv("OAI_PROXY_API_KEY", "secret")
    command, env = reviewer._agent_command("gpt-5.6-sol", tmp_path / "review.md")
    assert command[:4] == ["codex", "exec", "--model", "gpt-5.6-sol"]
    assert "--output-last-message" in command
    assert env["OPENAI_BASE_URL"] == "https://oai-proxy.eumemic.ai/v1"
    assert env["OPENAI_API_KEY"] == "secret"


def test_claude_command_uses_anthropic_proxy(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setenv("ANT_PROXY_API_KEY", "secret")
    command, env = reviewer._agent_command("claude-opus-5", tmp_path / "review.md")
    assert command[0] == "claude"
    assert "--print" in command
    assert env["ANTHROPIC_BASE_URL"] == "https://ant-proxy.eumemic.ai"


def test_pi_command_writes_xai_provider(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setenv("XAI_PROXY_API_KEY", "secret")
    command, env = reviewer._agent_command("grok-4.6", tmp_path / "review.md")
    assert command[0] == "pi"
    assert command[command.index("--provider") + 1] == "xai-proxy"
    config = Path(env["PI_CODING_AGENT_DIR"]).joinpath("models.json").read_text()
    assert "https://xai-proxy.eumemic.ai/v1" in config
    assert "grok-4.6" in config


def test_run_agent_extracts_heading_from_stdout(monkeypatch: Any) -> None:
    monkeypatch.setenv("ANT_PROXY_API_KEY", "secret")
    completed = subprocess.CompletedProcess([], 0, "preamble\n### Code review\n\nFinding.", "")
    monkeypatch.setattr(reviewer.subprocess, "run", lambda *args, **kwargs: completed)
    assert reviewer.run_agent("claude-opus-5", "prompt", 10) == "### Code review\n\nFinding."


def test_run_agent_prefers_codex_last_message(monkeypatch: Any) -> None:
    monkeypatch.setenv("OAI_PROXY_API_KEY", "secret")

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        path = Path(command[command.index("--output-last-message") + 1])
        path.write_text("chatty\n### Code review\n\nLooks good.")
        return subprocess.CompletedProcess(command, 0, "event output", "")

    monkeypatch.setattr(reviewer.subprocess, "run", run)
    assert reviewer.run_agent("gpt-5.6-sol", "prompt", 10).endswith("Looks good.")


def test_main_posts_and_verifies_marker(monkeypatch: Any, capsys: Any) -> None:
    for key, value in {
        "GH_TOKEN": "token",
        "REPO": "eumemic/aios",
        "PR_NUMBER": "1",
        "HEAD_SHA": "abc123",
        "REVIEW_MODEL": "gpt-5.6-sol",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(reviewer, "run_agent", lambda *args: "### Code review\n\nPass.")
    real_run = reviewer.subprocess.run
    monkeypatch.setattr(
        reviewer.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "abc123full\n", ""),
    )
    posted: dict[str, str] = {}

    def github(method: str, url: str, token: str, body: dict[str, str]) -> dict[str, str]:
        posted.update(body)
        return {"html_url": "https://github.test/comment/1", "body": body["body"]}

    monkeypatch.setattr(reviewer, "_github_request", github)
    reviewer.main()
    assert "<!-- eumemic-bot-review:abc123 -->" in posted["body"]
    assert "posted and verified" in capsys.readouterr().out
    monkeypatch.setattr(reviewer.subprocess, "run", real_run)


def test_workflow_is_local_head_pinned_and_has_no_aios_session_config() -> None:
    text = (_ROOT / ".github/workflows/eumemic-bot-review.yml").read_text()
    workflow = yaml.safe_load(text)
    job = workflow["jobs"]["review"]
    checkout = job["steps"][0]["with"]
    review = next(step for step in job["steps"] if step.get("id") == "review")
    assert checkout["ref"] == "${{ github.event.pull_request.head.sha }}"
    assert job["timeout-minutes"] <= 20
    assert review["env"]["REVIEW_TIMEOUT_SECONDS"] == "900"
    assert "AIOS_API_KEY" not in text
    assert "DEV_REVIEW_AGENT_ID" not in text
    assert "/v1/sessions" not in _SCRIPT.read_text()
    assert "gpt-5.6-sol" in workflow["env"]["REVIEW_MODEL"]
