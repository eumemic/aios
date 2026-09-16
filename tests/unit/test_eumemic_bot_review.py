"""Unit tests for the local coding-agent review launcher."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from email.message import EmailMessage
from pathlib import Path, PurePosixPath
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


@contextmanager
def _running_broker(
    upstream: str, header: str = "authorization", key: str = "upstream-secret"
) -> Iterator[Any]:
    broker = reviewer._ProxyBroker(upstream, header, key)
    broker.start()
    try:
        yield broker
    finally:
        broker.close()


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
    The URL is the loopback broker, not oai-proxy: the reusable key never
    enters the harness process.
    """
    with _running_broker(reviewer.OAI_PROXY_URL) as broker:
        command, env = reviewer._agent_command("gpt-5.6-sol", tmp_path / "review.md", broker)
        assert command[:4] == ["codex", "exec", "--model", "gpt-5.6-sol"]
        assert command[command.index("--sandbox") + 1] == "danger-full-access"
        assert command[command.index("--output-last-message") + 1] == str(tmp_path / "review.md")
        overrides = [command[i + 1] for i, arg in enumerate(command) if arg == "-c"]
        provider = next(o.split("=", 1)[1] for o in overrides if o.startswith("model_provider="))
        table = next(o for o in overrides if o.startswith(f"model_providers.{provider}="))
        assert f'base_url="{broker.base_url}"' in table
        assert broker.base_url.startswith("http://127.0.0.1:")
        assert reviewer.OAI_PROXY_URL not in table
        assert 'wire_api="responses"' in table
        assert 'env_key="OPENAI_API_KEY"' in table
        assert env["OPENAI_API_KEY"] == broker.token
        assert env["OPENAI_API_KEY"] != "upstream-secret"
        assert "OPENAI_BASE_URL" not in env


def test_claude_command_uses_anthropic_proxy(
    monkeypatch: Any, clean_env: None, tmp_path: Path
) -> None:
    with _running_broker(reviewer.ANT_PROXY_URL, header="x-api-key") as broker:
        command, env = reviewer._agent_command("claude-opus-5", tmp_path / "review.md", broker)
        assert command[0] == "claude"
        assert "--print" in command
        assert env["ANTHROPIC_BASE_URL"] == broker.base_url
        assert env["ANTHROPIC_API_KEY"] == broker.token
        assert env["ANTHROPIC_API_KEY"] != "upstream-secret"


def test_pi_command_writes_xai_provider(monkeypatch: Any, clean_env: None, tmp_path: Path) -> None:
    with _running_broker(reviewer.XAI_PROXY_URL) as broker:
        command, env = reviewer._agent_command("grok-4.6", tmp_path / "review.md", broker)
        assert command[0] == "pi"
        assert command[command.index("--provider") + 1] == "xai-proxy"
        config = Path(env["PI_CODING_AGENT_DIR"]).joinpath("models.json").read_text()
        assert broker.base_url in config
        assert "upstream-secret" not in config
        assert broker.token in config
        assert reviewer.XAI_PROXY_URL not in config
        assert "grok-4.6" in config


def test_missing_proxy_key_for_routed_family_is_fatal(monkeypatch: Any, clean_env: None) -> None:
    monkeypatch.setenv("ANT_PROXY_API_KEY", "wrong-family")
    with pytest.raises(SystemExit):
        reviewer._broker_for("gpt-5.6-sol")


def test_proxy_key_prefers_the_staged_file_and_unlinks_it(
    monkeypatch: Any, clean_env: None, tmp_path: Path
) -> None:
    staged = tmp_path / "eumemic-review-proxy-key"
    staged.write_text("from-file\n")
    monkeypatch.setenv(reviewer.PROXY_KEY_FILE_ENV, str(staged))
    monkeypatch.setenv("OAI_PROXY_API_KEY", "from-env")
    assert reviewer._proxy_key("OAI_PROXY_API_KEY", "OPENAI_API_KEY") == "from-file"
    assert not staged.exists()


def test_empty_staged_proxy_key_is_fatal(monkeypatch: Any, clean_env: None, tmp_path: Path) -> None:
    staged = tmp_path / "eumemic-review-proxy-key"
    staged.write_text("   \n")
    monkeypatch.setenv(reviewer.PROXY_KEY_FILE_ENV, str(staged))
    with pytest.raises(SystemExit):
        reviewer._proxy_key("OAI_PROXY_API_KEY", "OPENAI_API_KEY")
    assert not staged.exists()


def test_broker_rejects_a_missing_or_wrong_token() -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with _running_broker(reviewer.OAI_PROXY_URL) as broker:
        request = urllib.request.Request(
            broker.base_url + "/responses",
            data=b"{}",
            method="POST",
            headers={"Authorization": "Bearer not-the-token", "Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            opener.open(request, timeout=2)
        assert exc.value.code == 401


def test_broker_replaces_the_loopback_token_with_the_upstream_key(monkeypatch: Any) -> None:
    seen: dict[str, str] = {}

    class _FakeResponse:
        status = 200
        headers = EmailMessage()
        _body = b"ok"

        def __init__(self) -> None:
            self.headers["Content-Type"] = "text/plain"
            self.headers["Content-Length"] = "2"

        def read1(self, n: int) -> bytes:
            body, self._body = self._body, b""
            return body

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def open_upstream(
        request: urllib.request.Request, timeout: float | None = None
    ) -> _FakeResponse:
        seen["url"] = request.full_url
        seen["authorization"] = request.get_header("Authorization") or ""
        return _FakeResponse()

    monkeypatch.setattr(reviewer._UPSTREAM_OPENER, "open", open_upstream)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with _running_broker(reviewer.OAI_PROXY_URL, key="real-proxy-key") as broker:
        request = urllib.request.Request(
            broker.base_url + "/responses",
            data=b"{}",
            method="POST",
            headers={
                "Authorization": f"Bearer {broker.token}",
                "Content-Type": "application/json",
            },
        )
        with opener.open(request, timeout=2) as response:
            assert response.status == 200
            assert response.read() == b"ok"
        assert seen["url"] == "https://oai-proxy.eumemic.ai/v1/responses"
        assert seen["authorization"] == "Bearer real-proxy-key"
        assert broker.token not in seen["authorization"]


@pytest.mark.parametrize(
    "model,kept,upstream,header",
    [
        ("gpt-5.6-sol", "OPENAI_API_KEY", reviewer.OAI_PROXY_URL, "authorization"),
        ("claude-opus-5", "ANTHROPIC_API_KEY", reviewer.ANT_PROXY_URL, "x-api-key"),
    ],
)
def test_agent_env_drops_the_install_token_and_unrouted_keys(
    monkeypatch: Any,
    clean_env: None,
    tmp_path: Path,
    model: str,
    kept: str,
    upstream: str,
    header: str,
) -> None:
    """The agent runs PR-authored code; it must not inherit a writable token."""
    monkeypatch.setenv("GH_TOKEN", "ghs_installation")
    monkeypatch.setenv("GITHUB_TOKEN", "ghs_actions")
    monkeypatch.setenv("ACTIONS_RUNTIME_TOKEN", "runtime")
    monkeypatch.setenv("REVIEW_PROXY_KEY_FILE", "/tmp/eumemic-review-proxy-key")
    monkeypatch.setenv("OAI_PROXY_API_KEY", "oai")
    monkeypatch.setenv("ANT_PROXY_API_KEY", "ant")
    monkeypatch.setenv("XAI_PROXY_API_KEY", "xai")
    with _running_broker(upstream, header=header, key="oai-or-ant") as broker:
        _, env = reviewer._agent_command(model, tmp_path / "review.md", broker)
    assert "ghs_installation" not in env.values()
    assert not {"GH_TOKEN", "GITHUB_TOKEN", "ACTIONS_RUNTIME_TOKEN"} & set(env)
    assert not {
        "OAI_PROXY_API_KEY",
        "ANT_PROXY_API_KEY",
        "XAI_PROXY_API_KEY",
        "REVIEW_PROXY_KEY_FILE",
    } & set(env)
    assert env[kept] == broker.token
    assert "oai-or-ant" not in env.values()


def test_prompt_pins_the_reviewed_range_to_base_and_head() -> None:
    prompt = reviewer._prompt("eumemic/aios", "7", "headsha", "basesha")
    assert "git diff basesha...headsha" in prompt
    assert reviewer.ARTIFACT_HEADING in prompt


def test_run_agent_extracts_heading_from_stdout(monkeypatch: Any, clean_env: None) -> None:
    monkeypatch.setenv("ANT_PROXY_API_KEY", "secret")
    completed = subprocess.CompletedProcess([], 0, "preamble\n### Code review\n\nFinding.", "")
    monkeypatch.setattr(reviewer.subprocess, "run", lambda *args, **kwargs: completed)
    assert reviewer.run_agent("claude-opus-5", "prompt", 10) == "### Code review\n\nFinding."


def test_run_agent_unlinks_the_staged_key_and_does_not_hand_it_to_the_harness(
    monkeypatch: Any, clean_env: None, tmp_path: Path
) -> None:
    """The High: danger-full-access plus a reusable proxy secret is the leak.

    The harness may keep a network, but the only credential it can read is the
    loopback broker token, and the staged file is gone before exec.
    """
    staged = tmp_path / "eumemic-review-proxy-key"
    staged.write_text("reusable-proxy-secret")
    monkeypatch.setenv(reviewer.PROXY_KEY_FILE_ENV, str(staged))
    seen: dict[str, Any] = {}

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen["command"] = command
        seen["env"] = kwargs["env"]
        assert not staged.exists()
        return subprocess.CompletedProcess(command, 0, "### Code review\n\nFinding.", "")

    monkeypatch.setattr(reviewer.subprocess, "run", run)
    assert reviewer.run_agent("claude-opus-5", "prompt", 10) == "### Code review\n\nFinding."
    env = seen["env"]
    assert "reusable-proxy-secret" not in env.values()
    assert env["ANTHROPIC_API_KEY"] != "reusable-proxy-secret"
    assert env["ANTHROPIC_BASE_URL"].startswith("http://127.0.0.1:")
    assert reviewer.ANT_PROXY_URL not in env.values()
    assert reviewer.PROXY_KEY_FILE_ENV not in env


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
    """Ordering is the point: _pin_checkout may fetch, the agent must not be able to.

    Seal sits between scrub and the harness: the launcher must be undumpable
    before it reads the proxy key that run_agent holds for the broker.
    """
    _agent_env(monkeypatch, tmp_path)
    order: list[str] = []

    def scrub() -> None:
        order.append("scrub")

    def seal() -> None:
        order.append("seal")

    def agent(*args: Any) -> str:
        order.append("agent")
        return "### Code review\n\nPass."

    monkeypatch.setattr(reviewer, "_drop_persisted_git_credentials", scrub)
    monkeypatch.setattr(reviewer, "_seal_process", seal)
    monkeypatch.setattr(reviewer, "run_agent", agent)
    reviewer.run_agent_phase()
    assert order == ["scrub", "seal", "agent"]


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
    monkeypatch.setattr(reviewer, "_seal_process", lambda: None)
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
    job = workflow["jobs"]["agent"]
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
    jobs = yaml.safe_load(_WORKFLOW.read_text())["jobs"]
    agent_steps = {step["id"]: step for step in jobs["agent"]["steps"] if "id" in step}
    publish_steps = {step["id"]: step for step in jobs["publish"]["steps"] if "id" in step}
    assert all(
        agent_steps[name]["continue-on-error"]
        for name in ("harness", "proxykey", "agent", "artifact")
    )
    assert all(publish_steps[name]["continue-on-error"] for name in ("artifact", "app", "publish"))
    summary = next(
        step for step in jobs["publish"]["steps"] if "GITHUB_STEP_SUMMARY" in step.get("run", "")
    )
    assert summary["if"].startswith("always()")
    assert "needs.agent.result != 'success'" in summary["if"]
    for name in ("artifact", "app", "publish"):
        assert f"steps.{name}.outcome != 'success'" in summary["if"]
    # The summary only means something if it is rare. `always()` on the job
    # would also fire it for every run superseded by cancel-in-progress, which
    # published nothing because it was replaced — not because anything missed.
    # `!cancelled()` still runs the job when the agent job outright failed.
    assert jobs["publish"]["if"].startswith("${{ !cancelled() &&")


def test_workflow_mints_only_after_agent_exits_and_never_gives_agent_gh_token() -> None:
    jobs = yaml.safe_load(_WORKFLOW.read_text())["jobs"]
    agent_steps = jobs["agent"]["steps"]
    publish_steps = jobs["publish"]["steps"]
    agent = next(step for step in agent_steps if step.get("id") == "agent")
    publish = next(step for step in publish_steps if step.get("id") == "publish")
    assert "GH_TOKEN" not in agent.get("env", {})
    assert agent["run"].endswith(" agent")
    assert jobs["publish"]["needs"] == "agent"
    assert publish["env"]["GH_TOKEN"] == "${{ steps.app.outputs.token }}"
    assert all("GH_TOKEN" not in step.get("env", {}) for step in agent_steps)


def test_publish_is_a_fresh_job_that_executes_no_repository_code() -> None:
    text = _WORKFLOW.read_text()
    jobs = yaml.safe_load(text)["jobs"]
    publish = jobs["publish"]
    steps = publish["steps"]
    assert publish["needs"] == "agent"
    assert not any(step.get("uses", "").startswith("actions/checkout") for step in steps)
    run = next(step for step in steps if step.get("id") == "publish")["run"]
    assert "gh api --method POST" in run
    assert '(.user.login == "eumemic-bot[bot]")' in run
    assert "git show" not in text
    assert "mktemp" not in text
    assert "eumemic_bot_review.py" not in run


def test_jobs_transfer_only_the_markdown_review_artifact() -> None:
    jobs = yaml.safe_load(_WORKFLOW.read_text())["jobs"]
    upload = next(step for step in jobs["agent"]["steps"] if step.get("id") == "artifact")
    download = next(step for step in jobs["publish"]["steps"] if step.get("id") == "artifact")
    assert upload["uses"] == "actions/upload-artifact@v4"
    assert upload["with"]["path"] == ".eumemic-bot-review.md"
    assert download["uses"] == "actions/download-artifact@v4"
    assert download["with"]["name"] == upload["with"]["name"]


def test_upload_opts_into_the_hidden_artifact_filename() -> None:
    """upload-artifact drops dot-prefixed paths unless told otherwise.

    Since v4.4 the action skips "any file beginning with `.`", and the launcher
    writes `.eumemic-bot-review.md`. The pattern then matches nothing,
    `if-no-files-found: error` fails the upload, and the publish job has no
    artifact to download — the review silently never posts, which is the exact
    failure this workflow keeps producing. Asserted conditionally so a rename
    to a non-hidden name stays valid without the input.
    """
    jobs = yaml.safe_load(_WORKFLOW.read_text())["jobs"]
    agent_env = next(step for step in jobs["agent"]["steps"] if step.get("id") == "agent")["env"]
    upload = next(step for step in jobs["agent"]["steps"] if step.get("id") == "artifact")["with"]
    download = next(step for step in jobs["publish"]["steps"] if step.get("id") == "artifact")[
        "with"
    ]
    publish_env = next(step for step in jobs["publish"]["steps"] if step.get("id") == "publish")[
        "env"
    ]
    # What the launcher writes is what gets uploaded, and what the publisher
    # reads is where the download lands. Both halves are silent when wrong.
    assert agent_env["REVIEW_ARTIFACT_PATH"].endswith(f"/{upload['path']}")
    assert publish_env["REVIEW_ARTIFACT_PATH"].endswith(
        f"/{download['path']}/{PurePosixPath(upload['path']).name}"
    )
    if PurePosixPath(upload["path"]).name.startswith("."):
        assert upload["include-hidden-files"] is True


def test_workflow_does_not_give_the_agent_step_a_reusable_proxy_secret() -> None:
    """A secret the agent step receives is readable from /proc for the whole run.

    The staging step is a different process that has exited before the agent
    starts. The launcher then reads REVIEW_PROXY_KEY_FILE, unlinks it, and
    hands the harness a loopback broker token. The routed family's secret must
    not also appear on the agent step, or the broker is bypassed.
    """
    steps = yaml.safe_load(_WORKFLOW.read_text())["jobs"]["agent"]["steps"]
    install = next(step for step in steps if step.get("id") == "harness")["run"]
    staged = next(step for step in steps if step.get("id") == "proxykey")
    agent = next(step for step in steps if step.get("id") == "agent")
    agent_env = agent["env"]
    drop = next(
        step
        for step in steps
        if "eumemic-review-proxy-key" in step.get("run", "") and step.get("id") != "proxykey"
    )
    assert agent_env["REVIEW_PROXY_KEY_FILE"].endswith("/eumemic-review-proxy-key")
    assert agent["if"] == "steps.proxykey.outcome == 'success'"
    assert staged["if"] == "steps.harness.outcome == 'success'"
    for name in ("OAI_PROXY_API_KEY", "ANT_PROXY_API_KEY", "XAI_PROXY_API_KEY"):
        assert name not in agent_env
        assert f"secrets.{name}" not in json.dumps(agent)
    for family, name in (("oai", "OAI"), ("ant", "ANT"), ("xai", "XAI")):
        assert f"family='{family}'" in install
        value = staged["env"][f"{name}_PROXY_API_KEY"]
        assert f"steps.harness.outputs.family == '{family}'" in value
        assert f"secrets.{name}_PROXY_API_KEY" in value
        assert f"${name}_PROXY_API_KEY" in staged["run"]
    assert drop["if"] == "always()"
    assert "rm -f" in drop["run"]


def test_workflow_installs_the_harness_for_every_routed_prefix() -> None:
    job = yaml.safe_load(_WORKFLOW.read_text())["jobs"]["agent"]
    install = next(step for step in job["steps"] if step.get("id") == "harness")["run"]
    assert "@openai/codex" in install
    assert "@anthropic-ai/claude-code" in install
    assert "@mariozechner/pi-coding-agent" in install
    # Routed, not all three: switching model must change what gets installed.
    for prefix in ("gpt-*)", "claude-*)", "grok-*)"):
        assert prefix in install
