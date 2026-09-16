"""Unit tests for the local coding-agent review launcher."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
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

_DIFF_EVIDENCE = (137, "a" * 64)


def _ok(stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], 0, stdout, "")


def _good_artifact(body: str = "A real finding.", lines: int = 137, digest: str = "a" * 64) -> str:
    return f"### Code review\n\n{body}\n\n<!-- inspected: lines={lines} sha256={digest} -->"


def _agent_returning(stdout: str, returncode: int = 0) -> Any:
    def run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], returncode, stdout, "")

    return run


@pytest.fixture
def passthrough_drop(monkeypatch: Any) -> None:
    """Run the harness argv as-is so tests can inspect command/env."""
    monkeypatch.setattr(
        reviewer, "_drop_into_agent_user", lambda command, env, temp: (command, env)
    )


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
    assert "sha256sum" in prompt
    assert reviewer.EVIDENCE_TEMPLATE in prompt
    # Expected evidence values must NOT be in the prompt, or a blocked agent
    # can parrot them back and the evidence proves nothing.
    assert "137" not in prompt


def test_run_agent_extracts_heading_from_stdout(
    monkeypatch: Any, clean_env: None, passthrough_drop: None
) -> None:
    monkeypatch.setenv("ANT_PROXY_API_KEY", "secret")
    artifact = "preamble\n" + _good_artifact("Finding.")
    completed = subprocess.CompletedProcess([], 0, artifact, "")
    monkeypatch.setattr(reviewer.subprocess, "run", lambda *args, **kwargs: completed)
    assert reviewer.run_agent("claude-opus-5", "prompt", 10, _DIFF_EVIDENCE) == _good_artifact(
        "Finding."
    )


def test_run_agent_unlinks_the_staged_key_and_does_not_hand_it_to_the_harness(
    monkeypatch: Any, clean_env: None, tmp_path: Path, passthrough_drop: None
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
        return subprocess.CompletedProcess(command, 0, _good_artifact("Finding."), "")

    monkeypatch.setattr(reviewer.subprocess, "run", run)
    assert reviewer.run_agent("claude-opus-5", "prompt", 10, _DIFF_EVIDENCE) == _good_artifact(
        "Finding."
    )
    env = seen["env"]
    assert "reusable-proxy-secret" not in env.values()
    assert env["ANTHROPIC_API_KEY"] != "reusable-proxy-secret"
    assert env["ANTHROPIC_BASE_URL"].startswith("http://127.0.0.1:")
    assert reviewer.ANT_PROXY_URL not in env.values()
    assert reviewer.PROXY_KEY_FILE_ENV not in env


def test_run_agent_takes_the_final_heading_not_an_echoed_one(
    monkeypatch: Any, clean_env: None, passthrough_drop: None
) -> None:
    """Pi and Claude stdout carries tool activity, which can quote the heading."""
    monkeypatch.setenv("XAI_PROXY_API_KEY", "secret")
    real = _good_artifact("The real finding.")
    noisy = (
        'grep "### Code review" scripts/eumemic_bot_review.py\n'
        "### Code review\nARTIFACT_HEADING = ...\n"
        f"{real}\n"
    )
    monkeypatch.setattr(
        reviewer.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 0, noisy, "")
    )
    assert reviewer.run_agent("grok-4.6", "prompt", 10, _DIFF_EVIDENCE) == real


def test_run_agent_prefers_codex_last_message(
    monkeypatch: Any, clean_env: None, passthrough_drop: None
) -> None:
    monkeypatch.setenv("OAI_PROXY_API_KEY", "secret")

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        path = Path(command[command.index("--output-last-message") + 1])
        path.write_text("chatty\n" + _good_artifact("Looks good."))
        return subprocess.CompletedProcess(command, 0, "event output", "")

    monkeypatch.setattr(reviewer.subprocess, "run", run)
    assert "Looks good." in reviewer.run_agent("gpt-5.6-sol", "prompt", 10, _DIFF_EVIDENCE)


def test_run_agent_reports_partial_output_on_timeout(
    monkeypatch: Any, clean_env: None, capsys: Any, passthrough_drop: None
) -> None:
    """A 15-minute timeout is the likeliest failure; its log must not be empty."""
    monkeypatch.setenv("ANT_PROXY_API_KEY", "secret")

    def run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired("claude", 10, output="got this far", stderr="warned")

    monkeypatch.setattr(reviewer.subprocess, "run", run)
    with pytest.raises(SystemExit):
        reviewer.run_agent("claude-opus-5", "prompt", 10, _DIFF_EVIDENCE)
    captured = capsys.readouterr()
    assert "got this far" in captured.out
    assert "warned" in captured.err


def test_missing_artifact_heading_is_fatal(
    monkeypatch: Any, clean_env: None, passthrough_drop: None
) -> None:
    monkeypatch.setenv("ANT_PROXY_API_KEY", "secret")
    monkeypatch.setattr(
        reviewer.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, "I refuse to follow format", ""),
    )
    with pytest.raises(SystemExit):
        reviewer.run_agent("claude-opus-5", "prompt", 10, _DIFF_EVIDENCE)


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

    The unprivileged-user boundary is established before seal and before the
    harness: prctl is extra, not the credential boundary.
    """
    _agent_env(monkeypatch, tmp_path)
    order: list[str] = []

    def scrub() -> None:
        order.append("scrub")

    def boundary() -> str:
        order.append("boundary")
        return "eumemic-review"

    def seal() -> None:
        order.append("seal")

    def agent(*args: Any, **kwargs: Any) -> str:
        order.append("agent")
        return _good_artifact("Pass.")

    monkeypatch.setattr(reviewer, "_drop_persisted_git_credentials", scrub)
    monkeypatch.setattr(reviewer, "_require_agent_user", boundary)
    monkeypatch.setattr(reviewer, "_seal_process", seal)
    monkeypatch.setattr(reviewer, "run_agent", agent)
    reviewer.run_agent_phase()
    assert order == ["scrub", "boundary", "seal", "agent"]


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
    monkeypatch.setattr(reviewer, "_require_agent_user", lambda: "eumemic-review")
    monkeypatch.setattr(reviewer, "diff_evidence", lambda *args: _DIFF_EVIDENCE)
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
        reviewer, "run_agent", lambda *args, **kwargs: pytest.fail("agent must not be launched")
    )
    with pytest.raises(SystemExit):
        reviewer.run_agent_phase()


def test_agent_phase_writes_artifact_after_agent_returns(monkeypatch: Any, tmp_path: Path) -> None:
    artifact = _agent_env(monkeypatch, tmp_path)

    def agent(*args: Any, **kwargs: Any) -> str:
        assert not artifact.exists()
        return _good_artifact("Pass.")

    monkeypatch.setattr(reviewer, "run_agent", agent)
    reviewer.run_agent_phase()
    assert artifact.read_text() == _good_artifact("Pass.") + "\n"


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


def test_drop_wraps_the_harness_with_setpriv_no_new_privs(monkeypatch: Any, tmp_path: Path) -> None:
    """The coding agent must not share a uid or sudo with the key holder."""
    monkeypatch.setattr(reviewer, "_require_agent_user", lambda: "eumemic-review")
    monkeypatch.setattr(
        reviewer, "_sudo", lambda args: subprocess.CompletedProcess(args, 0, "", "")
    )
    command, env = reviewer._drop_into_agent_user(
        ["claude", "--print"],
        {"ANTHROPIC_API_KEY": "loopback-token", "PATH": "/usr/bin"},
        tmp_path,
    )
    assert command[:3] == ["sudo", "-n", "--"]
    assert "setpriv" in command
    assert "--no-new-privs" in command
    assert "--reuid=eumemic-review" in command
    assert "--regid=eumemic-review" in command
    assert "loopback-token" not in command
    assert "loopback-token" not in env.values()
    spec = json.loads(Path(command[-1]).read_text())
    assert spec["argv"] == ["claude", "--print"]
    assert spec["env"]["ANTHROPIC_API_KEY"] == "loopback-token"
    assert spec["env"]["USER"] == "eumemic-review"
    gitconfig = Path(spec["env"]["GIT_CONFIG_GLOBAL"])
    assert gitconfig.exists()
    gitconfig_text = gitconfig.read_text()
    assert os.getcwd() in gitconfig_text
    assert "directory" in gitconfig_text
    assert spec["env"]["HOME"] == str(tmp_path)


def test_temp_root_gets_traverse_bit_before_drop(monkeypatch: Any, clean_env: None) -> None:
    """B1: mkdtemp is 0700; without 0711 the dropped uid cannot enter agent/."""
    monkeypatch.setenv("OAI_PROXY_API_KEY", "secret")
    seen_roots: list[Path] = []
    real_chmod = os.chmod

    def tracking_chmod(path: str | Path, mode: int, *args: Any, **kwargs: Any) -> None:
        p = Path(path)
        if mode == 0o711 and p.name.startswith("eumemic-review-"):
            seen_roots.append(p)
        real_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(os, "chmod", tracking_chmod)
    monkeypatch.setattr(reviewer, "_drop_into_agent_user", lambda c, e, t: (c, e))
    monkeypatch.setattr(reviewer.subprocess, "run", _agent_returning("### Code review\n\nLGTM.", 0))
    with pytest.raises(SystemExit) as exc:
        reviewer.run_agent("gpt-5.6-sol", "prompt", 10, _DIFF_EVIDENCE)
    assert exc.value.code == reviewer.NO_EVIDENCE_EXIT_CODE
    assert seen_roots, "launcher temp root must be chmod 0711 for agent traverse"


def test_drop_chowns_only_the_path_it_is_given(monkeypatch: Any, tmp_path: Path) -> None:
    """F2: the launcher TemporaryDirectory parent must stay launcher-owned."""
    monkeypatch.setattr(reviewer, "_require_agent_user", lambda: "eumemic-review")
    calls: list[list[str]] = []

    def sudo(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(reviewer, "_sudo", sudo)
    agent_home = tmp_path / "agent"
    agent_home.mkdir()
    reviewer._drop_into_agent_user(["claude"], {"K": "v"}, agent_home)
    assert ["chown", "-R", "eumemic-review", str(agent_home)] in calls
    assert not any(str(tmp_path) == arg for args in calls for arg in args[1:])


def test_rmtree_maybe_foreign_deletes_ours_without_sudo(monkeypatch: Any, tmp_path: Path) -> None:
    target = tmp_path / "agent"
    target.mkdir()
    (target / "x").write_text("y")
    monkeypatch.setattr(
        reviewer, "_sudo", lambda args: pytest.fail("sudo must not run for our tree")
    )
    reviewer._rmtree_maybe_foreign(target)
    assert not target.exists()


def test_rmtree_maybe_foreign_uses_sudo_when_not_ours(monkeypatch: Any, tmp_path: Path) -> None:
    target = tmp_path / "agent"
    target.mkdir()
    calls: list[list[str]] = []

    def sudo(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return _ok()

    monkeypatch.setattr(reviewer, "_tree_is_ours", lambda path: False)
    monkeypatch.setattr(reviewer, "_sudo", sudo)
    reviewer._rmtree_maybe_foreign(target)
    assert calls == [["rm", "-rf", str(target)]]


def test_rmtree_maybe_foreign_swallows_sudo_failure(monkeypatch: Any, tmp_path: Path) -> None:
    target = tmp_path / "agent"
    target.mkdir()
    monkeypatch.setattr(reviewer, "_tree_is_ours", lambda path: False)
    monkeypatch.setattr(
        reviewer,
        "_sudo",
        lambda args: subprocess.CompletedProcess(args, 1, "", "operation not permitted"),
    )
    reviewer._rmtree_maybe_foreign(target)


def test_no_evidence_still_cleans_agent_home_and_keeps_exit_3(
    monkeypatch: Any, clean_env: None
) -> None:
    """F2: cleanup must run on the refusal path and must not hide NO_EVIDENCE."""
    monkeypatch.setenv("OAI_PROXY_API_KEY", "secret")
    cleaned: list[Path] = []
    monkeypatch.setattr(reviewer, "_drop_into_agent_user", lambda c, e, t: (c, e))
    monkeypatch.setattr(reviewer, "_rmtree_maybe_foreign", lambda p: cleaned.append(p))
    monkeypatch.setattr(reviewer.subprocess, "run", _agent_returning("### Code review\n\nLGTM.", 0))
    with pytest.raises(SystemExit) as exc:
        reviewer.run_agent("gpt-5.6-sol", "prompt", 10, _DIFF_EVIDENCE)
    assert exc.value.code == reviewer.NO_EVIDENCE_EXIT_CODE
    assert cleaned and cleaned[0].name == "agent"


def test_dropped_gitconfig_marks_the_checkout_safe() -> None:
    cwd = os.getcwd()
    text = reviewer._agent_gitconfig_text(cwd)
    assert "[safe]" in text
    assert f"directory = {cwd}" in text


def test_ensure_dropped_uid_can_enter_opens_a_0700_home(tmp_path: Path) -> None:
    """Live FATAL: git cannot chdir into $GITHUB_WORKSPACE when $HOME is 0700."""
    home = tmp_path / "runner"
    checkout = home / "work" / "aios" / "aios"
    checkout.mkdir(parents=True)
    git_dir = checkout / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (checkout / "file.py").write_text("x = 1\n")
    secret = home / ".secret"
    secret.write_text("do-not-widen\n")
    os.chmod(secret, 0o600)
    os.chmod(home, 0o700)
    os.chmod(checkout, 0o700)
    os.chmod(git_dir, 0o700)
    os.chmod(git_dir / "HEAD", 0o600)
    os.chmod(checkout / "file.py", 0o600)
    reviewer._ensure_dropped_uid_can_enter(checkout)
    assert home.stat().st_mode & 0o777 == 0o711
    assert secret.stat().st_mode & 0o777 == 0o600
    assert checkout.stat().st_mode & 0o005 == 0o005
    assert git_dir.stat().st_mode & 0o005 == 0o005
    assert (git_dir / "HEAD").stat().st_mode & 0o004
    assert (checkout / "file.py").stat().st_mode & 0o004


def test_ensure_dropped_uid_can_enter_does_not_strip_existing_bits(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "repo"
    checkout.mkdir()
    os.chmod(tmp_path, 0o755)
    os.chmod(checkout, 0o755)
    reviewer._ensure_dropped_uid_can_enter(checkout)
    assert tmp_path.stat().st_mode & 0o777 == 0o755
    assert checkout.stat().st_mode & 0o777 == 0o755


def test_run_agent_opens_the_checkout_before_the_dropped_diff(
    monkeypatch: Any, clean_env: None
) -> None:
    monkeypatch.setenv("ANT_PROXY_API_KEY", "secret")
    order: list[str] = []

    def drop(
        command: list[str], env: dict[str, str], temp: Path
    ) -> tuple[list[str], dict[str, str]]:
        order.append("drop")
        return ["sudo", "-n", "--", "setpriv", "--no-new-privs", "--", *command], env

    def enter(path: Path) -> None:
        order.append("enter")
        assert Path(path) == Path(os.getcwd())

    def verify(*args: Any, **kwargs: Any) -> None:
        order.append("verify")

    monkeypatch.setattr(reviewer, "_drop_into_agent_user", drop)
    monkeypatch.setattr(reviewer, "_ensure_dropped_uid_can_enter", enter)
    monkeypatch.setattr(reviewer, "_verify_dropped_diff", verify)
    monkeypatch.setattr(reviewer.subprocess, "run", _agent_returning(_good_artifact(), 0))
    reviewer.run_agent(
        "claude-opus-5", "prompt", 10, _DIFF_EVIDENCE, base_sha="base", head_sha="head"
    )
    assert order == ["drop", "enter", "verify"]


def test_verify_dropped_diff_dies_on_checkout_permission_denied(
    monkeypatch: Any, tmp_path: Path
) -> None:
    (tmp_path / "gitconfig").write_text("[safe]\n")
    monkeypatch.setattr(
        reviewer.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            [],
            128,
            b"",
            b"fatal: cannot change to '/home/runner/work/aios/aios': Permission denied",
        ),
    )
    with pytest.raises(SystemExit) as exc:
        reviewer._verify_dropped_diff(tmp_path, "base", "head", _DIFF_EVIDENCE)
    assert exc.value.code == 1


def test_verify_dropped_diff_dies_on_dubious_ownership(monkeypatch: Any, tmp_path: Path) -> None:
    (tmp_path / "gitconfig").write_text("[safe]\n")
    monkeypatch.setattr(
        reviewer.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            [], 128, b"", b"fatal: detected dubious ownership"
        ),
    )
    with pytest.raises(SystemExit) as exc:
        reviewer._verify_dropped_diff(tmp_path, "base", "head", _DIFF_EVIDENCE)
    assert exc.value.code == 1


def test_verify_dropped_diff_accepts_a_matching_digest(monkeypatch: Any, tmp_path: Path) -> None:
    payload = b"diff --git a/x b/x\n+one\n"
    (tmp_path / "gitconfig").write_text("[safe]\n")
    monkeypatch.setattr(
        reviewer.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, payload, b""),
    )
    reviewer._verify_dropped_diff(
        tmp_path, "base", "head", (2, hashlib.sha256(payload).hexdigest())
    )


def test_require_agent_user_refuses_when_the_user_can_sudo(monkeypatch: Any) -> None:
    class Info:
        pw_uid = 12345

    monkeypatch.setattr(reviewer.sys, "platform", "linux")
    monkeypatch.setattr(reviewer, "_agent_user_may_sudo", lambda user: True)
    monkeypatch.setitem(sys.modules, "pwd", types.SimpleNamespace(getpwnam=lambda name: Info()))
    with pytest.raises(SystemExit):
        reviewer._require_agent_user()


def test_require_agent_user_refuses_a_same_uid(monkeypatch: Any) -> None:
    class Info:
        pw_uid = os.geteuid()

    monkeypatch.setattr(reviewer.sys, "platform", "linux")
    monkeypatch.setattr(reviewer, "_agent_user_may_sudo", lambda user: False)
    monkeypatch.setitem(sys.modules, "pwd", types.SimpleNamespace(getpwnam=lambda name: Info()))
    with pytest.raises(SystemExit):
        reviewer._require_agent_user()


def test_run_agent_actually_drops_before_exec(
    monkeypatch: Any, clean_env: None, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANT_PROXY_API_KEY", "secret")
    seen: dict[str, Any] = {}

    def drop(
        command: list[str], env: dict[str, str], temp: Path
    ) -> tuple[list[str], dict[str, str]]:
        seen["dropped"] = True
        seen["env"] = env
        assert "secret" not in env.values()
        return command, env

    monkeypatch.setattr(reviewer, "_drop_into_agent_user", drop)
    monkeypatch.setattr(reviewer.subprocess, "run", _agent_returning(_good_artifact(), 0))
    reviewer.run_agent("claude-opus-5", "prompt", 10, _DIFF_EVIDENCE)
    assert seen["dropped"] is True


def test_comments_do_not_claim_prctl_seals_github_runners() -> None:
    """Honesty: passwordless sudo on ubuntu-latest outranks PR_SET_DUMPABLE."""
    script = _SCRIPT.read_text()
    docs = (_ROOT / "docs/eumemic-bot-review.md").read_text()
    workflow = _WORKFLOW.read_text()
    blob = script + docs + workflow
    assert (
        "does NOT seal GitHub-hosted runners" in script
        or "does **not**\nseal GitHub-hosted runners" in script
        or "does **not** seal GitHub-hosted runners" in script
    )
    assert "passwordless sudo" in blob
    assert (
        "Broker+seal alone is insufficient" in script
        or "broker+seal alone is insufficient" in blob.lower()
    )
    # Must not claim dumpable is the property credential separation rests on.
    assert "credential separation rests on" not in script


def test_agent_exiting_zero_having_run_no_commands_must_not_write(
    monkeypatch: Any, clean_env: None, passthrough_drop: None
) -> None:
    """Zero-exit + heading only is not a review and must not become an artifact."""
    monkeypatch.setenv("OAI_PROXY_API_KEY", "secret")
    lgtm = "### Code review\n\nNo actionable findings in this range. LGTM."
    monkeypatch.setattr(reviewer.subprocess, "run", _agent_returning(lgtm, 0))
    with pytest.raises(SystemExit) as exc:
        reviewer.run_agent("gpt-5.6-sol", "prompt", 10, _DIFF_EVIDENCE)
    assert exc.value.code == reviewer.NO_EVIDENCE_EXIT_CODE


def test_no_evidence_is_a_distinct_loud_state_not_an_ordinary_failure(
    monkeypatch: Any, clean_env: None, capsys: Any, passthrough_drop: None
) -> None:
    monkeypatch.setenv("OAI_PROXY_API_KEY", "secret")
    monkeypatch.setattr(reviewer.subprocess, "run", _agent_returning("### Code review\n\nLGTM.", 0))
    with pytest.raises(SystemExit) as exc:
        reviewer.run_agent("gpt-5.6-sol", "prompt", 10, _DIFF_EVIDENCE)
    assert exc.value.code == reviewer.NO_EVIDENCE_EXIT_CODE
    assert exc.value.code != 1
    captured = capsys.readouterr()
    assert reviewer.NO_EVIDENCE_BANNER in captured.err
    assert "::error" in captured.out


def test_fabricated_inspection_evidence_must_not_write(
    monkeypatch: Any, clean_env: None, passthrough_drop: None
) -> None:
    monkeypatch.setenv("OAI_PROXY_API_KEY", "secret")
    monkeypatch.setattr(
        reviewer.subprocess, "run", _agent_returning(_good_artifact(lines=999, digest="b" * 64), 0)
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
def test_a_genuinely_inspected_review_is_written_whatever_its_verdict(
    monkeypatch: Any, clean_env: None, passthrough_drop: None, verdict: str
) -> None:
    monkeypatch.setenv("OAI_PROXY_API_KEY", "secret")
    artifact = _good_artifact(verdict)
    monkeypatch.setattr(reviewer.subprocess, "run", _agent_returning(artifact, 0))
    assert verdict in reviewer.run_agent("gpt-5.6-sol", "prompt", 10, _DIFF_EVIDENCE)


def test_a_matching_line_count_alone_is_NOT_evidence(
    monkeypatch: Any, clean_env: None, passthrough_drop: None
) -> None:
    monkeypatch.setenv("OAI_PROXY_API_KEY", "secret")
    artifact = "### Code review\n\nOK.\n\n<!-- inspected: lines=137 sha256=" + "c" * 64 + " -->"
    monkeypatch.setattr(reviewer.subprocess, "run", _agent_returning(artifact, 0))
    with pytest.raises(SystemExit) as exc:
        reviewer.run_agent("gpt-5.6-sol", "prompt", 10, _DIFF_EVIDENCE)
    assert exc.value.code == reviewer.NO_EVIDENCE_EXIT_CODE


def test_the_published_diff_line_count_paired_with_a_forged_digest_is_refused() -> None:
    public_line_count = 2243
    real = (2243, "085b04f85930cebf3d476a4905763f2c6d1ede92e9d2c1a060fe254d499dfd3b")
    artifact = (
        f"### Code review\n\nLGTM.\n\n"
        f"<!-- inspected: lines={public_line_count} sha256={'f' * 64} -->"
    )
    with pytest.raises(SystemExit) as exc:
        reviewer.require_inspection_evidence(artifact, real)
    assert exc.value.code == reviewer.NO_EVIDENCE_EXIT_CODE


def test_an_abbreviated_digest_is_refused() -> None:
    with pytest.raises(SystemExit) as exc:
        reviewer.require_inspection_evidence(
            "### Code review\n\n<!-- inspected: lines=137 sha256=aaaaaaaaaaaaaaaa -->",
            _DIFF_EVIDENCE,
        )
    assert exc.value.code == reviewer.NO_EVIDENCE_EXIT_CODE


def test_the_full_digest_is_what_writes(
    monkeypatch: Any, clean_env: None, passthrough_drop: None
) -> None:
    monkeypatch.setenv("OAI_PROXY_API_KEY", "secret")
    # Wrong line count, right digest: digest alone must suffice.
    artifact = (
        "### Code review\n\nReal finding.\n\n<!-- inspected: lines=999 sha256=" + "a" * 64 + " -->"
    )
    monkeypatch.setattr(reviewer.subprocess, "run", _agent_returning(artifact, 0))
    assert "Real finding." in reviewer.run_agent("gpt-5.6-sol", "prompt", 10, _DIFF_EVIDENCE)


def test_evidence_regex_will_not_even_match_a_short_digest() -> None:
    assert reviewer._EVIDENCE_RE.search("<!-- inspected: lines=1 sha256=" + "a" * 64 + " -->")
    for short in ("a" * 16, "a" * 63, "a" * 8):
        assert not reviewer._EVIDENCE_RE.search(f"<!-- inspected: lines=1 sha256={short} -->"), (
            f"regex matched a {len(short)}-char digest; the width is the guard"
        )


def test_agent_phase_does_not_write_artifact_without_inspection_evidence(
    monkeypatch: Any, tmp_path: Path
) -> None:
    artifact = _agent_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        reviewer, "run_agent", lambda *args, **kwargs: "### Code review\n\nLGTM, nothing to flag."
    )
    with pytest.raises(SystemExit) as exc:
        reviewer.run_agent_phase()
    assert exc.value.code == reviewer.NO_EVIDENCE_EXIT_CODE
    assert not artifact.exists()


def test_diff_evidence_is_computed_from_the_real_diff(monkeypatch: Any) -> None:
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
