#!/usr/bin/env python3
"""Run a proxy-backed coding agent and write its review artifact.

Used by .github/workflows/eumemic-bot-review.yml. The workflow checks out the PR
head and installs the harness for the routed model. Its ``agent`` phase writes
the review artifact without an installation token anywhere on its runner. A
separate workflow job publishes with inline ``gh api`` logic and never checks
out or executes this PR-head script. The legacy ``publish`` phase remains for
manual compatibility but is not part of the Action's trust path.

The routed proxy key is never handed to the harness. ``_ProxyBroker`` keeps it
in this process and the agent is given a random loopback-only token instead.

``prctl(PR_SET_DUMPABLE)`` is **not** the credential boundary and does **not**
seal GitHub-hosted runners: ubuntu-latest grants the ``runner`` user passwordless
sudo, which can read this process's memory regardless of the dumpable flag.
The boundary is a different OS user (``AGENT_USER``): the harness is exec'd
through ``setpriv --no-new-privs`` as that user, who cannot sudo, cannot ptrace
this process, and cannot read its ``/proc``. Broker+seal alone is insufficient.

Writing the ``### Code review`` artifact is gated on evidence of inspection,
not on the harness exit status. A zero exit with only the heading is refused.
The agent must echo the sha256 of ``git diff base...head``; the launcher
recomputes it. A mismatch or missing line is a distinct never-publishable
state (``NO_EVIDENCE_EXIT_CODE``).

Env:
  REVIEW_ARTIFACT_PATH, REPO, PR_NUMBER, HEAD_SHA
  BASE_SHA (agent phase only), GH_TOKEN (publish phase only)
  REVIEW_MODEL (default: DEFAULT_MODEL below) — routed by prefix to a harness
  REVIEW_TIMEOUT_SECONDS (default: _REVIEW_SECONDS below) — agent wall clock. It
    must run out before the job's timeout-minutes: this script's FATAL leaves the
    step's continue-on-error to keep the check green and still write the
    "did not post" job summary, whereas a runner kill takes both away.
  REVIEW_PROXY_KEY_FILE — file holding the routed family's proxy key, read and
    unlinked before the harness starts. This is how CI delivers it; the
    conventional env-var names still work for manual runs (see _proxy_key).
"""

from __future__ import annotations

import ctypes
import hashlib
import hmac
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import IO, Any, ClassVar, NoReturn
from urllib.parse import urlsplit

ARTIFACT_HEADING = "### Code review"
DEFAULT_MODEL = "gpt-5.6-sol"
_REVIEW_SECONDS = 900

# A verdict is only written when the agent proved it read the diff. The proof
# is a line the agent can only produce by hashing that diff: the launcher
# computes the same digest itself and compares. Nothing derivable from the
# prompt alone counts — the expected values are deliberately NOT in the prompt,
# only the recipe for deriving them.
EVIDENCE_TEMPLATE = "<!-- inspected: lines=<N> sha256=<HEX> -->"
# The FULL 64-hex digest, not a prefix. A prefix shortens the only
# high-entropy channel, and the width itself is load-bearing.
_EVIDENCE_RE = re.compile(
    r"<!--\s*inspected:\s*lines=(\d+)\s+sha256=([0-9a-fA-F]{64})\s*-->", re.IGNORECASE
)
# "The agent inspected nothing" is NOT an ordinary failure: a harness can exit
# 0 having never run a command. Distinct exit code, distinct banner, never
# written as a publishable artifact.
NO_EVIDENCE_EXIT_CODE = 3
NO_EVIDENCE_BANNER = "NO EVIDENCE OF INSPECTION — refusing to publish a verdict"

# OS user the coding-agent harness runs as. Must not be the key-holder user
# (typically `runner` on ubuntu-latest) and must not have passwordless sudo.
AGENT_USER = "eumemic-review"
# Tiny exec trampoline: sudo/setpriv cannot forward an arbitrary env dict
# without quoting holes, so the dropped process reads argv/env/cwd from a
# JSON spec (loopback token only — never the reusable proxy key).
_HARNESS_TRAMPOLINE = (
    "import json,os,sys;"
    "spec=json.load(open(sys.argv[1],encoding='utf-8'));"
    "os.chdir(spec['cwd']);"
    "os.execvpe(spec['argv'][0],spec['argv'],spec['env'])"
)

OAI_PROXY_URL = "https://oai-proxy.eumemic.ai/v1"
ANT_PROXY_URL = "https://ant-proxy.eumemic.ai"
XAI_PROXY_URL = "https://xai-proxy.eumemic.ai/v1"

# The agent reads PR-authored files (source, AGENTS.md, CLAUDE.md) and can run
# shell commands with a network, so it must not inherit anything reusable. The
# installation token in particular can comment and push as eumemic-bot; a proxy
# key spends money and outlives the job. The agent gets back exactly one
# credential, and it is a _ProxyBroker token that is worthless off this runner.
#
# This list is defence in depth, never the guarantee. unsetenv does not rewrite
# /proc/<pid>/environ, so a same-uid agent can read every variable this process
# was exec'd with. That is why the harness does not share a uid with this
# process (see _drop_into_agent_user): a different uid cannot read our /proc,
# and without sudo it cannot become us. The workflow still mints the App token
# in a different job and stages the proxy key through a file rather than the
# agent step's env, so even this process's /proc never holds GH_TOKEN. File
# credentials are handled by _drop_persisted_git_credentials and _proxy_key.
_STRIPPED_ENV = (
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "ACTIONS_RUNTIME_TOKEN",
    "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
    "REVIEW_PROXY_KEY_FILE",
    "OAI_PROXY_API_KEY",
    "ANT_PROXY_API_KEY",
    "XAI_PROXY_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "XAI_API_KEY",
)

PROXY_KEY_FILE_ENV = "REVIEW_PROXY_KEY_FILE"

# prctl(2). Clearing the dumpable flag reassigns this process's /proc entries to
# root and makes ptrace_may_access refuse every *unprivileged* tracer. It does
# not stop passwordless sudo. Do not treat this as sealing a GH runner.
_PR_SET_DUMPABLE = 4

_BROKER_CHUNK = 64 * 1024
# A socket timeout, not a run budget: it bounds the wait for the next byte from
# the proxy. The whole review is bounded by REVIEW_TIMEOUT_SECONDS.
_BROKER_UPSTREAM_TIMEOUT = 300

# Hop-by-hop headers, plus the ones the broker owns. Both credential headers go
# because whatever the harness presented is replaced with the real key, `host`
# and the framing headers because urllib recomputes them for the upstream hop,
# and `accept-encoding` so the response comes back as identity bytes we can
# relay without knowing how to decode them.
_REQUEST_HEADER_DROPS = frozenset(
    {
        "accept-encoding",
        "authorization",
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "x-api-key",
    }
)

# Content-Length is deliberately *kept*: a non-streaming JSON reply relays with
# its original framing. Transfer-Encoding goes because http.client has already
# de-chunked the body, which leaves SSE close-delimited.
_RESPONSE_HEADER_DROPS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """The broker holds the real key; it must not follow a Location off-origin."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> urllib.request.Request | None:
        return None


_UPSTREAM_OPENER = urllib.request.build_opener(_NoRedirect)

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


def _seal_process() -> None:
    """Defence in depth against an *unprivileged* same-uid reader of /proc.

    This is NOT the credential boundary and it does NOT seal GitHub-hosted
    runners. ubuntu-latest gives `runner` passwordless sudo; sudo can read this
    process's memory whether dumpable is set or not. Broker+seal alone is
    insufficient. The boundary is `_drop_into_agent_user`: the harness runs as
    a different uid that cannot sudo.

    Clearing dumpable still stops an unprivileged same-uid grandchild from
    reading /proc/<pid>, so a failure to set it is fatal — but prctl is not
    what keeps the reusable proxy key away from the coding agent.
    """
    if sys.platform != "linux":
        _die(f"cannot seal the launcher against /proc on {sys.platform}; run the agent on Linux")
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        sealed = libc.prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0)
    except OSError as exc:
        _die(f"cannot load libc to seal the launcher: {exc}")
    if sealed != 0:
        _die(f"prctl(PR_SET_DUMPABLE, 0) failed: {os.strerror(ctypes.get_errno())}")


def _proxy_key(primary: str, fallback: str) -> str:
    """Return the routed family's proxy key, preferring the staged file.

    CI stages the key into a 0600 file from a step that has exited by the time
    the agent runs, so no live process outside this sealed one holds it in its
    environment. The file is unlinked as soon as it is read, before the harness
    exists. The env-var names remain for manual runs, where the launcher's
    caller is the operator rather than something the agent can read off /proc.
    """
    staged = os.environ.get(PROXY_KEY_FILE_ENV, "").strip()
    if staged:
        path = Path(staged)
        try:
            value = path.read_text().strip()
        except OSError as exc:
            _die(f"cannot read {PROXY_KEY_FILE_ENV} {staged}: {exc}")
        path.unlink(missing_ok=True)
        if not value:
            _die(f"{PROXY_KEY_FILE_ENV} {staged} is empty")
        return value
    value = os.environ.get(primary, "").strip() or os.environ.get(fallback, "").strip()
    if not value:
        _die(f"{primary} (or {fallback}) is not set")
    return value


def _sudo(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["sudo", "-n", "--", *args], text=True, capture_output=True, check=False)


def _create_agent_user() -> None:
    result = _sudo(
        [
            "useradd",
            "--system",
            "--create-home",
            "--home-dir",
            f"/tmp/{AGENT_USER}",
            "--shell",
            "/usr/sbin/nologin",
            AGENT_USER,
        ]
    )
    if result.returncode:
        _die(
            f"cannot create unprivileged agent user {AGENT_USER} "
            f"(the coding agent must not share sudo with the key holder): "
            f"{(result.stderr or result.stdout).strip()[:300]}"
        )


def _agent_user_may_sudo(user: str) -> bool:
    """True if `user` can raise itself to the key-holder's domain."""
    nested = subprocess.run(
        ["sudo", "-n", "-u", user, "--", "sudo", "-n", "true"],
        capture_output=True,
        check=False,
    )
    if nested.returncode == 0:
        return True
    try:
        import grp
        import pwd

        info = pwd.getpwnam(user)
        privileged = {"sudo", "admin", "wheel"}
        for gid in os.getgrouplist(user, info.pw_gid):
            if grp.getgrgid(gid).gr_name in privileged:
                return True
    except (KeyError, OSError):
        # Cannot prove the user is unprivileged — fail closed.
        return True
    return False


def _require_agent_user() -> str:
    """Return the coding-agent OS user, creating it if needed.

    The credential boundary is this user, not prctl. ubuntu-latest gives the
    runner passwordless sudo, so a same-uid agent can read the key-holder's
    memory regardless of PR_SET_DUMPABLE. The harness therefore runs as a
    different uid with no sudo and with no-new-privs.
    """
    if sys.platform != "linux":
        _die(
            f"cannot separate the agent from the key holder on {sys.platform}; "
            "run the agent on Linux"
        )
    import pwd

    try:
        info = pwd.getpwnam(AGENT_USER)
    except KeyError:
        _create_agent_user()
        try:
            info = pwd.getpwnam(AGENT_USER)
        except KeyError:
            _die(f"created {AGENT_USER} but lookup still failed")
    if info.pw_uid == 0 or info.pw_uid in {os.geteuid(), os.getuid()}:
        _die(
            f"agent user {AGENT_USER} uid {info.pw_uid} shares this process's "
            f"privilege domain (euid {os.geteuid()}); refusing to start the harness"
        )
    if _agent_user_may_sudo(AGENT_USER):
        _die(
            f"agent user {AGENT_USER} can sudo; that is the same domain as the "
            "key holder on ubuntu-latest, so the reusable proxy key would be readable"
        )
    return AGENT_USER


def _drop_into_agent_user(
    command: list[str], env: dict[str, str], temp: Path
) -> tuple[list[str], dict[str, str]]:
    """Wrap `command` so it runs as AGENT_USER with no-new-privs.

    The reusable proxy key stays in *this* process. The child receives only
    `env`, which holds the loopback broker token. ``setpriv --no-new-privs`` is
    what stops the child from sudoing back into the key holder's domain;
    PR_SET_DUMPABLE does not.
    """
    user = _require_agent_user()
    dropped_env = {**env, "HOME": str(temp), "USER": user, "LOGNAME": user}
    spec_path = temp / "harness-spec.json"
    spec_path.write_text(
        json.dumps({"argv": command, "env": dropped_env, "cwd": os.getcwd()}),
        encoding="utf-8",
    )
    os.chmod(temp, 0o755)
    os.chmod(spec_path, 0o644)
    owned = _sudo(["chown", "-R", user, str(temp)])
    if owned.returncode:
        _die(f"cannot hand {temp} to {user}: {owned.stderr.strip()[:300]}")
    wrapped = [
        "sudo",
        "-n",
        "--",
        "setpriv",
        f"--reuid={user}",
        f"--regid={user}",
        "--clear-groups",
        "--no-new-privs",
        "--inh-caps=-all",
        "--",
        "/usr/bin/python3",
        "-c",
        _HARNESS_TRAMPOLINE,
        str(spec_path),
    ]
    # sudo/setpriv themselves must not see the loopback token or anything else
    # from the harness env; the trampoline reads the spec after the uid drop.
    return wrapped, {"PATH": "/usr/sbin:/usr/bin:/bin", "LANG": "C"}


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
            f"could not compute the diff for {base_sha}...{head_sha}: "
            f"{result.stderr.decode(errors='replace')[:300]}"
        )
    raw = result.stdout
    if not raw.strip():
        _die(f"`git diff {base_sha}...{head_sha}` is empty; there is nothing to review")
    return raw.count(b"\n"), hashlib.sha256(raw).hexdigest()


def _die_without_evidence(detail: str) -> NoReturn:
    """The loud, distinct, never-publishable state."""
    print(f"FATAL: {NO_EVIDENCE_BANNER}: {detail}", file=sys.stderr)
    print(f"::error title={NO_EVIDENCE_BANNER}::{detail}", file=sys.stdout)
    raise SystemExit(NO_EVIDENCE_EXIT_CODE)


def require_inspection_evidence(artifact: str, expected: tuple[int, str]) -> None:
    """Refuse to write an artifact unless it proves the agent read the diff.

    THE DIGEST IS THE ONLY ACCEPTING CHANNEL, and it must match in full.

    The line count is parsed and reported on a mismatch because it makes the
    diagnostic legible; it cannot authorise publication. It is public at the
    PR's `.diff` URL, so it never distinguished a real read from a network
    fetch. A zero-exit harness whose artifact is only `{ARTIFACT_HEADING}` is
    the state this gate exists to catch.
    """
    expected_lines, expected_digest = expected
    match = _EVIDENCE_RE.search(artifact)
    if match is None:
        _die_without_evidence(
            f"the agent's `{ARTIFACT_HEADING}` carries no well-formed "
            f"`{EVIDENCE_TEMPLATE}` line (the sha256 must be all 64 hex characters), "
            "so nothing shows it read the diff. Its verdict is not publishable."
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


class _ProxyBroker:
    """A loopback reverse proxy that owns the routed proxy key.

    The agent has a shell and unrestricted egress, so any credential in its
    environment is one it can read and send anywhere — and PR-authored files
    steer it. It cannot be given no credential at all, because it has to reach a
    model. The broker resolves that by making the credential the agent holds
    worth nothing off this runner: a random per-run token, accepted only on
    127.0.0.1, by a listener that dies with this process. The reusable key never
    enters the agent's environment; the agent does not share a uid with this
    process, so it also cannot read the key out of our memory. The key is
    stamped onto each request here, on the way out.
    """

    def __init__(self, upstream: str, header: str, key: str) -> None:
        parts = urlsplit(upstream)
        self._origin = f"{parts.scheme}://{parts.netloc}"
        # Proxy base URLs carry a path prefix (`/v1`) that the harness appends
        # to, so the broker keeps the prefix in the URL it advertises and
        # forwards whatever path arrives verbatim.
        self._base_path = parts.path.rstrip("/")
        self._header = header
        self._credential = f"Bearer {key}" if header == "authorization" else key
        self.token = secrets.token_urlsafe(32)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        handler = type("_BoundBrokerHandler", (_BrokerHandler,), {"broker": self})
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        serving = threading.Event()

        def serve() -> None:
            serving.set()
            assert self._server is not None
            self._server.serve_forever(poll_interval=0.05)

        self._thread = threading.Thread(target=serve, name="proxy-broker", daemon=True)
        self._thread.start()
        if not serving.wait(timeout=5):
            _die("proxy broker thread failed to start")

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("broker is not running")
        port = self._server.server_address[1]
        return f"http://127.0.0.1:{port}{self._base_path}"

    def authorize(self, headers: Message) -> bool:
        """Accept only this run's token, presented the way the harness sends it."""
        presented = headers.get("Authorization", "").strip()
        if presented.lower().startswith("bearer "):
            presented = presented[len("bearer ") :].strip()
        presented = presented or headers.get("x-api-key", "").strip()
        return bool(presented) and hmac.compare_digest(presented, self.token)

    def upstream_request(
        self, method: str, path: str, headers: Message, body: bytes | None
    ) -> urllib.request.Request:
        forwarded = {
            name: value
            for name, value in headers.items()
            if name.lower() not in _REQUEST_HEADER_DROPS
        }
        forwarded[self._header] = self._credential
        return urllib.request.Request(
            self._origin + path, data=body, headers=forwarded, method=method
        )


class _BrokerHandler(BaseHTTPRequestHandler):
    """Forward one harness request to the routed proxy, streaming the reply back."""

    broker: ClassVar[_ProxyBroker]
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        """Stay silent: request lines would put agent-chosen paths in the step log."""

    def do_GET(self) -> None:
        self._forward()

    def do_POST(self) -> None:
        self._forward()

    def do_DELETE(self) -> None:
        self._forward()

    def _forward(self) -> None:
        if not self.broker.authorize(self.headers):
            self._refuse(401, "broker token is missing or wrong")
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        request = self.broker.upstream_request(self.command, self.path, self.headers, body)
        try:
            with _UPSTREAM_OPENER.open(request, timeout=_BROKER_UPSTREAM_TIMEOUT) as response:
                self._relay(response.status, response.headers, response)
        except urllib.error.HTTPError as error:
            # A 4xx/5xx from the proxy is the harness's business, not ours: it
            # backs off on rate limits and reports auth failures from the real
            # status and body, which a broker-invented error would hide.
            with error:
                self._relay(error.code, error.headers, error)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._refuse(502, f"routed proxy request failed: {exc}")

    def _relay(self, status: int, headers: Message, stream: IO[bytes]) -> None:
        self.close_connection = True
        self.send_response(status)
        for name, value in headers.items():
            if name.lower() not in _RESPONSE_HEADER_DROPS:
                self.send_header(name, value)
        self.send_header("Connection", "close")
        self.end_headers()
        # read1, not read: both proxies answer with SSE, and read(n) on a
        # chunked response accumulates chunks until it has n bytes — which
        # would hold every token back until the completion finished.
        read = stream.read1 if hasattr(stream, "read1") else stream.read
        while chunk := read(_BROKER_CHUNK):
            self.wfile.write(chunk)
            self.wfile.flush()

    def _refuse(self, status: int, detail: str) -> None:
        body = json.dumps({"error": {"message": detail, "type": "broker_error"}}).encode()
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


def _broker_for(model: str) -> _ProxyBroker:
    """Build the routed family's broker, holding its proxy key."""
    kind = model_kind(model)
    if kind == "codex":
        key = _proxy_key("OAI_PROXY_API_KEY", "OPENAI_API_KEY")
        return _ProxyBroker(OAI_PROXY_URL, "authorization", key)
    if kind == "claude":
        key = _proxy_key("ANT_PROXY_API_KEY", "ANTHROPIC_API_KEY")
        return _ProxyBroker(ANT_PROXY_URL, "x-api-key", key)
    key = _proxy_key("XAI_PROXY_API_KEY", "XAI_API_KEY")
    return _ProxyBroker(XAI_PROXY_URL, "authorization", key)


def _agent_command(
    model: str, artifact_path: Path, broker: _ProxyBroker
) -> tuple[list[str], dict[str, str]]:
    """Build the harness command and its broker environment.

    Every route is pointed at the running broker and given its per-run token, so
    the credential the agent can read is loopback-only and dies with this
    process. The routed proxy key never appears here.
    """
    kind = model_kind(model)
    env = {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV}
    if kind == "codex":
        env["OPENAI_API_KEY"] = broker.token
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
                # Codex's own sandbox stays off: GitHub-hosted runners do not
                # permit its bubblewrap loopback setup, and a sandbox that
                # cannot start is a review that never posts. Isolation is the
                # unprivileged OS user (no sudo, no-new-privs) plus a broker
                # token for a listener that only exists while this launcher
                # does. The network the agent keeps buys it nothing it can
                # replay later; it cannot read the reusable key from this
                # process.
                "danger-full-access",
                "--ephemeral",
                "-c",
                f"model_provider={provider}",
                "-c",
                f'model_providers.{provider}={{name="eumemic oai-proxy",'
                f'base_url="{broker.base_url}",env_key="OPENAI_API_KEY",'
                f'wire_api="responses"}}',
                "--output-last-message",
                str(artifact_path),
                "-",
            ],
            env,
        )
    if kind == "claude":
        env.update(ANTHROPIC_API_KEY=broker.token, ANTHROPIC_BASE_URL=broker.base_url)
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

    config_dir = artifact_path.parent / "pi-config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "models.json").write_text(
        json.dumps(
            {
                "providers": {
                    "xai-proxy": {
                        "name": "xAI (eumemic proxy)",
                        "baseUrl": broker.base_url,
                        "api": "openai-responses",
                        "apiKey": broker.token,
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
    env.update(XAI_API_KEY=broker.token, PI_CODING_AGENT_DIR=str(config_dir))
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


def run_agent(model: str, prompt: str, timeout: int, evidence: tuple[int, str]) -> str:
    broker = _broker_for(model)
    broker.start()
    try:
        return _run_harness(model, prompt, timeout, broker, evidence)
    finally:
        # The token is only worth anything while this listener is up, so it goes
        # down on every path out, including the timeout FATAL.
        broker.close()


def _run_harness(
    model: str, prompt: str, timeout: int, broker: _ProxyBroker, evidence: tuple[int, str]
) -> str:
    with tempfile.TemporaryDirectory(prefix="eumemic-review-") as temp:
        artifact_path = Path(temp) / "last-message.md"
        command, env = _agent_command(model, artifact_path, broker)
        command, env = _drop_into_agent_user(command, env, Path(temp))
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
        # last-message.md is owned by AGENT_USER after a real drop; make it
        # readable before we pick it up. Skipped when the temp dir is still
        # ours (tests that passthrough the drop). Not a secret boundary.
        if os.stat(temp).st_uid != os.getuid():
            _sudo(["chmod", "-R", "a+rX", str(temp)])
        output = artifact_path.read_text() if artifact_path.exists() else result.stdout
        artifact = _artifact_in(output)
        if artifact is None:
            _die(f"{model} returned no `{ARTIFACT_HEADING}` artifact")
        # Exit 0 proves only that the harness process ended. A heading-only
        # artifact is not a review. The digest check is what separates a review
        # from a fluent guess, and it gates writing independently of exit status.
        require_inspection_evidence(artifact, evidence)
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
    # Privilege boundary before the key is read: the harness will not share
    # this uid. prctl is extra against same-uid /proc reads, not the boundary.
    _require_agent_user()
    _seal_process()
    evidence = diff_evidence(base_sha, head_sha)
    print(
        f"reviewing {repo}#{pr_number}@{head_sha} against {base_sha} with {model} "
        f"({model_kind(model)}); diff is {evidence[0]} lines, sha256 {evidence[1]}"
    )
    review = run_agent(model, _prompt(repo, pr_number, head_sha, base_sha), timeout, evidence)
    # Belt: run_agent already checked, but writing is the publishable act.
    require_inspection_evidence(review, evidence)
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
