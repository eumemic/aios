"""Running==merged reconciler fixture (aios#1327, plane B2).

Drives the production ``build_running_merged_reconciler_script`` directly against the
real script host (no DB, no sandbox, no model) with a deterministic capability
responder simulating each ``http_request`` / ``bash`` return — the dev_pipeline fixture
harness pattern. The reconciler is fully deterministic, so this verifies the full state
machine AND the load-bearing durable property: replay is deterministic (every call_key
is stable across every replay).

The verdict matrix under test:
  ok / drifted / cannot-determine (per app, NEVER aggregated), short/long SHA
  normalisation, the per-app-not-aggregate must-fix, the no-silent-degrade master read,
  idempotent routing, and the A-interim needs-deploy signal.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from aios.workflows.host_launcher import run_script_host
from aios.workflows.running_merged_reconciler import (
    DRIFT_ISSUE_TITLE,
    MARKER_DRIFT,
    build_running_merged_reconciler_script,
)

pytestmark = pytest.mark.integration

REPO = "eumemic/aios"
MASTER_SHA = "abc1234deadbeefabc1234deadbeefabc1234dea"  # 40-char SHA-1


class Scenario:
    """A deterministic responder + a record of what the script asked for. Keys every
    decision on the capability spec (and, for the per-app bash reads, the app uuid in
    the command) so the responder is a pure function of the call_key — replay-stable.
    """

    def __init__(
        self,
        *,
        master_sha: str | None = MASTER_SHA,
        master_status: int = 200,
        # {uuid: running_sha_or_sentinel} — what the ops-audit bash read returns per app.
        running: dict[str, str] | None = None,
        # Interim mode: the deploy-ledger last track-G SHA the bash read returns.
        ledger_sha: str | None = MASTER_SHA,
        # Whether an open drift issue with DRIFT_ISSUE_TITLE already exists (upsert path).
        existing_drift_issue: bool = False,
        # The comment thread of the existing drift issue (model an at-least-once replay
        # where the markered comment was already posted).
        drift_issue_comments: list[str] | None = None,
    ) -> None:
        self.master_sha = master_sha
        self.master_status = master_status
        self.running = running or {}
        self.ledger_sha = ledger_sha
        self.existing_drift_issue = existing_drift_issue
        self.drift_issue_comments = drift_issue_comments or []
        # Records for assertions.
        self.http: list[tuple[str, str]] = []
        self.issues_created: list[dict[str, Any]] = []
        self.comments_posted: list[str] = []
        self.bash_commands: list[str] = []

    DRIFT_ISSUE_NUMBER = 777

    def _http(self, args: dict[str, Any]) -> dict[str, Any]:
        path, method = args["path"], args["method"]
        self.http.append((method, path))
        clean = path.partition("?")[0]

        # master HEAD read
        if method == "GET" and clean == f"/repos/{REPO}/commits/master":
            if self.master_status != 200:
                return {"status": self.master_status, "body": "{}"}
            return {"status": 200, "body": json.dumps({"sha": self.master_sha})}

        # list open issues (idempotency-by-title) — paginated, single page
        if method == "GET" and clean == f"/repos/{REPO}/issues":
            if self.existing_drift_issue:
                body = json.dumps([{"number": self.DRIFT_ISSUE_NUMBER, "title": DRIFT_ISSUE_TITLE}])
            else:
                body = "[]"
            return {"status": 200, "headers": {}, "body": body}

        # the existing drift issue's comment thread (upsert dedup read)
        cm = re.match(rf"^/repos/{re.escape(REPO)}/issues/(\d+)/comments$", clean)
        if method == "GET" and cm:
            return {
                "status": 200,
                "headers": {},
                "body": json.dumps(
                    [{"id": i, "body": b} for i, b in enumerate(self.drift_issue_comments)]
                ),
            }
        if method == "POST" and cm:  # upsert comment on existing issue
            raw = args.get("body")
            if isinstance(raw, str):
                self.comments_posted.append(json.loads(raw).get("body", ""))
            return {"status": 201, "body": "{}"}

        # create the drift issue
        if method == "POST" and clean == f"/repos/{REPO}/issues":
            raw = args.get("body")
            if isinstance(raw, str):
                self.issues_created.append(json.loads(raw))
            return {"status": 201, "body": json.dumps({"number": self.DRIFT_ISSUE_NUMBER})}

        return {"status": 200, "body": "{}"}

    def _bash(self, command: str) -> dict[str, Any]:
        self.bash_commands.append(command)
        # The running-SHA read: identify the app uuid from the command.
        m = re.search(r"--app (\S+) --emit-build-sha", command)
        if m:
            uuid = m.group(1)
            sha = self.running.get(uuid, "__MISSING__")
            return {
                "exit_code": 0,
                "stdout": f"RUNNING_SHA={sha}\n",
                "stderr": "",
                "timed_out": False,
                "truncated": False,
            }
        # The interim ledger read.
        if "ledger-last-track-g-sha" in command:
            sha = self.ledger_sha if self.ledger_sha is not None else "__MISSING__"
            return {
                "exit_code": 0,
                "stdout": f"LEDGER_SHA={sha}\n",
                "stderr": "",
                "timed_out": False,
                "truncated": False,
            }
        return {"exit_code": 0, "stdout": "", "stderr": "", "timed_out": False, "truncated": False}

    def outcome(self, cap: Any) -> dict[str, Any]:
        cid, spec = cap.capability_id, cap.spec
        if cid == "tool":
            if spec["tool_name"] == "http_request":
                return {"ok": self._http(spec["input"])}
            if spec["tool_name"] == "bash":
                return {"ok": self._bash(spec["input"]["command"])}
        raise AssertionError(f"unhandled capability {cid} spec={spec!r}")


async def _drive(
    scenario: Scenario,
    *,
    input: dict[str, Any] | None = None,
    max_steps: int = 40,
    **build_kwargs: Any,
) -> tuple[Any, list[str], list[str]]:
    """Drive the reconciler to a terminal outcome, returning
    (return_value, phases, ordered_call_keys). Asserts replay-determinism (unique keys).
    """
    src = build_running_merged_reconciler_script(**build_kwargs)
    inp = input if input is not None else {"repo": REPO}
    memo: dict[str, Any] = {}
    keys: list[str] = []
    phases: list[str] = []
    for _ in range(max_steps):
        out = await run_script_host(source=src, input=inp, memo=memo)
        phases = [a.payload["text"] for a in out.annotations if a.payload["kind"] == "phase"]
        if out.kind == "returned":
            assert len(keys) == len(set(keys)), "replay produced a duplicate call_key"
            return out.value, phases, keys
        assert out.kind == "suspended", (out.kind, out.error_repr, out.error_traceback)
        assert len(out.emitted) == 1, [(e.capability_id, e.spec) for e in out.emitted]
        cap = out.emitted[0]
        keys.append(cap.call_key)
        memo[cap.call_key] = scenario.outcome(cap)
    raise AssertionError(f"reconciler did not terminate within {max_steps} steps")


def _verdict(value: dict[str, Any], app: str) -> str:
    for v in value["verdicts"]:
        if v["app"] == app:
            return v["verdict"]
    raise AssertionError(f"no verdict for {app}: {value['verdicts']!r}")


# ─── full mode: ok ────────────────────────────────────────────────────────────


async def test_ok_when_running_equals_master() -> None:
    scn = Scenario(running={"aios-api": MASTER_SHA, "aios-worker": MASTER_SHA})
    value, phases, _ = await _drive(scn)
    assert value["state"] == "reconciled"
    assert value["mode"] == "full"
    assert _verdict(value, "aios-api") == "ok"
    assert _verdict(value, "aios-worker") == "ok"
    assert value["clean"] is True
    # no issue filed, no comment posted, no owner-wake
    assert scn.issues_created == []
    assert scn.comments_posted == []
    assert value["routed_issue"] is None
    assert ("POST", f"/repos/{REPO}/issues") not in scn.http
    assert phases == ["read-master", "read-running", "route"]


# ─── full mode: drifted ─────────────────────────────────────────────────────


async def test_drifted_when_running_lags_master() -> None:
    other = "def5678" + "0" * 33  # a different 40-char SHA
    scn = Scenario(running={"aios-api": other, "aios-worker": other})
    value, _, _ = await _drive(scn)
    assert _verdict(value, "aios-api") == "drifted"
    assert _verdict(value, "aios-worker") == "drifted"
    assert value["clean"] is False
    # exactly one idempotent issue filed, carrying both SHAs
    assert len(scn.issues_created) == 1
    issue = scn.issues_created[0]
    assert issue["title"] == DRIFT_ISSUE_TITLE
    assert MASTER_SHA in issue["body"]
    assert other in issue["body"]
    assert MARKER_DRIFT in issue["body"]


# ─── full mode: cannot-determine (running unreadable) ────────────────────────


@pytest.mark.parametrize("sentinel", ["__SSH_FAIL__", "__MISSING__", "unknown", ""])
async def test_cannot_determine_on_unreadable_running_sha(sentinel: str) -> None:
    scn = Scenario(running={"aios-api": sentinel, "aios-worker": sentinel})
    value, _, _ = await _drive(scn)
    assert _verdict(value, "aios-api") == "cannot-determine"
    assert _verdict(value, "aios-worker") == "cannot-determine"
    # routed, not silently dropped
    assert value["clean"] is False
    assert len(scn.issues_created) == 1


# ─── cannot-determine (master unreadable) — the no-silent-degrade invariant ──


async def test_cannot_determine_on_unreadable_master() -> None:
    # master GET fails -> EVERY app is cannot-determine (master axis failed), never ok.
    scn = Scenario(
        master_status=500,
        running={"aios-api": MASTER_SHA, "aios-worker": MASTER_SHA},
    )
    value, _, _ = await _drive(scn)
    assert value["master_head"] is None
    assert _verdict(value, "aios-api") == "cannot-determine"
    assert _verdict(value, "aios-worker") == "cannot-determine"
    assert value["clean"] is False


# ─── the per-app-not-aggregate must-fix ──────────────────────────────────────


async def test_per_app_no_aggregate_ok() -> None:
    # aios-api reads ok, aios-worker reads __SSH_FAIL__ -> the worker is
    # cannot-determine and is NOT hidden behind the api's match.
    scn = Scenario(running={"aios-api": MASTER_SHA, "aios-worker": "__SSH_FAIL__"})
    value, _, _ = await _drive(scn)
    assert _verdict(value, "aios-api") == "ok"
    assert _verdict(value, "aios-worker") == "cannot-determine"
    # NOT an aggregate ok: the run is not clean, and it routes the worker's unreadability.
    assert value["clean"] is False
    assert len(scn.issues_created) == 1
    assert "aios-worker" in scn.issues_created[0]["body"]


# ─── short/long SHA normalisation ────────────────────────────────────────────


async def test_short_long_sha_normalization() -> None:
    short = MASTER_SHA[:7]  # running reports a 7-char short SHA
    scn = Scenario(running={"aios-api": short, "aios-worker": short})
    value, _, _ = await _drive(scn)
    assert _verdict(value, "aios-api") == "ok"  # prefix-normalised, no false drift
    assert _verdict(value, "aios-worker") == "ok"
    assert value["clean"] is True
    assert scn.issues_created == []


# ─── replay determinism ──────────────────────────────────────────────────────


async def test_replay_deterministic() -> None:
    other = "def5678" + "0" * 33
    scn = Scenario(running={"aios-api": other, "aios-worker": MASTER_SHA})
    _, _, keys = await _drive(scn)  # _drive already asserts unique keys
    assert len(keys) == len(set(keys))


# ─── idempotent routing (re-fire does not double-file) ───────────────────────


async def test_idempotent_routing_existing_issue_upserts_comment() -> None:
    other = "def5678" + "0" * 33
    scn = Scenario(
        running={"aios-api": other, "aios-worker": other},
        existing_drift_issue=True,
    )
    value, _, _ = await _drive(scn)
    # An open drift issue already exists -> upsert a markered comment, do NOT create a 2nd issue.
    assert scn.issues_created == []
    assert ("POST", f"/repos/{REPO}/issues") not in scn.http
    assert len(scn.comments_posted) == 1
    assert MARKER_DRIFT in scn.comments_posted[0]
    assert value["routed_issue"] == Scenario.DRIFT_ISSUE_NUMBER


async def test_idempotent_routing_skips_when_marker_already_present() -> None:
    other = "def5678" + "0" * 33
    scn = Scenario(
        running={"aios-api": other, "aios-worker": other},
        existing_drift_issue=True,
        drift_issue_comments=[MARKER_DRIFT + "\n\nalready posted by a prior fire"],
    )
    _, _, _ = await _drive(scn)
    # The maker-marker replay guard: the thread already carries the marker -> skip the POST.
    assert scn.comments_posted == []
    assert scn.issues_created == []


# ─── A-interim needs-deploy signal (ships FIRST, before A1/A2/A3) ────────────


async def test_interim_needs_deploy_signal() -> None:
    # interim mode: ledger-SHA != master-HEAD -> needs-deploy; running axis is
    # cannot-determine and self-flagged not-yet-substrate-different.
    other = "def5678" + "0" * 33
    scn = Scenario(ledger_sha=other)
    value, _phases, _ = await _drive(scn, interim=True)
    assert value["mode"] == "interim"
    assert _verdict(value, "aios-api") == "needs-deploy"
    assert _verdict(value, "aios-worker") == "needs-deploy"
    for v in value["verdicts"]:
        assert v["running"] is None  # running axis cannot-determine in interim
        assert "not-yet-substrate-different" in v["note"]
    assert value["clean"] is False
    assert len(scn.issues_created) == 1
    # the interim path never invokes the ops-audit per-app running read
    assert all("--emit-build-sha" not in c for c in scn.bash_commands)


async def test_interim_ok_when_ledger_equals_master() -> None:
    scn = Scenario(ledger_sha=MASTER_SHA)
    value, _, _ = await _drive(scn, interim=True)
    assert _verdict(value, "aios-api") == "ok"
    assert _verdict(value, "aios-worker") == "ok"
    assert value["clean"] is True
    assert scn.issues_created == []


async def test_interim_cannot_determine_when_ledger_unreadable() -> None:
    scn = Scenario(ledger_sha=None)  # bash read yields __MISSING__
    value, _, _ = await _drive(scn, interim=True)
    assert _verdict(value, "aios-api") == "cannot-determine"
    assert _verdict(value, "aios-worker") == "cannot-determine"
    assert value["clean"] is False


# ─── trigger envelope unwrapping (WorkflowAction fire) ───────────────────────


async def test_accepts_workflow_action_envelope() -> None:
    scn = Scenario(running={"aios-api": MASTER_SHA, "aios-worker": MASTER_SHA})
    value, _, _ = await _drive(scn, input={"trigger": {"kind": "cron"}, "input": {"repo": REPO}})
    assert value["state"] == "reconciled"
    assert value["clean"] is True
