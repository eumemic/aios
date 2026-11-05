"""The ``pipeline-provenance`` required check (aios#2572 provenance miss).

A pipeline PR (``dev-pipeline/*`` head ref OR ``pipeline:v2``) into master passes ONLY with BOTH
a ``reviewed:green`` that is valid for the LIVE head (reconciler rule, company#349) AND exactly one
``risk:tier-N`` with N in ``1..AUTO_MERGE_MAX_TIER``. Everything else passes immediately.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from scripts import pipeline_provenance as pp

_ROOT = Path(__file__).parents[2]
_WORKFLOW = _ROOT / ".github" / "workflows" / "pipeline-provenance.yml"
_RERUN = _ROOT / ".github" / "workflows" / "pipeline-provenance-rerun.yml"

HEAD = "a" * 40
OLD_HEAD = "b" * 40
IN_CAP = f"risk:tier-{pp.AUTO_MERGE_MAX_TIER}"


def _record(verdict: str, sha: str) -> str:
    """The reconciler's own canonical record (``_canonical_record_body``)."""
    lines = [
        "### Code review",
        "",
        "_Canonical review record (reconciler-written, company#383)_",
        "",
        f"**Verdict: {verdict}** at `{sha}`.",
        "",
    ]
    if verdict == "MERGE-SAFE":
        lines.append("No blocking findings at this head.")
    else:
        lines.append("1. **BLOCKING — F1** (src/x.py:1). broken. Property: holds")
    return "\n".join(lines)


def _comment(body: str, login: str = "eumemic", assoc: str = "OWNER") -> dict[str, Any]:
    return {"user": {"login": login}, "author_association": assoc, "body": body}


GREEN_AT_HEAD = _comment(_record("MERGE-SAFE", HEAD))


def _pr(
    labels: list[str],
    *,
    ref: str = "dev-pipeline/issue-2572",
    sha: str = HEAD,
    base: str = "master",
) -> dict[str, Any]:
    return {
        "base": {"ref": base},
        "head": {"ref": ref, "sha": sha},
        "labels": [{"name": n} for n in labels],
    }


def _passes(pr: dict[str, Any], comments: list[dict[str, Any]]) -> bool:
    return pp.evaluate(pr, comments).passed


# ─── the required matrix ─────────────────────────────────────────────────────


def test_unlabelled_pipeline_pr_fails() -> None:
    verdict = pp.evaluate(_pr(["pipeline:v2"]), [GREEN_AT_HEAD])
    assert not verdict.passed
    assert len(verdict.reasons) == 2  # neither review nor tier


def test_green_alone_fails() -> None:
    assert not _passes(_pr(["pipeline:v2", "reviewed:green"]), [GREEN_AT_HEAD])


def test_tier_alone_fails() -> None:
    assert not _passes(_pr(["pipeline:v2", "risk:tier-1"]), [GREEN_AT_HEAD])


def test_tier_above_cap_fails() -> None:
    over = f"risk:tier-{pp.AUTO_MERGE_MAX_TIER + 1}"
    assert not _passes(_pr(["pipeline:v2", "reviewed:green", over]), [GREEN_AT_HEAD])


@pytest.mark.parametrize("label", ["risk:tier-0", "risk:tier--1", "risk:tier-x", "risk:tier-"])
def test_out_of_range_or_non_integer_tier_fails(label: str) -> None:
    assert not _passes(_pr(["pipeline:v2", "reviewed:green", label]), [GREEN_AT_HEAD])


def test_two_tiers_fail() -> None:
    labels = ["pipeline:v2", "reviewed:green", "risk:tier-1", "risk:tier-2"]
    assert not _passes(_pr(labels), [GREEN_AT_HEAD])


def test_stale_head_record_fails() -> None:
    """The commit-after-review mutation: the flat label survives the push; the record does not."""
    stale = _comment(_record("MERGE-SAFE", OLD_HEAD))
    assert not _passes(_pr(["pipeline:v2", "reviewed:green", "risk:tier-1"]), [stale])


def test_stale_legacy_sha_stamp_fails() -> None:
    labels = ["pipeline:v2", f"reviewed:green@{OLD_HEAD[:12]}", "risk:tier-1"]
    assert not _passes(_pr(labels), [GREEN_AT_HEAD])


def test_legacy_sha_stamp_for_live_head_passes() -> None:
    """Pre-company#349 stamps keep their sha-scoped meaning (``_reviewed_label_for``)."""
    labels = ["pipeline:v2", f"reviewed:green@{HEAD[:12]}", "risk:tier-1"]
    assert _passes(_pr(labels), [])


def test_untrusted_author_record_fails() -> None:
    forged = _comment(_record("MERGE-SAFE", HEAD), login="outsider", assoc="CONTRIBUTOR")
    assert not _passes(_pr(["pipeline:v2", "reviewed:green", "risk:tier-1"]), [forged])


def test_record_without_author_fields_fails_closed() -> None:
    assert not _passes(
        _pr(["pipeline:v2", "reviewed:green", "risk:tier-1"]),
        [{"body": _record("MERGE-SAFE", HEAD)}],
    )


def test_missing_record_fails() -> None:
    assert not _passes(_pr(["pipeline:v2", "reviewed:green", "risk:tier-1"]), [])


def test_veto_at_head_fails_even_with_an_earlier_green() -> None:
    """Latest record per author wins, so a later veto at the same head revokes the green."""
    veto = _comment(_record("NEEDS-CHANGES", HEAD))
    assert not _passes(_pr(["pipeline:v2", "reviewed:green", "risk:tier-1"]), [GREEN_AT_HEAD, veto])


def test_quoted_green_is_not_authority() -> None:
    quoted = _comment("\n".join("> " + line for line in _record("MERGE-SAFE", HEAD).splitlines()))
    assert not _passes(_pr(["pipeline:v2", "reviewed:green", "risk:tier-1"]), [quoted])


@pytest.mark.parametrize("tier", range(1, pp.AUTO_MERGE_MAX_TIER + 1))
def test_both_in_cap_passes(tier: int) -> None:
    labels = ["pipeline:v2", "reviewed:green", f"risk:tier-{tier}"]
    assert _passes(_pr(labels), [GREEN_AT_HEAD])


def test_green_after_older_stale_record_passes() -> None:
    old = _comment(_record("NEEDS-CHANGES", OLD_HEAD))
    assert _passes(_pr(["pipeline:v2", "reviewed:green", IN_CAP]), [old, GREEN_AT_HEAD])


def test_non_pipeline_pr_passes() -> None:
    assert _passes(_pr([], ref="feat/some-human-change"), [])


def test_dev_pipeline_ref_without_label_is_still_a_pipeline_pr() -> None:
    assert not _passes(_pr([], ref="dev-pipeline/issue-1"), [])
    assert _passes(
        _pr(["reviewed:green", "risk:tier-1"], ref="dev-pipeline/issue-1"), [GREEN_AT_HEAD]
    )


def test_pipeline_label_on_a_human_ref_is_gated() -> None:
    assert not _passes(_pr(["pipeline:v2"], ref="feat/hand-held"), [])


def test_non_master_base_is_not_gated() -> None:
    assert _passes(_pr(["pipeline:v2"], base="release"), [])


# ─── override:chief (Chief ruling 2026-11-05) ────────────────────────────────


def _override(sha: str, who: str = "Chief") -> str:
    return f"override:chief @ `{sha}` ruling: {who} 2026-11-05 12:28 CT, merge as-is."


OVERRIDE_PR = _pr(["pipeline:v2", "override:chief"])  # no reviewed:green, no risk tier


@pytest.mark.parametrize(
    "sha",
    [HEAD, HEAD[: pp.OVERRIDE_MIN_SHA_LEN]],
    ids=["full", "sha12"],
)
@pytest.mark.parametrize("who", ["Chief", "Tom"])
def test_override_with_valid_comment_passes_without_stamp_or_tier(sha: str, who: str) -> None:
    verdict = pp.evaluate(OVERRIDE_PR, [_comment(_override(sha, who))])
    assert verdict.passed, verdict.reasons


def test_override_label_without_comment_fails() -> None:
    assert not _passes(OVERRIDE_PR, [])


def test_override_label_without_comment_fails_even_when_stamped_and_tiered() -> None:
    """The label switches the PR onto the override path: only a valid override passes it."""
    pr = _pr(["pipeline:v2", "override:chief", "reviewed:green", "risk:tier-1"])
    assert not _passes(pr, [GREEN_AT_HEAD])


@pytest.mark.parametrize(
    ("login", "assoc"),
    [
        ("eumemic-bot[bot]", "CONTRIBUTOR"),  # trusted, but the advisory CI reviewer
        ("some-member", "MEMBER"),  # trusted by association, not the AIOS Bot identity
        ("aspire488", "CONTRIBUTOR"),  # outside contributor
    ],
)
def test_override_comment_by_another_author_fails(login: str, assoc: str) -> None:
    assert not _passes(OVERRIDE_PR, [_comment(_override(HEAD), login=login, assoc=assoc)])


def test_override_comment_with_untrusted_association_fails() -> None:
    """The exact login must ALSO pass the trusted-author predicate (fail closed on odd payloads)."""
    comment = {"user": {"login": pp.OVERRIDE_AUTHOR_LOGIN}, "body": _override(HEAD)}
    assert pp._comment_trusted(comment)  # login-allowlisted: no association needed
    assert _passes(OVERRIDE_PR, [comment])
    assert not _passes(OVERRIDE_PR, [{"author_association": "OWNER", "body": _override(HEAD)}])


@pytest.mark.parametrize("sha", [OLD_HEAD, OLD_HEAD[:12]], ids=["full", "sha12"])
def test_override_for_an_old_head_fails(sha: str) -> None:
    verdict = pp.evaluate(OVERRIDE_PR, [_comment(_override(sha))])
    assert not verdict.passed
    assert "another head" in verdict.reasons[0]


@pytest.mark.parametrize(
    "body",
    [
        _override(HEAD[:11]),  # shorter than 12 hex
        _override(HEAD, who="Shepherd"),  # not a Chief/Tom ruling
        f"override:chief @ `{HEAD}`",  # no ruling cited
        "> " + _override(HEAD),  # quoted
        "    " + _override(HEAD),  # indented code
        "```\n" + _override(HEAD) + "\n```",  # fenced
        "per the " + _override(HEAD),  # not at line start
    ],
)
def test_override_comment_shape_is_strict(body: str) -> None:
    assert not _passes(OVERRIDE_PR, [_comment(body)])


def test_override_line_inside_a_longer_comment_counts() -> None:
    body = "Relaying the ruling.\n\n" + _override(HEAD) + "\n\nThread: chief#123"
    assert _passes(OVERRIDE_PR, [_comment(body)])


def test_override_comment_without_label_grants_nothing() -> None:
    assert not _passes(_pr(["pipeline:v2"]), [_comment(_override(HEAD))])


def test_override_does_not_touch_non_pipeline_prs() -> None:
    assert _passes(_pr(["override:chief"], ref="feature/x"), [])


# ─── the CLI contract the workflow relies on ─────────────────────────────────


def _run(tmp_path: Path, pr: dict[str, Any], comments: list[dict[str, Any]]) -> int:
    pr_path = tmp_path / "pr.json"
    pr_path.write_text(json.dumps(pr))
    comments_path = tmp_path / "comments.jsonl"
    comments_path.write_text("".join(json.dumps(c) + "\n" for c in comments))
    return pp.main(["--pr", str(pr_path), "--comments", str(comments_path)])


def test_cli_exit_codes(tmp_path: Path) -> None:
    good = _pr(["pipeline:v2", "reviewed:green", "risk:tier-1"])
    assert _run(tmp_path, good, [GREEN_AT_HEAD]) == pp.EXIT_PASS
    assert _run(tmp_path, good, []) == pp.EXIT_FAIL


def test_cli_unreadable_input_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "pr.json").write_text("not json")
    (tmp_path / "comments.jsonl").write_text("")
    argv = ["--pr", str(tmp_path / "pr.json"), "--comments", str(tmp_path / "comments.jsonl")]
    assert pp.main(argv) == pp.EXIT_UNREADABLE
    assert pp.main(["--pr", str(tmp_path / "missing"), "--comments", str(tmp_path)]) != 0


@pytest.mark.parametrize(
    "pr",
    [
        {"head": {"ref": "dev-pipeline/x", "sha": HEAD}, "labels": []},  # no base: NOT "ungated"
        {"base": {"ref": "master"}, "head": {"sha": HEAD}, "labels": []},  # no head ref
        {"base": {"ref": "master"}, "head": {"ref": "dev-pipeline/x", "sha": HEAD}},  # no labels
        {"base": {"ref": "master"}, "head": {"ref": "x", "sha": HEAD}, "labels": [{"id": 1}]},
    ],
)
def test_cli_truncated_pr_payload_fails_closed(tmp_path: Path, pr: dict[str, Any]) -> None:
    """A payload missing the fields that decide gating must not read as "not a pipeline PR"."""
    assert _run(tmp_path, pr, []) == pp.EXIT_UNREADABLE


# ─── the workflow shape a REQUIRED check needs ───────────────────────────────


def _workflow() -> dict[Any, Any]:
    loaded = yaml.safe_load(_WORKFLOW.read_text())
    assert isinstance(loaded, dict)
    return loaded


def test_check_name_and_job_id_are_pipeline_provenance() -> None:
    job = _workflow()["jobs"]["pipeline-provenance"]
    assert job["name"] == "pipeline-provenance"


def test_check_job_is_never_skipped_or_path_filtered() -> None:
    """A required check needs a conclusion on EVERY PR: no job ``if:``, no paths/branches filter."""
    wf = _workflow()
    on = wf[True]  # PyYAML reads the bare ``on`` key as boolean True
    assert "if" not in wf["jobs"]["pipeline-provenance"]
    trigger = on["pull_request_target"]
    assert set(trigger) == {"types"}
    assert {"opened", "reopened", "synchronize", "labeled", "unlabeled", "edited"} <= set(
        trigger["types"]
    )


def test_gate_runs_base_branch_code_with_read_only_token() -> None:
    """``pull_request_target`` runs the BASE workflow + script, so a PR cannot edit its own gate;
    the job never checks out the PR head and holds no write scope."""
    wf = _workflow()
    job = wf["jobs"]["pipeline-provenance"]
    assert set(job["permissions"].values()) == {"read"}
    for step in job["steps"]:
        assert "ref" not in (step.get("with") or {}), "the gate must not check out PR code"


def test_review_comments_re_run_the_check() -> None:
    """A stamp posted after the last run must still be able to turn the check green: every PR
    comment re-runs the check's latest run for the live head (which re-reads state live)."""
    wf = yaml.safe_load(_RERUN.read_text())
    assert set(wf[True]["issue_comment"]["types"]) == {"created", "edited", "deleted"}
    job = wf["jobs"]["rerun-on-comment"]
    assert job["permissions"] == {"actions": "write", "pull-requests": "read"}
    script = job["steps"][-1]["run"]
    assert "actions/workflows/pipeline-provenance.yml/runs" in script
    assert "event=pull_request_target" in script
    assert "/rerun" in script
    assert "github.event.comment" not in script, "comment text must never reach the shell"


# ─── the rerun step, executed against a fake `gh` ────────────────────────────

_FAKE_GH = """\
import json, os, sys
d = os.environ["FAKE_GH_DIR"]
args = sys.argv[1:]
with open(os.path.join(d, "calls.log"), "a") as f:
    f.write(json.dumps(args) + "\\n")
assert args[0] == "api", args
url = args[-1]
if "-X" in args:
    sys.exit(0)
if "/pulls/" in url:
    sys.stdout.write(open(os.path.join(d, "pr.json")).read())
    sys.exit(0)
assert "--paginate" in args, args
n_path = os.path.join(d, "n")
n = int(open(n_path).read()) if os.path.exists(n_path) else 0
open(n_path, "w").write(str(n + 1))
pages = json.load(open(os.path.join(d, "runs.json")))
sys.stdout.write("".join(json.dumps(p) for p in pages[min(n, len(pages) - 1)]))
"""


def _rerun_script() -> str:
    wf = yaml.safe_load(_RERUN.read_text())
    script = wf["jobs"]["rerun-on-comment"]["steps"][-1]["run"]
    assert isinstance(script, str)
    return script


def _wf_run(
    run_id: int, pr_numbers: list[int], created: str, status: str = "completed"
) -> dict[str, Any]:
    return {
        "id": run_id,
        "status": status,
        "created_at": created,
        "head_sha": HEAD,
        "pull_requests": [{"number": n} for n in pr_numbers],
    }


def _exec_rerun(
    tmp_path: Path,
    pr: dict[str, Any],
    responses: list[list[dict[str, Any]]],
    *,
    number: int = 2573,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    """Run the workflow step's bash verbatim. ``responses[i]`` is the list of pages returned by the
    i-th runs lookup (the last one repeats); ``gh`` and ``sleep`` are stubbed on PATH."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    gh = bindir / "gh"
    gh.write_text(f"#!{sys.executable}\n" + _FAKE_GH)
    gh.chmod(0o755)
    stub_sleep = bindir / "sleep"
    stub_sleep.write_text("#!/bin/sh\n")
    stub_sleep.chmod(0o755)
    (tmp_path / "pr.json").write_text(json.dumps(pr))
    (tmp_path / "runs.json").write_text(json.dumps(responses))
    env = {
        "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GH_DIR": str(tmp_path),
        "REPO": "eumemic/aios",
        "PR": str(number),
        "GH_TOKEN": "unused",
    }
    proc = subprocess.run(
        ["bash", "-c", _rerun_script()], env=env, capture_output=True, text=True, check=False
    )
    log = tmp_path / "calls.log"
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return proc, calls


def _open_pr(labels: list[str], ref: str = "dev-pipeline/issue-2572") -> dict[str, Any]:
    return {**_pr(labels, ref=ref), "state": "open"}


def _reruns(calls: list[list[str]]) -> list[str]:
    return [c[-1] for c in calls if "-X" in c]


def test_rerun_finds_the_prs_newest_run_by_pr_number(tmp_path: Path) -> None:
    """Newest run FOR THIS PR wins: a newer run on the same head for a different PR, and a fork
    run with empty ``pull_requests``, are not it."""
    page = {
        "workflow_runs": [
            _wf_run(30, [9999], "2026-11-05T18:30:00Z"),  # same head, other PR, newest
            _wf_run(20, [2573], "2026-11-05T18:20:00Z"),
            _wf_run(25, [], "2026-11-05T18:25:00Z"),  # fork-shaped
            _wf_run(10, [2573], "2026-11-05T18:10:00Z"),
        ]
    }
    proc, calls = _exec_rerun(tmp_path, _open_pr(["pipeline:v2"]), [[page]])
    assert proc.returncode == 0, proc.stderr
    assert _reruns(calls) == ["repos/eumemic/aios/actions/runs/20/rerun"]
    lookup = next(c[-1] for c in calls if "--paginate" in c)
    assert "workflows/pipeline-provenance.yml/runs?" in lookup
    assert "event=pull_request_target" in lookup
    assert f"head_sha={HEAD}" in lookup


def test_rerun_merges_paginated_pages(tmp_path: Path) -> None:
    pages = [
        {"workflow_runs": [_wf_run(10, [2573], "2026-11-05T18:10:00Z")]},
        {"workflow_runs": [_wf_run(11, [2573], "2026-11-05T18:11:00Z")]},
    ]
    proc, calls = _exec_rerun(tmp_path, _open_pr(["pipeline:v2"]), [pages])
    assert proc.returncode == 0, proc.stderr
    assert _reruns(calls) == ["repos/eumemic/aios/actions/runs/11/rerun"]


def test_rerun_waits_for_an_in_flight_run(tmp_path: Path) -> None:
    flying = {"workflow_runs": [_wf_run(20, [2573], "2026-11-05T18:20:00Z", "in_progress")]}
    done = {"workflow_runs": [_wf_run(20, [2573], "2026-11-05T18:20:00Z")]}
    proc, calls = _exec_rerun(tmp_path, _open_pr(["pipeline:v2"]), [[flying], [flying], [done]])
    assert proc.returncode == 0, proc.stderr
    assert sum("--paginate" in c for c in calls) == 3
    assert _reruns(calls) == ["repos/eumemic/aios/actions/runs/20/rerun"]


@pytest.mark.parametrize(
    "runs",
    [[], [_wf_run(25, [], "2026-11-05T18:25:00Z")], [_wf_run(30, [1], "2026-11-05T18:30:00Z")]],
    ids=["none", "fork-empty-pull_requests", "other-pr-only"],
)
def test_rerun_fails_loudly_when_no_run_matches_the_pr(
    tmp_path: Path, runs: list[dict[str, Any]]
) -> None:
    proc, calls = _exec_rerun(tmp_path, _open_pr(["pipeline:v2"]), [[{"workflow_runs": runs}]])
    assert proc.returncode == 1
    assert "no pipeline-provenance run found for PR #2573" in proc.stdout
    assert _reruns(calls) == []


@pytest.mark.parametrize(
    "pr",
    [
        _open_pr([], ref="feature/x"),
        _open_pr(["risk:tier-1", "reviewed:green"], ref="prov-aios"),
        {**_open_pr(["pipeline:v2"]), "state": "closed"},
    ],
    ids=["plain", "labelled-non-pipeline", "closed"],
)
def test_rerun_skips_non_pipeline_and_closed_prs(tmp_path: Path, pr: dict[str, Any]) -> None:
    proc, calls = _exec_rerun(tmp_path, pr, [[{"workflow_runs": []}]])
    assert proc.returncode == 0, proc.stderr
    assert [c for c in calls if "/pulls/" not in c[-1]] == []  # no runs lookup, no rerun


def test_rerun_treats_dev_pipeline_ref_without_label_as_pipeline(tmp_path: Path) -> None:
    page = {"workflow_runs": [_wf_run(20, [2573], "2026-11-05T18:20:00Z")]}
    proc, calls = _exec_rerun(tmp_path, _open_pr([]), [[page]])
    assert proc.returncode == 0, proc.stderr
    assert _reruns(calls) == ["repos/eumemic/aios/actions/runs/20/rerun"]


def test_rerun_fails_closed_on_an_unreadable_pr_payload(tmp_path: Path) -> None:
    """A payload missing head.ref/labels must error, never read as "not a pipeline PR"."""
    pr = {"state": "open", "head": {"sha": HEAD}}
    proc, calls = _exec_rerun(tmp_path, pr, [[{"workflow_runs": []}]])
    assert proc.returncode != 0
    assert _reruns(calls) == []
