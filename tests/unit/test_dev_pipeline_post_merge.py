"""The dev-pipeline POST-MERGE checker — the post-merge twin of ``owner()``'s pre-merge gate
(task #76; aios#49/#111, the §5 residue of the reconciler design-of-record).

The post-merge checker is a SEPARATE cron-fired workflow that verifies, from DURABLE off-the-run
GitHub state (an uncorrelated substrate), that every recently-merged ``pipeline:v2`` PR (a) closed
its source issue, (b) did not leave master red, and (c) merged THROUGH the reconciler's maker≠checker
gate — escalating any violation via a ``needs:human/*`` label, never silently.

These unit tests pin the STATIC properties: the assembled script bans ``gate()`` + ``datetime``,
threads the reconciler's tier-cap, parses, defaults the run-provenance posture, and declares the
correct read+escalate-only deploy surface (no PUT/PATCH/DELETE — the structural guarantee that the
checker can NEVER mutate the merge it judges). The behavioural three-check machine is exercised
end-to-end against the host in ``tests/integration/test_wf_dev_pipeline_post_merge_fixture.py``.
"""

from __future__ import annotations

import ast

from aios.models.agents import HttpServerSpec
from aios.models.workflows import WorkflowCreate
from aios.workflows.dev_pipeline_post_merge import (
    DEFAULT_ESCALATED_LABEL,
    LABEL_CHECKED_PREFIX,
    NEEDS_HUMAN_BAD_PROVENANCE,
    NEEDS_HUMAN_ISSUE_OPEN,
    NEEDS_HUMAN_MASTER_RED,
    REQUIRED_HTTP_SERVERS,
    REQUIRED_TOOLS,
    build_post_merge_checker_fixture_script,
    build_post_merge_checker_script,
    build_post_merge_checker_workflow_create,
)
from aios.workflows.dev_pipeline_reconciler import (
    AUTO_MERGE_MAX_TIER,
    LABEL_PIPELINE_V2,
    LABEL_REVIEWED_PREFIX,
    LABEL_RISK_PREFIX,
)

# ════════════════════════════════════════════════════════════════════════════
# SAFETY — gate() is BANNED in the assembled script; no datetime/time import.
# ════════════════════════════════════════════════════════════════════════════


def _gate_call_count(src: str) -> int:
    """Count actual ``gate(...)`` CALL nodes (bare-name AND attribute form), so a smuggled
    suspend can't slip past the ban — mirrors the reconciler's guard."""
    tree = ast.parse(src)
    count = 0
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        func = n.func
        if (isinstance(func, ast.Name) and func.id == "gate") or (
            isinstance(func, ast.Attribute) and func.attr == "gate"
        ):
            count += 1
    return count


def test_assembled_script_has_zero_gate_calls() -> None:
    # The checker is a stateless sweep — gate() would suspend the whole run (an orphan), and the
    # checker MUST stay a re-derivable backstop. Escalation is a durable needs:human/* label.
    for src in (
        build_post_merge_checker_script(repo="o/r"),
        build_post_merge_checker_fixture_script(repo="o/r"),
    ):
        assert _gate_call_count(src) == 0
        tree = ast.parse(src)
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                assert all(a.name != "gate" for a in n.names), "gate must never be imported/aliased"
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Name):
                assert n.value.id != "gate", "gate must never be rebound to an alias"


def test_assembled_script_excludes_datetime_import() -> None:
    # No datetime/time — the checker reads recency off GitHub's merged_at ordering, never a clock.
    src = build_post_merge_checker_script(repo="o/r")
    assert "import datetime" not in src
    assert "import time" not in src


def test_assembled_script_parses() -> None:
    ast.parse(build_post_merge_checker_script(repo="o/r"))


def test_post_merge_body_emits_no_agent_or_gate_call() -> None:
    # The checker's verdict is MECHANICAL + off-the-run: it calls only tool() (http_request +
    # list_runs), never agent() and never gate(). The spliced DEV_PIPELINE_LIB *defines* helpers
    # that call agent() (e.g. _watch_ci), but the checker body never INVOKES them — so the body
    # itself (everything before the DEV_PIPELINE_LIB splice) contains no agent()/gate() call, and
    # the deploy surface carries no "agent"/"gate" tool.
    src = build_post_merge_checker_script(repo="o/r")
    body_only = src.split("# ─── scripted helpers")[0]  # the _BODY, before DEV_PIPELINE_LIB
    assert "await agent(" not in body_only
    assert "gate(" not in body_only
    assert {t.type for t in REQUIRED_TOOLS} == {"http_request", "list_runs"}


# ════════════════════════════════════════════════════════════════════════════
# PROVENANCE VOCABULARY — imported from the reconciler (one source of truth).
# ════════════════════════════════════════════════════════════════════════════


def test_provenance_vocabulary_is_the_reconcilers() -> None:
    # The checker validates EXACTLY the labels the reconciler STAMPS — a relabel on one side can
    # never silently desync the verification. These are imported, not re-spelled.
    src = build_post_merge_checker_script(repo="o/r")
    assert f"AUTO_MERGE_MAX_TIER = {AUTO_MERGE_MAX_TIER}" in src
    assert LABEL_PIPELINE_V2 in src
    assert LABEL_REVIEWED_PREFIX in src
    assert LABEL_RISK_PREFIX in src


def test_tier_cap_threads_from_the_reconciler() -> None:
    # The checker's provenance check uses the SAME #1158 tier-cap the reconciler's merge branch
    # enforced — so a tier>cap merge is flagged as bad provenance with the same threshold.
    src = build_post_merge_checker_script(repo="o/r", auto_merge_max_tier=2)
    assert "AUTO_MERGE_MAX_TIER = 2" in src


def test_the_three_escalation_labels_are_present() -> None:
    src = build_post_merge_checker_script(repo="o/r")
    for lbl in (NEEDS_HUMAN_ISSUE_OPEN, NEEDS_HUMAN_MASTER_RED, NEEDS_HUMAN_BAD_PROVENANCE):
        assert lbl in src
    assert LABEL_CHECKED_PREFIX in src
    assert DEFAULT_ESCALATED_LABEL in src


def test_require_run_provenance_defaults_off() -> None:
    # The run journal is a SECONDARY witness by default — the durable GitHub gate labels are the
    # primary provenance, so a degraded/empty list_runs read never manufactures a false violation.
    src = build_post_merge_checker_script(repo="o/r")
    assert "REQUIRE_RUN_PROVENANCE = False" in src


def test_require_run_provenance_flips_on() -> None:
    src = build_post_merge_checker_script(repo="o/r", require_run_provenance=True)
    assert "REQUIRE_RUN_PROVENANCE = True" in src


# ════════════════════════════════════════════════════════════════════════════
# DEPLOY SURFACE — read + escalate ONLY (no merge/close/resolve = structural
# maker≠checker-across-the-merge-boundary).
# ════════════════════════════════════════════════════════════════════════════


def test_required_tools_are_http_and_list_runs_only() -> None:
    # The checker READS GitHub + the run journal and POSTs only the escalation comment/labels.
    # No bash/read/write/edit/agent/gate — it cannot build, merge, or resolve anything.
    assert {t.type for t in REQUIRED_TOOLS} == {"http_request", "list_runs"}


def test_required_tools_have_no_duplicates() -> None:
    types = [t.type for t in REQUIRED_TOOLS]
    assert len(types) == len(set(types))


def test_github_server_is_get_post_only_no_mutation_of_the_merge() -> None:
    # The structural guarantee: the surface carries NO PUT/PATCH/DELETE, so the checker CANNOT
    # merge, close an issue, unlabel, or resolve a gate — it can only escalate (POST comment +
    # label). A checker that could mutate the merge it judges would not be an uncorrelated backstop.
    assert len(REQUIRED_HTTP_SERVERS) == 1
    server = REQUIRED_HTTP_SERVERS[0]
    assert server.name == "github"
    repos = [r for r in server.routes if r.path_pattern == "/repos/**"]
    assert len(repos) == 1
    assert set(repos[0].methods or []) == {"GET", "POST"}
    assert "PUT" not in (repos[0].methods or [])
    assert "PATCH" not in (repos[0].methods or [])
    assert "DELETE" not in (repos[0].methods or [])
    assert repos[0].allow_query is True


def test_workflow_create_is_valid_and_carries_surface() -> None:
    wc = build_post_merge_checker_workflow_create(name="dev-pipeline-post-merge-checker")
    assert isinstance(wc, WorkflowCreate)
    assert wc.name == "dev-pipeline-post-merge-checker"
    assert wc.script == build_post_merge_checker_script()
    assert {t.type for t in wc.tools} == {t.type for t in REQUIRED_TOOLS}
    assert len(wc.http_servers) == 1
    server = wc.http_servers[0]
    assert isinstance(server, HttpServerSpec)
    assert server.name == "github"


def test_workflow_create_github_server_name_threads_into_script() -> None:
    wc = build_post_merge_checker_workflow_create(name="r", github_server="gh")
    assert any(isinstance(s, HttpServerSpec) and s.name == "gh" for s in wc.http_servers)
    assert "'gh'" in wc.script or '"gh"' in wc.script


def test_workflow_create_forwards_require_run_provenance_kwarg() -> None:
    wc = build_post_merge_checker_workflow_create(name="r", require_run_provenance=True)
    assert "REQUIRE_RUN_PROVENANCE = True" in wc.script
