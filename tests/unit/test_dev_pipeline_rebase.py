"""Unit tests for the rebase/sync recovery stage (#1385).

``dev_pipeline.py`` used to have NO rebase/resolve logic: when master moved under an open
PR it went DIRTY/non-mergeable, GitHub could not compute ``refs/pull/N/merge``, CI never
re-ran, and the run jammed at the CI watch ("no checks reported"). Resolving such PRs was a
manual CEO-seat operation (hand ``git rebase`` + re-dispatch).

The fix adds a sync stage to the workflow *script body* that HEALS the branch instead of
parking or orphaning it:

  - ``_needs_rebase(pr)``    — read GitHub's ``mergeable``/``mergeable_state`` and decide.
  - ``_rebase_command(...)`` — a mechanical ``git rebase origin/master`` + force-push-with-
    lease bash node returning a DISTINCT exit code (noop / done / conflict / error).
  - ``_sync_branch(...)``    — the orchestrator: probe mergeability, try the mechanical
    rebase, hand a REAL conflict to the SAME fix agent ``fix_ci`` uses (bounded to
    ``MAX_REBASE_ATTEMPTS``), and report ``noop`` / ``rebased`` / ``conflict`` / ``error``
    for the caller to drive (re-enter CI / park at the new ``rebase_conflict`` gate).

These helpers live inside the workflow script source (``_BODY``), so they are not importable
as module attributes. We build the production script and ``exec`` it in a fresh namespace,
inject fake async ``gh`` / ``tool`` / ``agent`` capabilities, then drive the coroutines.
No LLM, no real I/O, no time.
"""

from __future__ import annotations

import asyncio
from typing import Any

from aios.workflows.dev_pipeline import build_dev_pipeline_script


class AgentError(Exception):
    """The runtime-provided AgentError name the body's ``except AgentError`` resolves."""


def _ns(
    *,
    gh: Any = None,
    tool: Any = None,
    agent: Any = None,
) -> dict[str, Any]:
    """A fresh exec namespace for the script body with runtime names injected."""
    src = build_dev_pipeline_script(
        implement_agent_id="a",
        review_agent_id="b",
        fix_agent_id="c",
        ci_agent_id="d",
        risk_agent_id="e",
    )
    namespace: dict[str, Any] = {}
    exec(compile(src, "dev_pipeline_script", "exec"), namespace)
    logs: list[str] = []
    namespace["log"] = lambda *a: logs.append(" ".join(str(x) for x in a))
    namespace["_LOGS"] = logs
    namespace["AgentError"] = AgentError
    if gh is not None:
        namespace["gh"] = gh
    if tool is not None:
        namespace["tool"] = tool
    if agent is not None:
        namespace["agent"] = agent
    return namespace


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


REPO = "eumemic/aios"
PR = 1214
BRANCH = "issue-1214"


# ─── _needs_rebase: read mergeability correctly ───────────────────────────────


def test_needs_rebase_true_for_dirty_and_behind() -> None:
    ns = _ns()
    assert ns["_needs_rebase"]({"mergeable_state": "dirty"}) is True
    assert ns["_needs_rebase"]({"mergeable_state": "behind"}) is True


def test_needs_rebase_true_when_mergeable_is_false() -> None:
    # GitHub already computed a real conflict.
    ns = _ns()
    assert ns["_needs_rebase"]({"mergeable": False, "mergeable_state": "unknown"}) is True


def test_needs_rebase_false_for_clean_and_benign_states() -> None:
    ns = _ns()
    for state in ("clean", "unstable", "has_hooks", "blocked"):
        assert ns["_needs_rebase"]({"mergeable": True, "mergeable_state": state}) is False


def test_needs_rebase_false_when_mergeable_unknown_and_state_benign() -> None:
    # mergeable=None means GitHub hasn't recomputed yet; a benign state must NOT force a
    # rebase of a clean PR (idempotence — we don't churn a healthy branch).
    ns = _ns()
    assert ns["_needs_rebase"]({"mergeable": None, "mergeable_state": "clean"}) is False


def test_needs_rebase_false_for_non_dict() -> None:
    ns = _ns()
    assert ns["_needs_rebase"](None) is False


# ─── _rebase_command: the mechanical rebase bash node ─────────────────────────


def test_rebase_command_force_pushes_with_lease_not_force() -> None:
    ns = _ns()
    cmd = ns["_rebase_command"](REPO, BRANCH)
    assert "git rebase origin/master" in cmd
    assert "--force-with-lease" in cmd
    # never a bare --force (which would clobber a concurrent push)
    assert "push --force " not in cmd and "push --force\n" not in cmd


def test_rebase_command_emits_distinct_exit_codes() -> None:
    ns = _ns()
    cmd = ns["_rebase_command"](REPO, BRANCH)
    # noop / done / conflict / error are each reachable via a distinct exit
    assert "exit 75" in cmd  # REBASE_EXIT_NOOP
    assert "exit 0" in cmd  # REBASE_EXIT_DONE
    assert "exit 76" in cmd  # REBASE_EXIT_CONFLICT
    assert "exit 77" in cmd  # REBASE_EXIT_ERROR
    # the no-op short-circuit (already current with master) precedes the rebase attempt
    assert "merge-base --is-ancestor origin/master HEAD" in cmd


def test_rebase_command_aborts_on_conflict() -> None:
    # A real conflict must `git rebase --abort` so the branch is left untouched.
    ns = _ns()
    cmd = ns["_rebase_command"](REPO, BRANCH)
    assert "git rebase --abort" in cmd


# ─── fakes for the _sync_branch orchestrator ──────────────────────────────────


class _FakeGH:
    def __init__(self, pr_payload: dict[str, Any]) -> None:
        self.pr_payload = pr_payload
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, method: str, path: str, body: Any = None) -> dict[str, Any]:
        self.calls.append((method, path))
        import json

        return {"status": 200, "body": json.dumps(self.pr_payload)}


class _FakeBash:
    """A fake ``tool`` that returns a queued exit_code per bash invocation."""

    def __init__(self, exit_codes: list[int]) -> None:
        self.exit_codes = list(exit_codes)
        self.commands: list[str] = []

    async def __call__(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        assert name == "bash"
        self.commands.append(args["command"])
        code = self.exit_codes.pop(0) if self.exit_codes else 0
        return {"exit_code": code, "stdout": "out", "stderr": "err", "timed_out": False}


class _FakeAgent:
    """A fake ``agent`` recording its inputs; returns a head_sha or raises per script."""

    def __init__(self, *, error: bool = False, head_sha: str = "newsha") -> None:
        self.error = error
        self.head_sha = head_sha
        self.inputs: list[dict[str, Any]] = []

    async def __call__(self, input: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        self.inputs.append(input)
        if self.error:
            raise AgentError("agent down")
        return {"head_sha": self.head_sha, "pushed": True}


def _sync(ns: dict[str, Any]) -> Any:
    return _run(ns["_sync_branch"](REPO, PR, BRANCH, ["a comment"], None))


# ─── _sync_branch: a clean/current branch is an idempotent no-op ──────────────


def test_sync_clean_pr_is_noop_without_touching_bash() -> None:
    gh = _FakeGH({"mergeable": True, "mergeable_state": "clean"})
    tool = _FakeBash([])
    ns = _ns(gh=gh, tool=tool, agent=_FakeAgent())
    result = _sync(ns)
    assert result == {"outcome": "noop"}
    # idempotence: a mergeable branch never runs the rebase bash node at all
    assert tool.commands == []


def test_sync_already_current_branch_is_noop() -> None:
    # DIRTY per GitHub but the mechanical rebase reports NOOP (already current — a re-driven
    # stage on an already-healed branch). The caller must NOT re-enter CI.
    gh = _FakeGH({"mergeable_state": "dirty"})
    tool = _FakeBash([75])  # REBASE_EXIT_NOOP
    ns = _ns(gh=gh, tool=tool, agent=_FakeAgent())
    result = _sync(ns)
    assert result == {"outcome": "noop"}


# ─── _sync_branch: a clean mechanical rebase heals the branch ─────────────────


def test_sync_clean_mechanical_rebase_returns_rebased() -> None:
    gh = _FakeGH({"mergeable_state": "behind"})
    tool = _FakeBash([0])  # REBASE_EXIT_DONE
    agent = _FakeAgent()
    ns = _ns(gh=gh, tool=tool, agent=agent)
    result = _sync(ns)
    assert result["outcome"] == "rebased"
    # a mechanical rebase needs no fix agent
    assert agent.inputs == []
    # exactly one bash invocation (the mechanical rebase)
    assert len(tool.commands) == 1


# The post-rebase tip the live PR re-read reports (#1389): a 40-char SHA-1, distinct from any
# stale pre-rebase head_sha the caller might otherwise carry into the CI re-entry.
_LIVE_HEAD_SHA = "abcdef0123456789abcdef0123456789abcdef01"


def test_sync_mechanical_rebase_threads_live_post_rebase_head_sha() -> None:
    # #1389: after the mechanical rebase force-pushes the rebased tip, _sync_branch RE-READS
    # the live PR head (GET /pulls/{n}) and returns THAT sha so the caller's CI re-entry keys
    # on the rebased tip — not the stale pre-rebase head_sha.
    gh = _FakeGH({"mergeable_state": "dirty", "head": {"sha": _LIVE_HEAD_SHA}})
    tool = _FakeBash([0])  # REBASE_EXIT_DONE
    ns = _ns(gh=gh, tool=tool, agent=_FakeAgent())
    result = _sync(ns)
    assert result["outcome"] == "rebased"
    assert result["head_sha"] == _LIVE_HEAD_SHA
    # the live re-read actually happened: a SECOND GET /pulls/{n} after the mergeability probe
    assert gh.calls.count(("GET", f"/repos/{REPO}/pulls/{PR}")) == 2


def test_sync_agent_resolved_conflict_prefers_live_head_over_agent_report() -> None:
    # #1389 (agent path): the fix agent self-reports a head_sha, but the confirm rebase may
    # have force-pushed a new tip on top of it. The live PR re-read is authoritative — its sha
    # is threaded, NOT the agent's (possibly stale) self-report.
    gh = _FakeGH({"mergeable_state": "dirty", "head": {"sha": _LIVE_HEAD_SHA}})
    tool = _FakeBash([76, 0])  # mechanical conflict, then confirm DONE
    agent = _FakeAgent(head_sha="agent-reported-sha")
    ns = _ns(gh=gh, tool=tool, agent=agent)
    result = _sync(ns)
    assert result["outcome"] == "rebased"
    assert result["head_sha"] == _LIVE_HEAD_SHA  # live tip wins over the agent's self-report


def test_sync_agent_resolved_conflict_falls_back_to_agent_sha_when_live_read_blank() -> None:
    # #1389 fallback: when the live PR re-read yields no usable SHA-1 (head absent), the agent's
    # reported head_sha is used rather than returning an empty head_sha.
    gh = _FakeGH({"mergeable_state": "dirty"})  # no head.sha in the payload
    tool = _FakeBash([76, 75])  # conflict, then confirm NOOP
    agent = _FakeAgent(head_sha="agentsha")
    ns = _ns(gh=gh, tool=tool, agent=agent)
    result = _sync(ns)
    assert result["outcome"] == "rebased"
    assert result["head_sha"] == "agentsha"


# ─── _sync_branch: a REAL conflict is resolved by the fix agent (bounded) ─────


def test_sync_conflict_resolved_by_fix_agent() -> None:
    # mechanical rebase CONFLICTs (76); the fix agent runs; the confirm rebase reports NOOP
    # (76 then 75) -> resolved.
    gh = _FakeGH({"mergeable_state": "dirty"})
    tool = _FakeBash([76, 75])  # conflict, then confirm noop
    agent = _FakeAgent(head_sha="resolvedsha")
    ns = _ns(gh=gh, tool=tool, agent=agent)
    result = _sync(ns)
    assert result["outcome"] == "rebased"
    assert result["head_sha"] == "resolvedsha"
    # the fix agent was dispatched exactly once with the rebase resolve-conflicts hint
    assert len(agent.inputs) == 1
    assert "rebase_hint" in agent.inputs[0]
    assert agent.inputs[0]["task"] == "fix_ci"


def test_sync_fix_agent_is_the_same_fix_agent_id() -> None:
    # The resolve-conflicts hand-off MUST use FIX_AGENT_ID (the same agent fix_ci invokes).
    captured: list[Any] = []
    gh = _FakeGH({"mergeable_state": "dirty"})
    tool = _FakeBash([76, 75])

    class _CapAgent:
        async def __call__(self, input: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
            captured.append(kwargs.get("agent_id"))
            return {"head_sha": "s", "pushed": True}

    ns = _ns(gh=gh, tool=tool, agent=_CapAgent())
    _sync(ns)
    # "c" is the fix_agent_id passed to build in _ns
    assert captured == ["c"]


def test_sync_unresolvable_conflict_returns_conflict_after_budget() -> None:
    # mechanical conflict, then BOTH fix attempts fail to heal (confirm stays conflicting):
    # 76 (mechanical), 76 (confirm 1), 76 (confirm 2). Bounded at MAX_REBASE_ATTEMPTS=2.
    gh = _FakeGH({"mergeable_state": "dirty"})
    tool = _FakeBash([76, 76, 76])
    agent = _FakeAgent()
    ns = _ns(gh=gh, tool=tool, agent=agent)
    result = _sync(ns)
    assert result["outcome"] == "conflict"
    # exactly MAX_REBASE_ATTEMPTS fix-agent dispatches (bounded)
    assert len(agent.inputs) == 2


def test_sync_fix_agent_error_is_bounded_then_conflict() -> None:
    # The fix agent errors on every attempt; the loop is bounded and ends at conflict (the
    # caller parks at the rebase_conflict gate) rather than crashing the run.
    gh = _FakeGH({"mergeable_state": "dirty"})
    tool = _FakeBash([76])  # mechanical conflict; no confirm runs (agent errors first)
    agent = _FakeAgent(error=True)
    ns = _ns(gh=gh, tool=tool, agent=agent)
    result = _sync(ns)
    assert result["outcome"] == "conflict"
    assert len(agent.inputs) == 2  # both attempts tried


# ─── _sync_branch: a plumbing failure fails closed ────────────────────────────


def test_sync_plumbing_error_fails_closed_to_error() -> None:
    # A clone/fetch/push failure (exit 77) cannot prove the branch is healthy -> fail closed
    # to "error" so the caller parks at the rebase_conflict gate (never waved through).
    gh = _FakeGH({"mergeable_state": "dirty"})
    tool = _FakeBash([77])  # REBASE_EXIT_ERROR
    ns = _ns(gh=gh, tool=tool, agent=_FakeAgent())
    result = _sync(ns)
    assert result["outcome"] == "error"
    assert "detail" in result


# ─── _shas_equal: the conservative commit-identity comparison (aios#1392 Fix 2) ──

# A real PR head and a wrong-parent orphan, both valid SHA-1s.
_HEAD = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
_ORPHAN = "0000000000000000000000000000000000000abc"


def test_shas_equal_exact_match() -> None:
    ns = _ns()
    assert ns["_shas_equal"](_HEAD, _HEAD) is True


def test_shas_equal_case_insensitive() -> None:
    ns = _ns()
    assert ns["_shas_equal"](_HEAD.upper(), _HEAD.lower()) is True


def test_shas_equal_abbreviation_prefix_matches() -> None:
    # git's short SHA is a prefix of the full one — they name the same commit.
    ns = _ns()
    assert ns["_shas_equal"](_HEAD[:8], _HEAD) is True
    assert ns["_shas_equal"](_HEAD, _HEAD[:12]) is True


def test_shas_equal_different_commits_do_not_match() -> None:
    ns = _ns()
    assert ns["_shas_equal"](_HEAD, _ORPHAN) is False


def test_shas_equal_rejects_too_short_or_empty() -> None:
    # A stray empty / 1-char value must never spuriously prefix-match (git's min unambiguous
    # abbreviation is 7) — otherwise "" would "match" everything and re-open the false-green.
    ns = _ns()
    assert ns["_shas_equal"]("", _HEAD) is False
    assert ns["_shas_equal"](_HEAD, "") is False
    assert ns["_shas_equal"]("a1b2c3", _HEAD) is False  # 6 chars < 7
    assert ns["_shas_equal"](None, _HEAD) is False
