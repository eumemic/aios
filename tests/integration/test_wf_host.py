"""B1.1 + B1.3 — the out-of-process script host + the .send() driver.

These spawn the real ``aios.workflows.wf_script_host`` subprocess (no DB). The
load-bearing tests are the isolation ones (the child can't reach the master key
/ pool) and the runaway-containment one (a CPU bomb is killed by the wall-clock
deadline and the parent stays healthy).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from typing import Any

import pytest

from aios.workflows.host_launcher import HostOutcome, run_script_host
from aios.workflows.wf_script_host import MAX_PARALLEL_FANOUT


async def _run(
    source: str, *, input: Any = None, memo: dict[str, Any] | None = None, **kw: Any
) -> HostOutcome:
    return await run_script_host(source=textwrap.dedent(source), input=input, memo=memo or {}, **kw)


# ─── driver behaviour (B1.3) ─────────────────────────────────────────────────


async def test_pure_script_returns() -> None:
    out = await _run("async def main(input):\n    return input['x'] + 1", input={"x": 41})
    assert out.kind == "returned"
    assert out.value == 42
    assert out.emitted == []


async def test_gate_suspends_with_one_frontier() -> None:
    out = await _run(
        "async def main(input):\n    r = await gate({'q': 'ok?'})\n    return {'answer': r}"
    )
    assert out.kind == "suspended"
    assert len(out.emitted) == 1
    assert out.emitted[0].capability_id == "gate"
    assert out.emitted[0].call_key.startswith("sha:")


async def test_memoized_result_fast_forwards() -> None:
    src = "async def main(input):\n    r = await gate({'q': 'ok?'})\n    return {'answer': r}"
    first = await _run(src)
    key = first.emitted[0].call_key
    # Replay with the gate resolved → fast-forward past it. The memo entry is the
    # tagged outcome {"ok": value}; the driver unwraps it into the await (R3).
    second = await _run(src, memo={key: {"ok": "yes"}})
    assert second.kind == "returned"
    assert second.value == {"answer": "yes"}
    assert second.emitted == []


async def test_memoized_error_outcome_throws_agent_error() -> None:
    """An {"error"} memo outcome is thrown at the await as AgentError; uncaught, it
    propagates out of main and terminates the run (the bubble)."""
    src = "async def main(input):\n    return await agent('a1', 'go')"
    first = await _run(src)
    key = first.emitted[0].call_key
    out = await _run(src, memo={key: {"error": {"message": "boom"}}})
    assert out.kind == "raised"
    assert "AgentError: boom" in (out.error_repr or "")


async def test_memoized_error_outcome_is_catchable() -> None:
    """The headline: the author can try/except AgentError and carry on. `kind`
    surfaces the failure mode (here a model failure)."""
    src = (
        "async def main(input):\n"
        "    try:\n"
        "        await agent('a1', 'go')\n"
        "    except AgentError as e:\n"
        "        return {'caught': True, 'kind': e.kind}\n"
        "    return {'caught': False}"
    )
    first = await _run(src)
    key = first.emitted[0].call_key
    out = await _run(src, memo={key: {"error": {"kind": "child_errored"}}})
    assert out.kind == "returned"
    assert out.value == {"caught": True, "kind": "child_errored"}


async def test_no_return_outcome_raises_agent_no_return_subtype() -> None:
    """A no_return outcome raises AgentNoReturnError — caught by a blanket
    `except AgentError` (it is a subtype) and directly as the subtype."""
    catch_base = (
        "async def main(input):\n"
        "    try:\n"
        "        await agent('a1', 'go')\n"
        "    except AgentError as e:\n"
        "        return isinstance(e, AgentNoReturnError)"
    )
    first = await _run(catch_base)
    key = first.emitted[0].call_key
    out = await _run(catch_base, memo={key: {"error": {"kind": "no_return"}}})
    assert out.kind == "returned"
    assert out.value is True  # caught as AgentError, and it IS the no-return subtype

    catch_subtype = catch_base.replace("except AgentError as e", "except AgentNoReturnError")
    catch_subtype = catch_subtype.replace(
        "return isinstance(e, AgentNoReturnError)", "return 'caught-subtype'"
    )
    out2 = await _run(catch_subtype, memo={key: {"error": {"kind": "no_return"}}})
    assert out2.kind == "returned" and out2.value == "caught-subtype"


async def test_suspend_does_not_run_post_await_body() -> None:
    out = await _run(
        """
        async def main(input):
            x = await gate(1)
            raise RuntimeError('must not run before resume')
            return x
        """
    )
    assert out.kind == "suspended"  # the body after the await never executed


async def test_author_exception_is_terminal_raised() -> None:
    out = await _run("async def main(input):\n    raise ValueError('boom')")
    assert out.kind == "raised"
    assert out.error_kind == "author_exception"
    assert "ValueError: boom" in (out.error_repr or "")


async def test_agent_emits_a_frontier_for_block2() -> None:
    out = await _run("async def main(input):\n    return await agent('a1', input={'p': 1})")
    assert out.kind == "suspended"
    assert out.emitted[0].capability_id == "agent"


async def test_agent_requires_non_none_input() -> None:
    # input is required: a child born with no first user message would sit idle
    # forever. A bad call raises in the author's script → terminal RAISED.
    missing = await _run("async def main(input):\n    return await agent('a1')")
    assert missing.kind == "raised"
    assert "input" in (missing.error_repr or "")  # TypeError: missing required 'input'

    explicit_none = await _run("async def main(input):\n    return await agent('a1', None)")
    assert explicit_none.kind == "raised"
    assert "non-None input" in (explicit_none.error_repr or "")


async def test_bad_capability_input_is_raised() -> None:
    out = await _run(
        "async def main(input):\n    return await gate({'t': (1, 2)})"
    )  # tuple spec rejected
    assert out.kind == "raised"
    assert "WorkflowInputTypeError" in (out.error_repr or "")


async def test_missing_main_is_raised() -> None:
    out = await _run("x = 1")
    assert out.kind == "raised"
    assert "main(input)" in (out.error_repr or "")


# ─── parallel / pipeline (B2.G — the concurrent scheduler) ───────────────────

_PARALLEL_SRC = (
    "async def main(input):\n"
    "    return await parallel([lambda: agent('a', '1'), lambda: agent('b', '2')])"
)


async def test_parallel_emits_all_branches_as_one_frontier() -> None:
    """parallel fans its branches out as a single frontier — all children at once,
    not one at a time."""
    out = await _run(_PARALLEL_SRC)
    assert out.kind == "suspended"
    assert len(out.emitted) == 2
    assert {e.capability_id for e in out.emitted} == {"agent"}
    assert len({e.call_key for e in out.emitted}) == 2  # distinct keys


async def test_parallel_collects_results_in_order() -> None:
    """With every branch memoized, parallel fast-forwards to a results list in
    thunk order."""
    first = await _run(_PARALLEL_SRC)
    k0, k1 = (e.call_key for e in first.emitted)
    out = await _run(_PARALLEL_SRC, memo={k0: {"ok": "r1"}, k1: {"ok": "r2"}})
    assert out.kind == "returned"
    assert out.value == ["r1", "r2"]


async def test_parallel_barrier_is_none_on_error() -> None:
    """A branch whose agent errored (and which doesn't catch it) yields None in its
    slot; the barrier itself still returns."""
    first = await _run(_PARALLEL_SRC)
    k0, k1 = (e.call_key for e in first.emitted)
    out = await _run(
        _PARALLEL_SRC, memo={k0: {"ok": "r1"}, k1: {"error": {"kind": "child_errored"}}}
    )
    assert out.kind == "returned"
    assert out.value == ["r1", None]


async def test_parallel_branch_can_catch_agent_error() -> None:
    """The AgentError is thrown into the right branch's coroutine, so a branch can
    try/except it and substitute its own value."""
    src = (
        "async def main(input):\n"
        "    async def safe():\n"
        "        try:\n"
        "            return await agent('a', '1')\n"
        "        except AgentError:\n"
        "            return 'fallback'\n"
        "    return await parallel([safe, lambda: agent('b', '2')])"
    )
    first = await _run(src)
    k0, k1 = (e.call_key for e in first.emitted)
    out = await _run(src, memo={k0: {"error": {"kind": "child_errored"}}, k1: {"ok": "r2"}})
    assert out.kind == "returned"
    assert out.value == ["fallback", "r2"]


async def test_parallel_non_agent_error_fails_the_run() -> None:
    """Fail-hard: the barrier absorbs ONLY an agent failure (AgentError). Any OTHER
    exception out of a branch — an author bug — fails the WHOLE run loudly, never a
    silent None slot. This is the barrier's most load-bearing property."""
    src = (
        "async def main(input):\n"
        "    async def boom():\n"
        "        raise ValueError('author bug in a branch')\n"
        "    return await parallel([boom, lambda: agent('b', '2')])"
    )
    out = await _run(src)
    assert out.kind == "raised"
    assert out.error_repr is not None
    assert "ValueError" in out.error_repr
    assert "author bug in a branch" in out.error_repr


async def test_parallel_absorbs_author_raised_agent_error() -> None:
    """AgentError is the agent-failure signal, matched by type: a branch that lets an
    AgentError escape — even one the author raised directly, not from an agent() —
    becomes a None slot, exactly like an agent() that errored. (Counterpart to
    test_parallel_non_agent_error_fails_the_run: AgentError → None, anything else →
    fail-hard.)"""
    src = (
        "async def main(input):\n"
        "    async def manual():\n"
        "        raise AgentError('treat as a failed agent')\n"
        "    return await parallel([manual, lambda: agent('b', '2')])"
    )
    first = await _run(src)
    assert first.kind == "suspended"
    key = next(e.call_key for e in first.emitted if e.capability_id == "agent")
    out = await _run(src, memo={key: {"ok": "rb"}})
    assert out.kind == "returned"
    assert out.value == [None, "rb"]


async def test_empty_parallel_returns_empty_list() -> None:
    out = await _run("async def main(input):\n    return await parallel([])")
    assert out.kind == "returned"
    assert out.value == []


async def test_parallel_fanout_over_cap_raises_before_spawning() -> None:
    """A single parallel() wider than MAX_PARALLEL_FANOUT fails the run in the host —
    deterministically, before any child capability is emitted (none to spawn). The
    lifetime total across calls is bounded separately, parent-side."""
    n = MAX_PARALLEL_FANOUT + 1
    src = (
        "async def main(input):\n"
        f"    return await parallel([(lambda: agent('a', 'x')) for _ in range({n})])"
    )
    out = await _run(src)
    assert out.kind == "raised"
    assert out.error_kind == "too_wide_fanout"  # a distinct, machine-parseable kind
    assert out.error_repr is not None
    assert "fan-out" in out.error_repr and str(n) in out.error_repr
    assert out.emitted == []  # nothing to spawn — failed before opening the frontier


async def test_parallel_fanout_at_cap_is_allowed() -> None:
    """Exactly MAX_PARALLEL_FANOUT branches is within the cap: the run suspends on the
    full frontier rather than raising (the boundary is strict ``>``)."""
    n = MAX_PARALLEL_FANOUT
    src = (
        "async def main(input):\n"
        f"    return await parallel([(lambda: agent('a', str(i))) for i in range({n})])"
    )
    out = await _run(src)
    assert out.kind == "suspended"
    assert len(out.emitted) == n


async def test_pipeline_runs_each_item_through_stages() -> None:
    """pipeline threads each item through stages (a sync transform then an agent
    leaf) independently; all items fan out at once."""
    src = (
        "async def main(input):\n"
        "    return await pipeline([1, 10], lambda x: x + 1, lambda x: agent('s', x))"
    )
    first = await _run(src)
    assert first.kind == "suspended"
    assert len(first.emitted) == 2  # one agent leaf per item, post sync stage
    k0, k1 = (e.call_key for e in first.emitted)
    out = await _run(src, memo={k0: {"ok": "a"}, k1: {"ok": "b"}})
    assert out.kind == "returned"
    assert out.value == ["a", "b"]


async def test_parallel_call_keys_are_replay_stable() -> None:
    """The scheduler's fixed creation-order sweep makes the per-branch call_keys
    identical across drives — the basis of replay-with-memo for parallel."""
    first = await _run(_PARALLEL_SRC)
    second = await _run(_PARALLEL_SRC)
    assert [e.call_key for e in first.emitted] == [e.call_key for e in second.emitted]


async def test_parallel_keys_are_branch_local_under_asymmetric_depth() -> None:
    """Determinism regression: two branches share a content hash but have different
    prerequisite depth (b0 awaits a gate first, b1 calls the agent directly). Keys
    are branch-LOCAL (prefixed by the branch path), so a partial-memo replay where
    b0 fast-forwards its gate and reaches the shared agent never steals b1's key —
    each branch keeps its own child, no result swap. (A global emission-order keyer
    would swap ordinals here.)"""
    src = (
        "async def main(input):\n"
        "    async def b0():\n"
        "        await gate('g')\n"
        "        return await agent('p', 'x')\n"
        "    async def b1():\n"
        "        return await agent('p', 'x')\n"
        "    return await parallel([b0, b1])"
    )
    # Drive 1 (empty memo): b0 blocks on its gate; b1 emits the shared agent.
    first = await _run(src)
    assert first.kind == "suspended"
    gate_key = next(e.call_key for e in first.emitted if e.capability_id == "gate")
    b1_agent_key = next(e.call_key for e in first.emitted if e.capability_id == "agent")

    # Drive 2 (gate resolved, b1's agent answered): b0 fast-forwards its gate and
    # reaches the shared agent — under its OWN branch path, a key distinct from b1's.
    second = await _run(src, memo={gate_key: {"ok": "g"}, b1_agent_key: {"ok": "rb1"}})
    assert second.kind == "suspended"
    new_keys = [e.call_key for e in second.emitted]
    assert b1_agent_key not in new_keys  # b1's key stayed memoized, not re-stolen
    assert len(new_keys) == 1  # only b0's own agent key
    b0_agent_key = new_keys[0]
    assert b0_agent_key != b1_agent_key  # branch-local — no ordinal swap

    # Drive 3: resolve b0's agent too → results correctly bound to their branches.
    out = await _run(
        src,
        memo={gate_key: {"ok": "g"}, b1_agent_key: {"ok": "rb1"}, b0_agent_key: {"ok": "rb0"}},
    )
    assert out.kind == "returned"
    assert out.value == ["rb0", "rb1"]


# ─── isolation (B1.1 — security-critical) ────────────────────────────────────


def test_host_transitive_imports_are_credential_free() -> None:
    """Importing the host must NOT pull in any module that holds the master key
    or the all-accounts pool — the whole isolation argument rests on this."""
    check = (
        "import aios.workflows.wf_script_host, sys;"
        "banned=[m for m in sys.modules if m.split('.')[0]=='aios' and "
        "m.startswith(('aios.harness','aios.crypto','aios.db','aios.services',"
        "'aios.tools','aios.sandbox'))];"
        "print(repr(banned))"
    )
    out = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", f"host leaked credential-bearing imports: {out.stdout}"


async def test_script_cannot_import_aios() -> None:
    out = await _run("async def main(input):\n    import aios.harness.runtime\n    return 1")
    assert out.kind == "raised"  # __import__ is not in SAFE_BUILTINS


async def test_injected_globals_cannot_reach_runtime() -> None:
    # gate.__globals__ is the host module's namespace — it has no runtime/crypto/pool.
    out = await _run(
        "async def main(input):\n"
        "    g = gate.__globals__\n"
        "    return sorted(k for k in g if 'runtime' in k or 'crypto' in k or 'pool' in k)"
    )
    assert out.kind == "returned"
    assert out.value == []


async def test_child_env_is_scrubbed_of_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    # gate.__globals__['os'].environ IS reachable (os is a host-module import), so
    # the child's *environment* must carry no secrets — that's what makes the
    # credential-free claim true. Regression for the env={**os.environ} spawn that
    # leaked AIOS_VAULT_KEY / AIOS_DB_URL / provider keys into author-reachable env.
    monkeypatch.setenv("AIOS_VAULT_KEY", "MASTER-KEY-SENTINEL")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SENTINEL")
    monkeypatch.setenv("AIOS_DB_URL", "postgres://SENTINEL")
    out = await _run(
        "async def main(input):\n"
        "    env = gate.__globals__['os'].environ\n"
        "    secrets = ('AIOS_VAULT_KEY', 'ANTHROPIC_API_KEY', 'AIOS_DB_URL')\n"
        "    return {'leaked': [k for k in secrets if k in env], 'has_path': 'PATH' in env}"
    )
    assert out.kind == "returned"
    assert out.value["leaked"] == []  # no secret crossed the spawn
    assert out.value["has_path"] is True  # but non-secret launch essentials still do


# ─── runaway containment (B1.1) ──────────────────────────────────────────────


async def test_cpu_bomb_is_killed_and_parent_survives() -> None:
    out = await _run(
        "async def main(input):\n    while True:\n        pass",
        cpu_seconds=2,
        deadline_seconds=2.0,
    )
    assert out.kind == "raised"
    assert out.error_kind in ("script_host_timeout", "script_host_crash")
    # The parent is still alive and usable — a normal run still works right after.
    ok = await _run("async def main(input):\n    return 'alive'")
    assert ok.kind == "returned" and ok.value == "alive"


async def test_address_space_bomb_is_contained_and_parent_survives() -> None:
    # RLIMIT_AS is active (a real DEFAULT_ADDRESS_SPACE_BYTES), so a giant alloc the
    # wall-clock deadline can't bound (it caps duration, not peak memory) is
    # contained — MemoryError, or an OOM/deadline kill — never returned. An explicit
    # small cap + deadline keep it fast; the alloc dwarfs any real RAM so it fails
    # even where setrlimit(RLIMIT_AS) is a no-op.
    out = await _run(
        "async def main(input):\n    big = [0] * (10**10)\n    return len(big)",
        address_space_bytes=256 * 1024 * 1024,
        deadline_seconds=5.0,
    )
    assert out.kind == "raised"
    ok = await _run("async def main(input):\n    return 'alive'")
    assert ok.kind == "returned" and ok.value == "alive"


async def test_log_goes_to_stderr_not_the_frame_stream() -> None:
    out = await _run("async def main(input):\n    log('diagnostic')\n    return 7")
    assert out.kind == "returned"
    assert out.value == 7  # stdout frame stream parsed cleanly despite the log()
    assert "diagnostic" in out.stderr


async def test_print_is_unavailable_so_author_cannot_corrupt_stdout() -> None:
    out = await _run("async def main(input):\n    print('x')\n    return 1")
    assert out.kind == "raised"  # print is not in SAFE_BUILTINS
