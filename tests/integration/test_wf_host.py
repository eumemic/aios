"""B1.1 + B1.3 — the out-of-process script host + the .send() driver.

These spawn the real ``aios.workflows.wf_script_host`` subprocess (no DB). The
load-bearing tests are the isolation ones (the child can't reach the master key
/ pool) and the runaway-containment one (a CPU bomb is killed by the wall-clock
deadline and the parent stays healthy).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
from typing import Any

import pytest

from aios.workflows.determinism import content_hash, storable_text
from aios.workflows.host_launcher import EmittedAnnotation, HostOutcome, run_script_host
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


async def test_budget_emits_frontier_and_memo_fast_forwards() -> None:
    src = "async def main(input):\n    return await budget()"
    first = await _run(src)
    assert first.kind == "suspended"
    assert len(first.emitted) == 1
    assert first.emitted[0].capability_id == "budget"
    key = first.emitted[0].call_key

    view = {"total_usd": 2.0, "spent_usd": 0.5, "remaining_usd": 1.5}
    second = await _run(src, memo={key: {"ok": view}})
    assert second.kind == "returned"
    assert second.value == view

    third = await _run(src, memo={key: {"ok": None}})
    assert third.kind == "returned"
    assert third.value is None


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
        "async def main(input):\n    return await gate({'t': {1, 2}})"
    )  # set spec rejected (hash-ordering hazard)
    assert out.kind == "raised"
    assert "WorkflowInputTypeError" in (out.error_repr or "")


async def test_tuple_spec_is_coerced_not_rejected() -> None:
    # Tuples canonicalize as lists: a tuple-spec gate suspends, keyed exactly as
    # its list twin (the author and an upstream JSON round-trip agree).
    out = await _run("async def main(input):\n    return await gate({'t': (1, 2)})")
    assert out.kind == "suspended"
    assert out.emitted[0].call_key == f"sha:{content_hash('gate', {'t': [1, 2]})}#0"


async def test_missing_main_is_raised() -> None:
    out = await _run("x = 1")
    assert out.kind == "raised"
    assert "main(input)" in (out.error_repr or "")


# ─── curated stdlib + classes (T5) ───────────────────────────────────────────


async def test_script_can_import_curated_stdlib() -> None:
    out = await _run(
        r"""
        import collections.abc  # an entry admits its submodules
        import json
        import math
        import re
        from urllib.parse import quote

        async def main(input):
            data = json.loads('{"items": ["a 12", "b 7"]}')
            nums = [int(re.search(r"\d+", s).group()) for s in data["items"]]
            return {
                "total": sum(nums),
                "hypot": math.hypot(3, 4),
                "quoted": quote("a b"),
                "encoded": json.dumps(sorted(nums)),
                "is_seq": isinstance(nums, collections.abc.Sequence),
            }
        """
    )
    assert out.kind == "returned"
    assert out.value == {
        "total": 19,
        "hypot": 5.0,
        "quoted": "a%20b",
        "encoded": "[7, 12]",
        "is_seq": True,
    }


async def test_script_parses_json_tool_body_with_regex() -> None:
    """The acceptance workload: parse a JSON HTTP body and regex it — the glue
    every non-trivial coordination script needs (previously unwritable: scripts
    hand-rolled JSON escaping and delegated parsing to agent() children)."""
    source = r"""
    import json
    import re

    async def main(input):
        resp = await tool("http_request", {"method": "GET", "url": "https://api.test/x"})
        body = json.loads(resp["body"])
        return re.findall(r"#(\d+)", body["title"])
    """
    first = await _run(source)
    assert first.kind == "suspended"  # the tool call is the frontier
    (cap,) = first.emitted
    memo = {cap.call_key: {"ok": {"status": 200, "body": '{"title": "fixes #12 and #7"}'}}}
    out = await _run(source, memo=memo)
    assert out.kind == "returned"
    assert out.value == ["12", "7"]


async def test_script_can_define_classes() -> None:
    out = await _run(
        """
        class Node:
            def __init__(self, name):
                self.name = name

            @property
            def label(self):
                return self.name.upper()

        class Leaf(Node):
            def __init__(self, name):
                super().__init__(name)

        async def main(input):
            return Leaf("tip").label
        """
    )
    assert out.kind == "returned"
    assert out.value == "TIP"


async def test_script_dataclasses_have_live_annotations() -> None:
    """Annotations are LIVE objects (``dont_inherit`` pins this: without it the
    script inherits the host's PEP 563 future and every annotation stringifies).
    The ``live`` probe is the actual pin — InitVar correctness alone would pass
    either way, because the registered script module lets dataclasses resolve
    even *string* annotations correctly."""
    out = await _run(
        """
        from dataclasses import InitVar, asdict, dataclass, field

        @dataclass
        class Row:
            x: int
            tags: list = field(default_factory=list)
            scale: InitVar[int] = 1

            def __post_init__(self, scale):
                self.x *= scale

        async def main(input):
            return {"row": asdict(Row(5, scale=3)), "live": Row.__annotations__["x"] is int}
        """
    )
    assert out.kind == "returned"
    assert out.value == {"row": {"x": 15, "tags": []}, "live": True}


async def test_annotations_evaluate_eagerly_against_script_namespace() -> None:
    # The flip side of live annotations: an out-of-scope name is a loud NameError
    # at def time, not a silently-stored string. (CPython 3.13 semantics — under
    # PEP 649 (3.14+) annotations evaluate lazily and this assertion will need
    # revisiting at the version bump; the `live` probe above is the durable pin.)
    out = await _run("async def main(input: Any):\n    return 1")
    assert out.kind == "raised"
    assert "NameError" in (out.error_repr or "")


async def test_disallowed_import_is_a_loud_import_error() -> None:
    stmts = (
        "import os",
        "import importlib",
        "import urllib.request",  # allowlisting urllib.parse does not open the package
        "from urllib import parse",  # the name imported is `urllib`: rejected; use `import urllib.parse`
        "from __future__ import annotations",  # would stringify annotations (PEP 563)
    )
    outs = await asyncio.gather(
        *(_run(f"{stmt}\n\nasync def main(input):\n    return 1") for stmt in stmts)
    )
    for stmt, out in zip(stmts, outs, strict=True):
        assert out.kind == "raised", stmt
        assert "ImportError" in (out.error_repr or ""), stmt
        assert "may only import" in (out.error_repr or ""), stmt  # the error IS the docs


async def test_non_json_return_is_an_author_error_not_a_host_crash() -> None:
    out = await _run(
        """
        from dataclasses import dataclass

        @dataclass
        class Point:
            x: int

        async def main(input):
            return Point(1)  # forgot asdict()
        """
    )
    assert out.kind == "raised"
    assert out.error_kind == "author_exception"
    assert "WorkflowInputTypeError" in (out.error_repr or "")
    assert "return: unsupported workflow input type Point" in (out.error_repr or "")
    # NaN would survive json.dumps (allow_nan) only to detonate at the parent's
    # jsonb cast — the domain validator rejects it at the source instead.
    nan = await _run("async def main(input):\n    return float('nan')")
    assert nan.kind == "raised"
    assert "return: non-finite float" in (nan.error_repr or "")
    # Same class: Postgres jsonb cannot store NUL — caught here, not as a parent
    # crashloop after a "successful" host run.
    nul = await _run("async def main(input):\n    return chr(0)")
    assert nul.kind == "raised"
    assert "return: NUL" in (nul.error_repr or "")
    # A cyclic return RecursionErrors inside the validator — still an author
    # error, never an rc=1 host crash with the cause lost to stderr.
    cyc = await _run("async def main(input):\n    x = []\n    x.append(x)\n    return x")
    assert cyc.kind == "raised"
    assert cyc.error_kind == "author_exception"
    assert "RecursionError" in (cyc.error_repr or "")


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


async def test_pipeline_stage_receives_item_and_index() -> None:
    """A three-arg stage gets ``(prev, item, index)``: for the FIRST stage ``prev``
    equals the item itself, and ``index`` is the element's position in ``items``."""
    out = await _run(
        "async def main(input):\n"
        "    return await pipeline([10, 20], lambda p, it, ix: {'p': p, 'it': it, 'ix': ix})"
    )
    assert out.kind == "returned"
    assert out.value == [
        {"p": 10, "it": 10, "ix": 0},
        {"p": 20, "it": 20, "ix": 1},
    ]


async def test_pipeline_two_arg_stage_gets_prev_and_item() -> None:
    """A two-arg stage gets ``(prev, item)``; for the first stage prev == item, so
    ``p + it`` is ``5 + 5``."""
    out = await _run("async def main(input):\n    return await pipeline([5], lambda p, it: p + it)")
    assert out.kind == "returned"
    assert out.value == [10]


async def test_pipeline_var_positional_stage_gets_all_three() -> None:
    """A ``*args`` stage opts into all three positional values ``(prev, item, index)``."""
    out = await _run("async def main(input):\n    return await pipeline([7], lambda *a: list(a))")
    assert out.kind == "returned"
    assert out.value == [[7, 7, 0]]


async def test_pipeline_non_inspectable_stage_defaults_to_one_arg() -> None:
    """A callable whose ``inspect.signature`` raises (``str`` is one such builtin in
    CPython 3.13) falls back to the legacy 1-arg (prev-only) call — so ``str`` is a
    valid single-argument transform."""
    out = await _run("async def main(input):\n    return await pipeline([42], str)")
    assert out.kind == "returned"
    assert out.value == ["42"]


async def test_pipeline_none_short_circuit_on_agent_error() -> None:
    """An item whose chain raises AgentError at an early stage skips the remaining
    stages and lands None in its slot — the downstream multi-arg stage never runs.
    (PIN: this already works via the parallel barrier; the arity rewrite must not
    regress it.)"""
    src = (
        "async def main(input):\n"
        "    return await pipeline([1, 2], lambda x: agent('s', x),"
        " lambda prev, item, index: prev + 1)"
    )
    first = await _run(src)
    assert first.kind == "suspended"
    k0, k1 = (e.call_key for e in first.emitted)
    out = await _run(src, memo={k0: {"ok": 10}, k1: {"error": {"kind": "child_errored"}}})
    assert out.kind == "returned"
    assert out.value == [11, None]  # item 1: 10 -> +1 -> 11; item 2: chain raised -> None


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
    assert out.kind == "raised"  # rejected by the curated-import shim (still ImportError)
    assert "ImportError" in (out.error_repr or "")


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


async def test_hash_ordering_is_pinned_across_wakes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The launcher pins PYTHONHASHSEED=0 in the child: str-hash-dependent orderings
    (``list({...})``) must be identical on every wake, whatever seed the worker
    itself runs under — an inherited/random seed would desync any call_key built
    from such an ordering."""
    strings = '{"alpha", "bravo", "charlie", "delta", "echo"}'  # 0/1/2 orderings all differ
    outs = []
    for worker_seed in ("1", "2"):
        monkeypatch.setenv("PYTHONHASHSEED", worker_seed)
        outs.append(await _run(f"async def main(input):\n    return list({strings})"))
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        f"import json; print(json.dumps(list({strings})))",
        stdout=asyncio.subprocess.PIPE,
        env={**os.environ, "PYTHONHASHSEED": "0"},  # inherit env, pin only the seed
    )
    stdout, _ = await proc.communicate()
    assert proc.returncode == 0
    assert [o.kind for o in outs] == ["returned", "returned"]
    assert outs[0].value == outs[1].value == json.loads(stdout)


# ─── runaway containment (B1.1) ──────────────────────────────────────────────


async def test_cpu_bomb_is_killed_and_parent_survives() -> None:
    out = await _run(
        "async def main(input):\n    while True:\n        pass",
        deadline_seconds=2.0,  # the CPU rlimit derives from this (deadline + 1s)
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


async def test_big_memo_replay_outlives_a_tiny_base_deadline() -> None:
    """The wall budget scales with INIT bytes (#780). The base here (0.01s) is
    deliberately below interpreter spawn time (~20-50ms), so an UNSCALED host
    deterministically times out — what passes is the multi-MiB memo's scale
    allowance. This pins that ``_scaled_seconds`` is actually wired into
    ``run_script_host``, not merely unit-green."""
    source = "async def main(input):\n    g = await gate('x')\n    return len(g)"
    first = await _run(source)
    (cap,) = first.emitted  # learn the gate's call_key
    memo = {
        cap.call_key: {"ok": "v"},
        # An unmatched memo key is never consulted by the driver — it exists only
        # to inflate the INIT frame past 2MiB.
        "pad": {"ok": "x" * (2 * 1024 * 1024)},
    }
    out = await _run(source, memo=memo, deadline_seconds=0.01)
    assert out.kind == "returned"
    assert out.value == 1


async def test_cpu_rlimit_derives_from_the_scaled_deadline() -> None:
    """The child's CPU budget must track the SCALED wall deadline (int(deadline)+1),
    not the base — pinned end-to-end by reading the env the child was handed (via
    the documented builtins-escape, like the env-scrub test). A base-derived
    mutation would hand a multi-MiB replay a 1s CPU budget."""
    source = (
        "async def main(input):\n"
        "    return gate.__globals__['os'].environ['AIOS_WF_RLIMIT_CPU_S']\n"
    )
    tiny = await _run(source, deadline_seconds=2.0)
    assert tiny.kind == "returned"
    assert tiny.value == "3"  # tiny INIT: int(~2.0) + 1
    big = await _run(
        source,
        memo={"pad": {"ok": "x" * (2 * 1024 * 1024)}},  # ~2MiB INIT → ~60s wall
        deadline_seconds=0.01,
    )
    assert big.kind == "returned"
    assert 61 <= int(big.value) <= 62  # int(0.01 + ~2*30) + 1


async def test_print_is_unavailable_so_author_cannot_corrupt_stdout() -> None:
    out = await _run("async def main(input):\n    print('x')\n    return 1")
    assert out.kind == "raised"  # print is not in SAFE_BUILTINS


# ─── log() / phase() as journaled annotations (B-783) ────────────────────────


def _ann_key(kind: str, text: str, *, path: str = "") -> str:
    """The branch-local annotation call_key the host derives — ``path`` is the
    parallel branch prefix ("" for root, "0.0/" for the first child)."""
    return f"{path}sha:{content_hash('annotation', {'kind': kind, 'text': text})}#0"


async def test_log_is_a_journaled_annotation_not_stderr() -> None:
    # The stderr reversion: log() no longer writes stderr — it emits a journaled
    # annotation frame, keyed branch-locally so the journal can dedup it across replays.
    out = await _run("async def main(input):\n    log('diagnostic')\n    return 7")
    assert out.kind == "returned"
    assert out.value == 7
    assert "diagnostic" not in out.stderr  # stderr is crash-diagnostics-only now
    assert out.annotations == [
        EmittedAnnotation(
            call_key=_ann_key("log", "diagnostic"),
            payload={"kind": "log", "text": "diagnostic"},
        )
    ]


async def test_phase_is_a_journaled_annotation() -> None:
    out = await _run("async def main(input):\n    phase('build')\n    return 1")
    assert out.kind == "returned"
    assert out.annotations == [
        EmittedAnnotation(
            call_key=_ann_key("phase", "build"), payload={"kind": "phase", "text": "build"}
        )
    ]


async def test_log_space_joins_args_like_print() -> None:
    out = await _run("async def main(input):\n    log('a', 1, True)\n    return 1")
    assert out.annotations[0].payload == {"kind": "log", "text": "a 1 True"}


async def test_annotations_emit_in_order_before_the_frontier() -> None:
    # phase() then log() then a suspending gate: both annotations are emitted (in
    # execution order) alongside the one frontier capability.
    out = await _run(
        "async def main(input):\n    phase('p')\n    log('l')\n    return await gate(1)"
    )
    assert out.kind == "suspended"
    assert [(a.payload["kind"], a.payload["text"]) for a in out.annotations] == [
        ("phase", "p"),
        ("log", "l"),
    ]
    assert len(out.emitted) == 1  # the gate frontier


async def test_annotation_call_key_is_stable_across_replays() -> None:
    # The host re-emits annotations on EVERY wake with the IDENTICAL call_key; emit-once
    # is the journal's job (ON CONFLICT), so stability of the key across wakes is what
    # the host must guarantee.
    src = "async def main(input):\n    log('once')\n    r = await gate(1)\n    return r"
    first = await _run(src)
    assert [a.payload["text"] for a in first.annotations] == ["once"]
    gate_key = first.emitted[0].call_key
    second = await _run(src, memo={gate_key: {"ok": "v"}})
    assert second.kind == "returned"
    assert second.value == "v"
    assert second.annotations[0].call_key == first.annotations[0].call_key


_PARALLEL_LOG_SRC = """
    async def main(input):
        def mk(n):
            async def run():
                log('hi')
                return await gate(n)
            return run
        return await parallel([mk(1), mk(2)])
"""


async def test_parallel_branches_key_annotations_locally() -> None:
    # Both branches log the identical text 'hi'. Branch-local keying (path prefix)
    # makes the two annotations DISTINCT call_keys — they neither collide nor dedupe
    # each other — which is exactly why annotations route through the branch keyer.
    out = await _run(_PARALLEL_LOG_SRC)
    assert out.kind == "suspended"
    assert len(out.annotations) == 2
    assert all(a.payload == {"kind": "log", "text": "hi"} for a in out.annotations)
    assert {a.call_key for a in out.annotations} == {
        _ann_key("log", "hi", path="0.0/"),
        _ann_key("log", "hi", path="0.1/"),
    }


async def test_log_with_unstorable_text_does_not_kill_the_run() -> None:
    # NUL / unpaired surrogate is common in logged tool output (decoded bytes, JSON with
    # control chars). A diagnostic must never fail the run, so the host sanitizes the
    # text rather than letting the call_key validator raise (which would error the run).
    out = await _run(
        "async def main(input):\n    log('a' + chr(0) + chr(0xD800) + 'b')\n    return 9"
    )
    assert out.kind == "returned"
    assert out.value == 9
    text = out.annotations[0].payload["text"]
    assert "\x00" not in text
    text.encode("utf-8")  # storable: raises if a lone surrogate survived


def test_storable_neutralizes_nul_and_surrogates_losslessly() -> None:
    assert storable_text("ordinary text — é 🎉") == "ordinary text — é 🎉"
    assert "\x00" not in storable_text("a\x00b")
    storable_text("x" + chr(0xD800)).encode("utf-8")  # no raise → storable
