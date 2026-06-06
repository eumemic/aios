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

from aios.workflows.host_launcher import HostOutcome, run_script_host


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
    # Replay with the gate resolved → fast-forward past it to completion.
    second = await _run(src, memo={key: "yes"})
    assert second.kind == "returned"
    assert second.value == {"answer": "yes"}
    assert second.emitted == []


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


async def test_bad_capability_input_is_raised() -> None:
    out = await _run(
        "async def main(input):\n    return await gate({'t': 1.5})"
    )  # float spec rejected
    assert out.kind == "raised"
    assert "WorkflowInputTypeError" in (out.error_repr or "")


async def test_missing_main_is_raised() -> None:
    out = await _run("x = 1")
    assert out.kind == "raised"
    assert "main(input)" in (out.error_repr or "")


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


async def test_log_goes_to_stderr_not_the_frame_stream() -> None:
    out = await _run("async def main(input):\n    log('diagnostic')\n    return 7")
    assert out.kind == "returned"
    assert out.value == 7  # stdout frame stream parsed cleanly despite the log()
    assert "diagnostic" in out.stderr


async def test_print_is_unavailable_so_author_cannot_corrupt_stdout() -> None:
    out = await _run("async def main(input):\n    print('x')\n    return 1")
    assert out.kind == "raised"  # print is not in SAFE_BUILTINS
