#!/usr/bin/env python3
"""WaM fusion-eval FOUNDATION harness: R0 passthrough ≈ native, by TOST.

The single gating deliverable of the workflows-as-models (WaM) fusion eval. It
proves the binding plumbing is transparent BEFORE any fusion recipe is measured:
if R0 (a one-node workflow ``resp = await call_llm(req); return resp``, bound as
``model:"workflow:<id>"``) does not round-trip equivalently to the SAME model
bound natively, every downstream fusion delta is uninterpretable.

What it does (all against the LIVE API, on an isolated budget-capped account):
  1. Register R0 (a passthrough workflow wrapping BASE_MODEL).
  2. Create two agents: ``native`` (model=BASE_MODEL) and ``r0`` (model=workflow:<id>).
  3. For each task: run it through both agents as single-turn sessions at temp=0,
     collect the assistant turn + the run-level call_llm cost.
  4. Assert the cost meter does not double-charge (inner-run meter vs session usage).
  5. Run a paired TOST equivalence test on pre-declared continuous metrics.

This is a BLACK-BOX probe: it talks to the deployed API exactly as a fusion-recipe
author would. It extends the #1221/#1282 blind-bakeoff *pattern* (isolated dispatch,
paired runs) but is purpose-built for a statistically-grounded equivalence verdict
rather than a subjective A/B/C ranking — the scoreboard the recipes need.

------------------------------------------------------------------------------
PRE-DECLARED EQUIVALENCE MARGINS (stated up front, before seeing the data):

  Primary metric — EXACT-MATCH RATE on normalized output text.
    At temperature=0 a transparent binding must reproduce native's output
    token-for-token. We pre-declare: exact-match rate >= 0.90 is the pass bar
    for the binding being behaviorally transparent. (Reported, not TOST'd — it is
    a rate, not a paired continuous metric.)

  TOST metric 1 — per-task OUTPUT LENGTH difference (chars), R0 - native.
    margin M_len = 8 characters. A transparent binding should not systematically
    lengthen or shorten the answer; ±8 chars is well inside one short word and is
    the noise floor of temp=0 re-emission.

  TOST metric 2 — per-task OUTPUT-TOKEN difference, R0 - native.
    margin M_tok = 2 tokens. The native session and the R0 inner run both report
    usage.output_tokens (same units), so this is the apples-to-apples usage metric:
    the binding must not change what the inference emits (same model, same prompt)
    beyond tokenizer/usage jitter. (A separate NO-DOUBLE-CHARGE assertion checks
    that R0's spend is metered once on the inner run and the outer session is not
    re-billed — see _report.)

  alpha = 0.05 for each TOST (so each equivalence claim is at the 95% level).

Equivalence is POSITIVELY ESTABLISHED for a metric iff its TOST concludes
equivalent (p_TOST < alpha, CI within +/-margin). The foundation passes iff BOTH
TOSTs conclude equivalent AND exact-match rate >= 0.90 AND no-double-charge holds.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from aios_client import AiosClient
from tost import paired_tost

HERE = Path(__file__).parent
BASE_MODEL = "anthropic/claude-sonnet-4-5"  # temp=0-respecting; see README for the opus-4-8 caveat.
TEMP0_PARAMS = {"temperature": 0}

# Pre-declared margins (see module docstring).
M_LEN = 8.0  # chars (output-text length difference)
M_TOK = 2.0  # output-token difference (apples-to-apples usage metric)
M_COST = 200.0  # micro-USD (reported; native session exposes tokens not cost)
EXACT_MATCH_BAR = 0.90
ALPHA = 0.05

# The R0 passthrough workflow body. ``__BASE_MODEL__`` is substituted at provision
# time (not an f-string: the script body itself contains ``{`` / ``}``).
R0_SCRIPT_TEMPLATE = """async def main(input):
    # R0 passthrough: forward the bound LlmRequest to call_llm and return the raw
    # assistant turn verbatim. This is the binding's identity element — if R0 is
    # not equivalent to native, the plumbing (not the recipe) is the variable.
    resp = await call_llm({
        "model": "__BASE_MODEL__",
        "messages": input["messages"],
        "tools": input.get("tools"),
        "params": input.get("params"),
        "session_id": input.get("session_id"),
    })
    return resp
"""
R0_SCRIPT = R0_SCRIPT_TEMPLATE.replace("__BASE_MODEL__", BASE_MODEL)

SYSTEM = (
    "You are a terse factual assistant. Answer with the shortest correct response. No preamble."
)


def _normalize(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".")


@dataclass
class PairResult:
    task: str
    native_text: str | None
    r0_text: str | None
    native_out_tokens: int | None  # native session usage.output_tokens
    r0_out_tokens: int | None  # R0 inner-run output.usage.output_tokens
    r0_inner_cost_uusd: int | None  # inner-run call_llm meter (charged once)
    r0_session_out_tokens: int | None  # R0 SESSION usage (must be ~0 — no double-bill)
    r0_inner_run_id: str | None
    r0_harvest_markers: list[str]
    exact_match: bool
    len_diff: float | None  # r0 - native, chars
    tok_diff: float | None  # r0 - native, output tokens
    note: str = ""


def _run_one(client: AiosClient, agent_id: str, env_id: str, prompt: str, timeout_s: float):
    """Run a single-turn session; return (assistant_text, session_id)."""
    sess = client.create_session(agent_id, env_id, prompt)
    sid = sess["id"]
    asst = client.wait_for_assistant(sid, timeout_s=timeout_s)
    text = asst.get("content") if asst else None
    return text, sid


def _out_tokens(usage: dict | None) -> int | None:
    if not isinstance(usage, dict):
        return None
    v = usage.get("output_tokens")
    return v if isinstance(v, int) else None


def _run_pair(
    client: AiosClient, native_agent: str, r0_agent: str, env_id: str, task: str, timeout_s: float
) -> PairResult:
    # Native arm — direct provider call; usage billed to the session meter.
    native_text, native_sid = _run_one(client, native_agent, env_id, task, timeout_s)
    native_out = _out_tokens(client.session_usage(native_sid))

    # R0 arm — same prompt, workflow-bound (park → inner run → harvest).
    r0_text, r0_sid = _run_one(client, r0_agent, env_id, task, timeout_s)
    inner_run_id = client.park_run_id(r0_sid)
    markers = client.harvest_markers(r0_sid)

    # R0's spend + token usage live on the INNER run (charged once at the call_llm
    # inference site). The R0 SESSION's own usage meter must stay ~0 for this turn —
    # the harvested turn is NOT re-billed to the session (no double-charge).
    r0_out = r0_inner_cost = None
    if inner_run_id:
        run = client.get_run(inner_run_id)
        r0_inner_cost = run.get("call_llm_cost_microusd")
        r0_out = _out_tokens((run.get("output") or {}).get("usage"))
    r0_session_out = _out_tokens(client.session_usage(r0_sid))

    em = _normalize(native_text) == _normalize(r0_text) and native_text is not None
    len_diff = (
        float(len(r0_text or "") - len(native_text or ""))
        if (native_text is not None and r0_text is not None)
        else None
    )
    tok_diff = (
        float(r0_out - native_out) if (r0_out is not None and native_out is not None) else None
    )
    return PairResult(
        task=task,
        native_text=native_text,
        r0_text=r0_text,
        native_out_tokens=native_out,
        r0_out_tokens=r0_out,
        r0_inner_cost_uusd=r0_inner_cost,
        r0_session_out_tokens=r0_session_out,
        r0_inner_run_id=inner_run_id,
        r0_harvest_markers=markers,
        exact_match=em,
        len_diff=len_diff,
        tok_diff=tok_diff,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="WaM R0-vs-native fusion-eval foundation (TOST).")
    ap.add_argument("--n", type=int, default=40, help="number of tasks (<= corpus size)")
    ap.add_argument("--timeout-s", type=float, default=90.0, help="per-turn assistant wait")
    ap.add_argument(
        "--throttle-s", type=float, default=1.0, help="sleep between pairs (pool headroom)"
    )
    ap.add_argument("--r0-workflow-id", default=None, help="reuse an existing R0 workflow id")
    ap.add_argument("--native-agent-id", default=None)
    ap.add_argument("--r0-agent-id", default=None)
    ap.add_argument("--env-id", default=None, help="reuse an existing environment id")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    args = ap.parse_args()

    # Line-buffer stdout so progress streams live (under nohup / a monitor pipe).
    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(line_buffering=True)

    client = AiosClient()
    acct = client.whoami()
    print(
        f"# account: {acct['id']}  spend_limit_usd={acct.get('config', {}).get('spend_limit_usd')}"
    )

    tasks = json.loads((HERE / "tasks" / "r0_tasks.json").read_text())["tasks"][: args.n]
    print(f"# {len(tasks)} tasks, base model {BASE_MODEL}, temp=0")

    # ── provision (idempotent via flags) ─────────────────────────────────────
    env_id = args.env_id or client.ensure_environment("wam-eval-env")["id"]
    if args.r0_workflow_id:
        wf_id, wf_ver = args.r0_workflow_id, None
    else:
        wf = client.ensure_workflow(
            "r0-passthrough", R0_SCRIPT, "WaM R0 passthrough for fusion-eval"
        )
        wf = wf.get("data", wf) if isinstance(wf, dict) else wf
        wf_id, wf_ver = wf["id"], wf.get("version")
    # Pin the exact registered version so a later workflow edit can't silently
    # re-score the old body (re-registration discipline; see ensure_workflow).
    wf_bind = f"workflow:{wf_id}@{wf_ver}" if wf_ver is not None else f"workflow:{wf_id}"
    print(f"# R0 workflow: {wf_id} v{wf_ver}  (bind as {wf_bind})")
    # SYMMETRIC temp=0: a session turn builds its LlmRequest with
    # ``params=agent.litellm_extra`` (harness/loop.py), and the WaM park forwards
    # that SAME ``request.params`` into R0's call_llm input. Setting
    # litellm_extra={"temperature": 0} on BOTH agents therefore pins temp=0
    # identically on both arms — native passes it straight to the provider, R0
    # forwards it through the binding. This makes the comparison fair AND
    # near-deterministic, so any residual R0-vs-native delta is the binding, not
    # sampling. (See README "temperature note" re: opus-4-8 deprecating it.)
    tag = time.strftime("%H%M%S")  # unique agent names so re-runs don't 409
    native_agent = (
        args.native_agent_id
        or client.create_agent(f"eval-native-{tag}", BASE_MODEL, SYSTEM, TEMP0_PARAMS)["id"]
    )
    r0_agent = (
        args.r0_agent_id
        or client.create_agent(f"eval-r0-{tag}", wf_bind, SYSTEM, TEMP0_PARAMS)["id"]
    )
    print(f"# native agent: {native_agent}\n# r0 agent:     {r0_agent}")

    # ── run paired sample ────────────────────────────────────────────────────
    results: list[PairResult] = []
    for i, task in enumerate(tasks, 1):
        try:
            pr = _run_pair(client, native_agent, r0_agent, env_id, task, args.timeout_s)
        except Exception as exc:  # a single pair failing must not void the run
            pr = PairResult(
                task,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                [],
                False,
                None,
                None,
                note=str(exc)[:160],
            )
        results.append(pr)
        em = "✓" if pr.exact_match else "✗"
        print(
            f"[{i:>2}/{len(tasks)}] {em} native={pr.native_text!r:.36} "
            f"r0={pr.r0_text!r:.36} len_d={pr.len_diff} tok_d={pr.tok_diff} "
            f"r0_cost_uusd={pr.r0_inner_cost_uusd}"
        )
        time.sleep(args.throttle_s)

    _report(results, args.out)
    return 0


def _report(results: list[PairResult], out_path: str) -> None:
    paired = [r for r in results if r.native_text is not None and r0_ok(r)]
    n_paired = len(paired)
    em_rate = sum(1 for r in paired if r.exact_match) / n_paired if n_paired else 0.0

    len_diffs = [r.len_diff for r in paired if r.len_diff is not None]
    tok_diffs = [r.tok_diff for r in paired if r.tok_diff is not None]

    print("\n" + "=" * 78)
    print(f"PAIRED SAMPLE: {n_paired} usable pairs (of {len(results)} attempted)")
    print(f"EXACT-MATCH RATE (normalized output): {em_rate:.3f}  (bar >= {EXACT_MATCH_BAR})")

    # NO-DOUBLE-CHARGE invariant. The bound workflow meters its OWN call_llm spend
    # on the inner run; the binding boundary carries usage/cost through for the span
    # but does NOT re-charge at harvest. We assert two things empirically:
    #   (a) every R0 inner run charged a positive call_llm_cost_microusd (it paid once), and
    #   (b) the R0 SESSION's own output-token meter stayed ~0 (the harvested turn was
    #       not separately re-billed to the session — i.e. it is not charged twice).
    inner_charged = [r for r in paired if (r.r0_inner_cost_uusd or 0) > 0]
    session_rebill = [r for r in paired if (r.r0_session_out_tokens or 0) > 0]
    no_double_charge = len(inner_charged) == n_paired and len(session_rebill) == 0
    print("\nCOST-METER (no double-charge):")
    print(f"  R0 inner runs that charged once (>0 uusd): {len(inner_charged)}/{n_paired}")
    print(f"  R0 sessions that ALSO billed tokens (should be 0): {len(session_rebill)}")
    print(f"  -> no-double-charge: {'HOLDS' if no_double_charge else 'VIOLATED'}")
    for r in paired[:4]:
        print(
            f"    {r.task[:30]!r:32} inner_cost_uusd={r.r0_inner_cost_uusd} "
            f"inner_out_tok={r.r0_out_tokens} session_out_tok={r.r0_session_out_tokens}"
        )

    tost_len = paired_tost(len_diffs, margin=M_LEN, alpha=ALPHA) if len(len_diffs) >= 2 else None
    tost_tok = paired_tost(tok_diffs, margin=M_TOK, alpha=ALPHA) if len(tok_diffs) >= 2 else None

    print(f"\n--- TOST metric 1: output-length difference (chars), margin +/-{M_LEN:g} ---")
    print(tost_len.summary() if tost_len else "  insufficient paired length data")
    print(f"\n--- TOST metric 2: output-token difference, margin +/-{M_TOK:g} ---")
    print(tost_tok.summary() if tost_tok else "  insufficient paired token data")

    foundation_pass = (
        em_rate >= EXACT_MATCH_BAR
        and tost_len is not None
        and tost_len.equivalent
        and tost_tok is not None
        and tost_tok.equivalent
        and no_double_charge
    )
    print("\n" + "=" * 78)
    verdict = (
        "PASS — R0 ≈ native (equivalence positively established)"
        if foundation_pass
        else "NOT ESTABLISHED"
    )
    print(f"FOUNDATION VERDICT: {verdict}")
    print("=" * 78)

    payload = {
        "base_model": BASE_MODEL,
        "n_attempted": len(results),
        "n_paired": n_paired,
        "exact_match_rate": em_rate,
        "exact_match_bar": EXACT_MATCH_BAR,
        "margins": {"len_chars": M_LEN, "out_tokens": M_TOK, "alpha": ALPHA},
        "no_double_charge": no_double_charge,
        "inner_charged_count": len(inner_charged),
        "session_rebill_count": len(session_rebill),
        "tost_len": asdict(tost_len) if tost_len else None,
        "tost_tok": asdict(tost_tok) if tost_tok else None,
        "foundation_pass": foundation_pass,
        "pairs": [asdict(r) for r in results],
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")


def r0_ok(r: PairResult) -> bool:
    return r.r0_text is not None and "model_workflow_harvest" in r.r0_harvest_markers


if __name__ == "__main__":
    sys.exit(main())
