#!/usr/bin/env python3
"""WaM fusion MEASURING harness — Phase A: does heterogeneous R1 beat the matched-fan-in baseline?

Builds on the foundation harness (R0 ≈ native, proven in #1650). The foundation established
the binding is transparent — so a delta measured here is the RECIPE, not the plumbing. This
harness answers the first REAL fusion question, on the REASONING/CORRECTNESS tier with
programmatically-checkable answers (no LLM-judge — that would itself be a fusion node).

THE THREE CONDITIONS (each bound as model:"workflow:<id>", measured identically):
  1. best-single        — one call, the best single model (R0-shaped). No fan-in.
  2. self-fusion        — the R1 chain (Thinker→Worker→Verifier, retry-cap 2) with the best
                          single model in ALL THREE roles. THE MATCHED-FAN-IN BASELINE:
                          identical structure/compute, homogeneous substrate.
  3. heterogeneous-R1   — the same chain with heterogeneous models; Verifier on a DIFFERENT
                          substrate than the Worker (the substrate-different-verdict invariant).

THE HONEST QUESTION (red-team-enforced): does (3) beat (2) at EQUAL fan-in, past the MDE,
with FWER controlled? An (3)-beats-(1) win is confounded by fan-in and is NOT the verdict.
A "fusion wins" headline without the (2) control is exactly the fraud the red-team killed.

PRE-DECLARED (before the data):
  * Primary comparison: heterogeneous-R1 vs self-fusion (paired McNemar on per-item correctness).
  * Secondary (reported, Holm-corrected with the primary): R1 vs best-single; self-fusion vs best-single.
  * MDE = +8 percentage points accuracy (the smallest effect we'd call worth fusion's Nx cost).
  * alpha = 0.05, Holm-Bonferroni across the 3 comparisons.
  * Per-item deltas + discordant cells reported (an aggregate win hiding regressions = routing bug).
  * True $/task from the run-level call_llm_cost_microusd meter (fusion's Nx cost is part of the verdict).

The BEST single model is chosen by a CALIBRATION pass over the pool (not assumed) unless pinned.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from aios_client import AiosClient
from recipes import build_best_single_script, build_r1_script
from scoring import score
from stats_fusion import holm_correction, mcnemar, mde_note

HERE = Path(__file__).parent

# ── pool (reachable on prod, verified 2026-06-30) ────────────────────────────
# substrate label is what makes the verifier "different substrate than worker" checkable.
POOL = {
    "opus": {"model": "anthropic/claude-opus-4-8", "substrate": "anthropic"},  # ant-proxy (subsidized)
    "sonnet": {"model": "anthropic/claude-sonnet-4-5", "substrate": "anthropic"},  # ant-proxy (subsidized)
    "gpt": {"model": "openrouter/openai/gpt-5.5", "substrate": "openai"},  # topped-up; formats cleanly
    "kimi": {"model": "openrouter/moonshotai/kimi-k2.6", "substrate": "moonshot"},
    "glm": {"model": "openrouter/z-ai/glm-5.2", "substrate": "zai"},  # panel-only, NEVER verdict
}
GLM_VERDICT_BANNED = "glm"  # GLM may be a panel member but never the Verifier (recipe constraint)

# Heterogeneous R1 role assignment (Thinker A, Worker B, Verifier C). The two RELIABLE,
# clean-formatting, subsidized substrates: Worker = GPT-5.5 (openai, topped-up OpenRouter)
# and Verifier = Opus (anthropic, ant-proxy) — the load-bearing substrate-different pair.
# Verifier substrate (anthropic) != Worker substrate (openai) is asserted at provision.
# Thinker = Sonnet (anthropic, ant-proxy) — a fast/cheap approach-sketch; its substrate
# equals the Verifier's but that's fine (the invariant is Verifier != Worker).
R1_ROLES = {"A": "sonnet", "B": "gpt", "C": "opus"}  # thinker=anthropic, worker=openai, verifier=anthropic

MDE_PP = 8.0
ALPHA = 0.05

ANSWER_INSTR = (
    "Solve the problem. Think step by step if needed, then end your reply with a final line "
    "of EXACTLY the form: ANSWER: <value>  (just the value, no extra words)."
)
SYSTEM = "You are a careful problem-solver. Show brief reasoning, then the ANSWER: line."


@dataclass
class CondResult:
    """One condition's per-item outcomes."""

    name: str
    workflow_id: str
    bind: str
    correct: list[bool] = field(default_factory=list)
    extracted: list[str | None] = field(default_factory=list)
    cost_uusd: list[int | None] = field(default_factory=list)  # per-item total chain cost
    texts: list[str | None] = field(default_factory=list)
    run_ids: list[str | None] = field(default_factory=list)
    session_rebill_tokens: list[int | None] = field(default_factory=list)

    def accuracy(self) -> float:
        return sum(self.correct) / len(self.correct) if self.correct else 0.0

    def mean_cost_uusd(self) -> float:
        vals = [c for c in self.cost_uusd if c is not None]
        return sum(vals) / len(vals) if vals else 0.0


def _provision_condition(
    client: AiosClient, name: str, script: str, env_id: str, tag: str
) -> tuple[str, str, str]:
    """Register a recipe workflow (idempotent, version-pinned) + a bound agent. Returns
    (workflow_id, bind_string, agent_id)."""
    wf = client.ensure_workflow(f"fusion-{name}", script, f"WaM fusion recipe: {name}")
    wf = wf.get("data", wf) if isinstance(wf, dict) else wf
    wf_id, wf_ver = wf["id"], wf.get("version")
    bind = f"workflow:{wf_id}@{wf_ver}" if wf_ver is not None else f"workflow:{wf_id}"
    agent_id = client.create_agent(f"fusion-{name}-{tag}", bind, SYSTEM, {"temperature": 0})["id"]
    return wf_id, bind, agent_id


def _run_item(client: AiosClient, agent_id: str, env_id: str, prompt: str, timeout_s: float):
    """Run one task through a bound agent; return (text, total_chain_cost_uusd, run_id, session_rebill)."""
    full_prompt = f"{prompt}\n\n{ANSWER_INSTR}"
    sess = client.create_session(agent_id, env_id, full_prompt)
    sid = sess["id"]
    asst = client.wait_for_assistant(sid, timeout_s=timeout_s)
    text = asst.get("content") if asst else None
    inner_run_id = client.park_run_id(sid)
    cost = None
    if inner_run_id:
        run = client.get_run(inner_run_id)
        cost = run.get("call_llm_cost_microusd")
    # no-double-charge probe: the bound session's own token meter should stay ~0.
    sess_usage = client.session_usage(sid)
    rebill = sess_usage.get("output_tokens") if isinstance(sess_usage, dict) else None
    return text, cost, inner_run_id, rebill


def _measure(
    client: AiosClient,
    cond: CondResult,
    agent_id: str,
    env_id: str,
    items: list[dict],
    timeout_s: float,
    throttle_s: float,
    label: str,
) -> None:
    for i, item in enumerate(items, 1):
        try:
            text, cost, run_id, rebill = _run_item(
                client, agent_id, env_id, item["prompt"], timeout_s
            )
            ok, ext = score(text, item["answer"])
        except Exception as exc:  # one item failing must not void the condition
            text, cost, run_id, rebill, ok, ext = None, None, None, None, False, f"ERR:{exc}"[:80]
        cond.correct.append(ok)
        cond.extracted.append(ext)
        cond.cost_uusd.append(cost)
        cond.texts.append(text)
        cond.run_ids.append(run_id)
        cond.session_rebill_tokens.append(rebill)
        mark = "✓" if ok else "✗"
        print(
            f"  [{label} {i:>2}/{len(items)}] {mark} {item['id']} "
            f"ext={str(ext)[:18]!r} want={item['answer']!r} cost_uusd={cost}"
        )
        time.sleep(throttle_s)


def _calibrate_best_single(
    client: AiosClient,
    env_id: str,
    items: list[dict],
    timeout_s: float,
    throttle_s: float,
    tag: str,
) -> str:
    """Pick the best single model by a one-call accuracy pass over the pool (excluding GLM
    from contention as a *primary* since it's panel-only, but it MAY be measured). Returns
    the pool key of the most accurate model."""
    print("# CALIBRATION: best single model over the pool (one-call accuracy) ...")
    scores: dict[str, float] = {}
    for key, spec in POOL.items():
        script = build_best_single_script(spec["model"])
        _, _, agent = _provision_condition(client, f"cal-{key}", script, env_id, tag)
        cond = CondResult(name=f"cal-{key}", workflow_id="", bind="")
        _measure(client, cond, agent, env_id, items, timeout_s, throttle_s, f"cal-{key}")
        scores[key] = cond.accuracy()
        print(f"#   {key} ({spec['model']}): acc={scores[key]:.3f}")
    # Best single: highest accuracy. GLM is allowed to WIN calibration (it's only banned as
    # the Verifier verdict), but if it ties, prefer a non-GLM for a cleaner baseline.
    best = max(scores, key=lambda k: (scores[k], k != GLM_VERDICT_BANNED))
    print(f"# BEST SINGLE = {best} ({POOL[best]['model']}), acc={scores[best]:.3f}")
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description="WaM fusion measuring harness (Phase A).")
    ap.add_argument("--n", type=int, default=30, help="number of items (<= corpus)")
    ap.add_argument("--timeout-s", type=float, default=180.0)
    ap.add_argument("--throttle-s", type=float, default=1.5)
    ap.add_argument("--env-id", default=None)
    ap.add_argument(
        "--best-single", default=None, help="pin best-single pool key (skip calibration)"
    )
    ap.add_argument("--out", default=str(HERE / "fusion_results.json"))
    ap.add_argument(
        "--skip-calibration", action="store_true", help="use --best-single without a pass"
    )
    ap.add_argument(
        "--tasks",
        default="reasoning_checkable.json",
        help="task file under tasks/ (e.g. reasoning_hard.json for a headroom corpus)",
    )
    args = ap.parse_args()

    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(line_buffering=True)

    client = AiosClient()
    acct = client.whoami()
    print(f"# account {acct['id']} spend_limit_usd={acct.get('config', {}).get('spend_limit_usd')}")

    items = json.loads((HERE / "tasks" / args.tasks).read_text())["items"][: args.n]
    print(f"# {len(items)} items from {args.tasks}; MDE={MDE_PP}pp alpha={ALPHA}")
    env_id = args.env_id or client.ensure_environment("wam-eval-env")["id"]
    tag = time.strftime("%H%M%S")

    # ── pick best single model ───────────────────────────────────────────────
    if args.best_single:
        best_key = args.best_single
        if not args.skip_calibration:
            print(f"# (best-single pinned to {best_key}; calibration skipped)")
    else:
        best_key = _calibrate_best_single(
            client, env_id, items, args.timeout_s, args.throttle_s, tag
        )
    best_model = POOL[best_key]["model"]

    # ── assert the recipe's substrate invariant ──────────────────────────────
    worker_sub = POOL[R1_ROLES["B"]]["substrate"]
    verifier_sub = POOL[R1_ROLES["C"]]["substrate"]
    assert verifier_sub != worker_sub, (
        f"recipe violation: Verifier substrate ({verifier_sub}) must differ from Worker ({worker_sub})"
    )
    assert R1_ROLES["C"] != GLM_VERDICT_BANNED, "GLM may not be the Verifier (panel-only)"
    print(
        f"# heterogeneous-R1 roles: A={R1_ROLES['A']} B={R1_ROLES['B']}(worker/{worker_sub}) "
        f"C={R1_ROLES['C']}(verifier/{verifier_sub})  [substrate-different ✓]"
    )

    # ── provision the three conditions ───────────────────────────────────────
    bs_wf, bs_bind, bs_agent = _provision_condition(
        client, "best-single", build_best_single_script(best_model), env_id, tag
    )
    sf_wf, sf_bind, sf_agent = _provision_condition(
        client, "self-fusion", build_r1_script(best_model, best_model, best_model), env_id, tag
    )
    het_wf, het_bind, het_agent = _provision_condition(
        client,
        "het-r1",
        build_r1_script(
            POOL[R1_ROLES["A"]]["model"], POOL[R1_ROLES["B"]]["model"], POOL[R1_ROLES["C"]]["model"]
        ),
        env_id,
        tag,
    )
    print(f"# best-single {bs_bind}\n# self-fusion {sf_bind}\n# het-r1      {het_bind}")

    best_single = CondResult("best-single", bs_wf, bs_bind)
    self_fusion = CondResult("self-fusion", sf_wf, sf_bind)
    het_r1 = CondResult("heterogeneous-R1", het_wf, het_bind)

    print("\n# RUN best-single ...")
    _measure(client, best_single, bs_agent, env_id, items, args.timeout_s, args.throttle_s, "best")
    print("\n# RUN self-fusion (matched-fan-in baseline) ...")
    _measure(client, self_fusion, sf_agent, env_id, items, args.timeout_s, args.throttle_s, "self")
    print("\n# RUN heterogeneous-R1 ...")
    _measure(client, het_r1, het_agent, env_id, items, args.timeout_s, args.throttle_s, "hetR1")

    _report(items, best_key, best_model, best_single, self_fusion, het_r1, args.out)
    return 0


def _report(items, best_key, best_model, best_single, self_fusion, het_r1, out_path) -> None:
    print("\n" + "=" * 80)
    print("WaM FUSION PHASE A — REASONING/CORRECTNESS tier")
    print(f"best single model: {best_key} ({best_model})")
    print("=" * 80)
    for c in (best_single, self_fusion, het_r1):
        print(
            f"  {c.name:18} accuracy={c.accuracy():.3f}  mean $/task={c.mean_cost_uusd() / 1e6:.5f}"
        )

    # The PRIMARY comparison + the two secondaries. B is treatment, A baseline.
    primary = mcnemar("R1_vs_selffusion", self_fusion.correct, het_r1.correct)  # equal fan-in
    sec1 = mcnemar("R1_vs_bestsingle", best_single.correct, het_r1.correct)  # confounded by fan-in
    sec2 = mcnemar("selffusion_vs_bestsingle", best_single.correct, self_fusion.correct)

    holm = holm_correction(
        [(primary.name, primary.p_exact), (sec1.name, sec1.p_exact), (sec2.name, sec2.p_exact)],
        alpha=ALPHA,
    )

    print("\n--- PAIRED COMPARISONS (McNemar exact, Holm-corrected, bootstrap CI on Δacc) ---")
    for r in (primary, sec1, sec2):
        h = holm[r.name]
        print("\n" + r.summary())
        print(
            f"    Holm: p_raw={h['p_raw']:.4g}  p_holm={h['p_holm']:.4g}  "
            f"reject@{ALPHA}={h['reject']}"
        )

    print("\n--- MDE / power ---")
    print("  " + mde_note(primary.n, self_fusion.accuracy(), MDE_PP, ALPHA))

    # no-double-charge spot-check across all conditions (a multi-call recipe must still
    # not re-bill the outer session).
    rebills = [
        t
        for c in (best_single, self_fusion, het_r1)
        for t in c.session_rebill_tokens
        if (t or 0) > 0
    ]
    print(f"\n--- COST-METER: sessions that re-billed tokens (should be 0): {len(rebills)} ---")

    # ── THE HONEST VERDICT ───────────────────────────────────────────────────
    primary_h = holm[primary.name]
    beats = primary.delta > 0 and primary_h["reject"]
    past_mde = primary.ci_low >= (MDE_PP / 100.0)  # CI lower bound clears the MDE
    print("\n" + "=" * 80)
    print("HONEST VERDICT (heterogeneous-R1 vs self-fusion-of-best-single, EQUAL fan-in):")
    print(
        f"  Δaccuracy = {primary.delta:+.3f}  (95% CI [{primary.ci_low:+.3f}, {primary.ci_high:+.3f}])"
    )
    print(
        f"  significant after Holm@{ALPHA}? {primary_h['reject']}  (p_holm={primary_h['p_holm']:.4g})"
    )
    print(
        f"  effect past MDE ({MDE_PP:.0f}pp)? {past_mde}  (CI lower {primary.ci_low:+.3f} vs {MDE_PP / 100:.2f})"
    )
    cost_mult = (
        het_r1.mean_cost_uusd() / best_single.mean_cost_uusd()
        if best_single.mean_cost_uusd() > 0
        else float("nan")
    )
    print(f"  cost: R1 is {cost_mult:.1f}x best-single's $/task")
    if beats and past_mde:
        verdict = "R1 BEATS the matched-fan-in baseline past the MDE — fusion earns its cost."
    elif beats:
        verdict = "R1 beats self-fusion significantly but the effect is BELOW the MDE — not worth Nx cost."
    elif primary.delta > 0:
        verdict = "R1 numerically ahead of self-fusion but NOT significant after Holm — no established win."
    else:
        verdict = "R1 does NOT beat self-fusion at equal fan-in — the heterogeneity adds no measured gain here."
    print(f"  >> {verdict}")
    print("=" * 80)

    payload = {
        "tier": "reasoning_checkable",
        "n": len(items),
        "best_single_key": best_key,
        "best_single_model": best_model,
        "mde_pp": MDE_PP,
        "alpha": ALPHA,
        "r1_roles": R1_ROLES,
        "pool": POOL,
        "conditions": {
            c.name: {
                "accuracy": c.accuracy(),
                "mean_cost_usd_per_task": c.mean_cost_uusd() / 1e6,
                "bind": c.bind,
                "correct": c.correct,
                "extracted": c.extracted,
                "cost_uusd": c.cost_uusd,
                "run_ids": c.run_ids,
            }
            for c in (best_single, self_fusion, het_r1)
        },
        "comparisons": {r.name: {**asdict(r), "holm": holm[r.name]} for r in (primary, sec1, sec2)},
        "session_rebill_count": len(rebills),
        "verdict": verdict,
        "r1_beats_matched_fanin_past_mde": bool(beats and past_mde),
        "items": [{"id": it["id"], "answer": it["answer"]} for it in items],
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
