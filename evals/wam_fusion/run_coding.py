#!/usr/bin/env python3
"""WaM fusion measuring harness — Phase B: CODING tier (pass@1 on real aios PRs).

The headroom tier the reasoning tier (Phase A) lacked: real merged-PR bugs where the
best single model genuinely FAILS some fraction, scored by a held-out test suite the
model never sees. Same matched-fan-in design, recipe (R1), stats (McNemar/Holm/
bootstrap), calibration, provisioning, and no-double-charge assertion as Phase A — only
the prompt + scorer + corpus change (coding, not checkable-answer).

THE HONEST QUESTION (unchanged): does heterogeneous-R1 beat self-fusion-of-best-single
at EQUAL fan-in, past the pre-declared MDE, FWER-controlled? Now on a tier with headroom.

Conditions (each bound as model:"workflow:<id>", measured identically):
  best-single / self-fusion-of-best-single / heterogeneous-R1.

Scoring: clone aios at the item's base, apply the candidate's source, run ONLY the
held-out test (PYTHONPATH forces the patched tree), pass@1 = green. A non-applying patch
= fail; an env that can't stand up = SKIP (excluded from the paired sample, logged).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from aios_client import AiosClient
from coding_prompt import CODING_SYSTEM, build_prompt, extract_sources
from coding_scorer import score_candidate, source_at_base
from recipes import build_best_single_script, build_r1_script
from run_fusion import GLM_VERDICT_BANNED, POOL, R1_ROLES
from stats_fusion import holm_correction, mcnemar, mde_note

HERE = Path(__file__).parent
MDE_PP = 8.0
ALPHA = 0.05


class Checkpoint:
    """Append-only JSON-lines checkpoint of per-(condition,item) results.

    Makes the whole measurement RESUMABLE: each scored cell is flushed to disk
    immediately, and a re-invocation loads what's done and skips it. This is what
    lets a run that exceeds one foreground window (or hits a mid-run blip) be driven
    to completion by re-running the SAME command — it continues, never restarts.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._cells: dict[tuple[str, str], dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                self._cells[(r["cond"], r["item"])] = r
        print(f"# checkpoint {path.name}: {len(self._cells)} cells already done")

    def get(self, cond: str, item: str) -> dict | None:
        return self._cells.get((cond, item))

    def put(self, cond, item, passed, skipped, cost, detail, run_id, rebill) -> None:
        rec = {
            "cond": cond, "item": item, "passed": passed, "skipped": skipped,
            "cost": cost, "detail": detail, "run_id": run_id, "rebill": rebill,
        }
        self._cells[(cond, item)] = rec
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")


@dataclass
class CodingCond:
    name: str
    bind: str
    correct: list[bool] = field(default_factory=list)
    skipped: list[bool] = field(default_factory=list)  # env-broke items (excluded)
    cost_uusd: list[int | None] = field(default_factory=list)
    detail: list[str] = field(default_factory=list)
    run_ids: list[str | None] = field(default_factory=list)
    session_rebill: list[int | None] = field(default_factory=list)

    def scored_idx(self) -> list[int]:
        return [i for i, s in enumerate(self.skipped) if not s]

    def accuracy(self) -> float:
        idx = self.scored_idx()
        return sum(self.correct[i] for i in idx) / len(idx) if idx else 0.0

    def mean_cost_uusd(self) -> float:
        v = [c for c in self.cost_uusd if c is not None]
        return sum(v) / len(v) if v else 0.0


def _provision(client: AiosClient, name: str, script: str, tag: str) -> str:
    """Register a recipe workflow (version-pinned) + a CODING-system agent; return agent id."""
    wf = client.ensure_workflow(f"coding-{name}", script, f"WaM coding recipe: {name}")
    wf = wf.get("data", wf) if isinstance(wf, dict) else wf
    bind = (
        f"workflow:{wf['id']}@{wf.get('version')}" if wf.get("version") else f"workflow:{wf['id']}"
    )
    agent = client.create_agent(f"coding-{name}-{tag}", bind, CODING_SYSTEM, {"temperature": 0})[
        "id"
    ]
    return agent, bind


def _run_and_score(
    client: AiosClient,
    cond: CodingCond,
    agent_id: str,
    env_id: str,
    repo: str,
    items: list[dict],
    timeout_s: float,
    test_timeout_s: int,
    throttle_s: float,
    label: str,
    ckpt: "Checkpoint | None" = None,
) -> None:
    for i, item in enumerate(items, 1):
        # RESUME: if this (condition,item) cell is already checkpointed, replay it
        # (no model call) so a re-invocation continues where the last one stopped —
        # the run survives the shell's per-call time cap AND any mid-run crash/blip.
        cached = ckpt.get(cond.name, item["id"]) if ckpt else None
        if cached is not None:
            cond.correct.append(cached["passed"])
            cond.skipped.append(cached["skipped"])
            cond.cost_uusd.append(cached["cost"])
            cond.detail.append(cached["detail"])
            cond.run_ids.append(cached["run_id"])
            cond.session_rebill.append(cached["rebill"])
            m = "PASS" if cached["passed"] else ("SKIP" if cached["skipped"] else "fail")
            print(f"  [{label} {i:>2}/{len(items)}] {m} {item['id']} (cached)")
            continue
        passed = skipped = False
        cost = run_id = rebill = None
        detail = ""
        try:
            base_src = source_at_base(repo, item["base_parent_sha"], item["src_files"])
            prompt = build_prompt(item["task"], base_src)
            # One reasonable retry on a TRANSIENT inference error (empty text with
            # cost==0 == the provider call errored/500'd, e.g. an AnthropicException
            # blip — NOT a wrong answer). A blip must not score as a model failure and
            # bias the condition down; after one retry it becomes an honest env-skip.
            text = cost = run_id = None
            for _attempt in range(2):
                sess = client.create_session(agent_id, env_id, prompt)
                sid = sess["id"]
                asst = client.wait_for_assistant(sid, timeout_s=timeout_s)
                text = asst.get("content") if asst else None
                run_id = client.park_run_id(sid)
                cost = client.get_run(run_id).get("call_llm_cost_microusd") if run_id else None
                transient = (not text) and (not cost)  # empty answer AND nothing charged
                if not transient:
                    break
                detail = "transient inference error (empty+cost0)"
            su = client.session_usage(sid)
            rebill = su.get("output_tokens") if isinstance(su, dict) else None
            if (not text) and (not cost):
                # Still a transient error after the retry -> env-skip (exclude, log).
                skipped, detail = True, "SKIP: transient inference error (empty+cost0 x2)"
            else:
                cand = extract_sources(text, item["src_files"])
                out = score_candidate(
                    item,
                    repo,
                    cand,
                    venv_python=os.environ.get("AIOS_VENV_PYTHON"),
                    timeout_s=test_timeout_s,
                )
                passed, skipped, detail = out.passed, out.skipped, out.detail
        except Exception as exc:  # an item infra hiccup => SKIP, never wedge the run
            skipped, detail = True, f"harness:{type(exc).__name__}:{exc}"[:90]
        cond.correct.append(passed)
        cond.skipped.append(skipped)
        cond.cost_uusd.append(cost)
        cond.detail.append(detail)
        cond.run_ids.append(run_id)
        cond.session_rebill.append(rebill)
        if ckpt is not None:
            ckpt.put(cond.name, item["id"], passed, skipped, cost, detail, run_id, rebill)
        mark = "PASS" if passed else ("SKIP" if skipped else "fail")
        print(
            f"  [{label} {i:>2}/{len(items)}] {mark} {item['id']} ({item['headroom']}) cost_uusd={cost} | {detail[:46]}"
        )
        time.sleep(throttle_s)


def _calibrate(client, env_id, repo, items, timeout_s, test_timeout_s, throttle_s, tag, cal_keys=None, ckpt=None) -> str:
    keys = cal_keys or list(POOL)
    print(f"# CALIBRATION: best single model over {keys} (coding pass@1) ...")
    scores = {}
    for key in keys:
        spec = POOL[key]
        # Deterministic agent name (no time tag) so a resumed run reuses the same
        # workflow/agent rather than creating a duplicate each restart.
        agent, _ = _provision(client, f"cal-{key}", build_best_single_script(spec["model"]), "fixed")
        cond = CodingCond(f"cal-{key}", "")
        _run_and_score(
            client, cond, agent, env_id, repo, items,
            timeout_s, test_timeout_s, throttle_s, f"cal-{key}", ckpt=ckpt,
        )
        scores[key] = cond.accuracy()
        print(f"#   {key} ({spec['model']}): pass@1={scores[key]:.3f}")
    best = max(scores, key=lambda k: (scores[k], k != GLM_VERDICT_BANNED))
    print(f"# BEST SINGLE = {best} ({POOL[best]['model']}), pass@1={scores[best]:.3f}")
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description="WaM fusion CODING measuring harness (Phase B).")
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--tasks", default="coding_aios.json")
    ap.add_argument(
        "--single-file-only",
        action="store_true",
        help="restrict to 1-src-file items (clean pass@1)",
    )
    ap.add_argument("--timeout-s", type=float, default=300.0, help="per-turn assistant wait")
    ap.add_argument("--test-timeout-s", type=int, default=120, help="per-item held-out test cap")
    ap.add_argument("--throttle-s", type=float, default=1.0)
    ap.add_argument("--env-id", default=None)
    ap.add_argument("--best-single", default=None)
    ap.add_argument("--skip-calibration", action="store_true")
    ap.add_argument("--cal-pool", default=None, help="comma-separated POOL keys to calibrate over (default: all)")
    ap.add_argument("--out", default=str(HERE / "coding_results.json"))
    ap.add_argument("--checkpoint", default=str(HERE / "coding_checkpoint.jsonl"), help="resumable per-cell checkpoint")
    args = ap.parse_args()

    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(line_buffering=True)

    client = AiosClient()
    acct = client.whoami()
    print(f"# account {acct['id']} spend_limit_usd={acct.get('config', {}).get('spend_limit_usd')}")

    corpus = json.loads((HERE / "tasks" / args.tasks).read_text())
    repo = corpus["repo"]
    items = corpus["items"]
    if args.single_file_only:
        items = [it for it in items if len(it["src_files"]) == 1]
    items = items[: args.n]
    print(
        f"# {len(items)} coding items from {args.tasks} (single-file-only={args.single_file_only}); MDE={MDE_PP}pp"
    )
    env_id = args.env_id or client.ensure_environment("wam-eval-env")["id"]
    # Fixed tag so a resumed run reuses the same agents/workflows (no dupes per restart).
    tag = "fixed"
    ckpt = Checkpoint(Path(args.checkpoint))

    cal_keys = args.cal_pool.split(",") if args.cal_pool else None
    if args.best_single and args.skip_calibration:
        best_key = args.best_single  # pinned, no calibration
    else:
        best_key = _calibrate(
            client, env_id, repo, items, args.timeout_s, args.test_timeout_s,
            args.throttle_s, tag, cal_keys=cal_keys, ckpt=ckpt,
        )
    best_model = POOL[best_key]["model"]

    worker_sub, verifier_sub = POOL[R1_ROLES["B"]]["substrate"], POOL[R1_ROLES["C"]]["substrate"]
    assert verifier_sub != worker_sub, "Verifier substrate must differ from Worker"
    assert R1_ROLES["C"] != GLM_VERDICT_BANNED, "GLM may not be the Verifier"
    print(
        f"# het-R1 roles: A={R1_ROLES['A']} B={R1_ROLES['B']}(worker/{worker_sub}) C={R1_ROLES['C']}(verifier/{verifier_sub}) [substrate-different]"
    )

    bs_agent, bs_bind = _provision(client, "best-single", build_best_single_script(best_model), tag)
    sf_agent, sf_bind = _provision(
        client, "self-fusion", build_r1_script(best_model, best_model, best_model), tag
    )
    het_agent, het_bind = _provision(
        client,
        "het-r1",
        build_r1_script(
            POOL[R1_ROLES["A"]]["model"], POOL[R1_ROLES["B"]]["model"], POOL[R1_ROLES["C"]]["model"]
        ),
        tag,
    )
    print(f"# best-single {bs_bind}\n# self-fusion {sf_bind}\n# het-r1 {het_bind}")

    best_single = CodingCond("best-single", bs_bind)
    self_fusion = CodingCond("self-fusion", sf_bind)
    het_r1 = CodingCond("heterogeneous-R1", het_bind)

    print("\n# RUN best-single ...")
    _run_and_score(client, best_single, bs_agent, env_id, repo, items,
                   args.timeout_s, args.test_timeout_s, args.throttle_s, "best", ckpt=ckpt)
    print("\n# RUN self-fusion (matched-fan-in baseline) ...")
    _run_and_score(client, self_fusion, sf_agent, env_id, repo, items,
                   args.timeout_s, args.test_timeout_s, args.throttle_s, "self", ckpt=ckpt)
    print("\n# RUN heterogeneous-R1 ...")
    _run_and_score(client, het_r1, het_agent, env_id, repo, items,
                   args.timeout_s, args.test_timeout_s, args.throttle_s, "hetR1", ckpt=ckpt)

    _report(items, best_key, best_model, best_single, self_fusion, het_r1, args.out)
    return 0


def _paired(a: CodingCond, b: CodingCond) -> tuple[list[bool], list[bool]]:
    """Per-item correctness for items SCORED (not skipped) in BOTH conditions."""
    idx = [i for i in range(len(a.correct)) if not a.skipped[i] and not b.skipped[i]]
    return [a.correct[i] for i in idx], [b.correct[i] for i in idx]


def _report(items, best_key, best_model, best_single, self_fusion, het_r1, out_path) -> None:
    print("\n" + "=" * 80)
    print("WaM FUSION PHASE B — CODING tier (pass@1 on real aios PRs)")
    print(f"best single model: {best_key} ({best_model})")
    print("=" * 80)
    for c in (best_single, self_fusion, het_r1):
        n_scored = len(c.scored_idx())
        print(
            f"  {c.name:18} pass@1={c.accuracy():.3f} (n_scored={n_scored})  mean $/task=${c.mean_cost_uusd() / 1e6:.5f}"
        )

    # paired comparisons on items scored in both arms
    sf_c, het_c = _paired(self_fusion, het_r1)
    bs_c1, het_c1 = _paired(best_single, het_r1)
    bs_c2, sf_c2 = _paired(best_single, self_fusion)
    primary = mcnemar("R1_vs_selffusion", sf_c, het_c) if len(sf_c) >= 1 else None
    sec1 = mcnemar("R1_vs_bestsingle", bs_c1, het_c1) if len(bs_c1) >= 1 else None
    sec2 = mcnemar("selffusion_vs_bestsingle", bs_c2, sf_c2) if len(bs_c2) >= 1 else None

    comps = [c for c in (primary, sec1, sec2) if c]
    holm = holm_correction([(c.name, c.p_exact) for c in comps], alpha=ALPHA) if comps else {}

    print("\n--- PAIRED COMPARISONS (McNemar exact, Holm-corrected, bootstrap CI on Δpass@1) ---")
    for c in comps:
        h = holm[c.name]
        print("\n" + c.summary())
        print(
            f"    Holm: p_raw={h['p_raw']:.4g} p_holm={h['p_holm']:.4g} reject@{ALPHA}={h['reject']}"
        )

    if primary:
        print("\n--- MDE / power ---")
        print("  " + mde_note(primary.n, self_fusion.accuracy(), MDE_PP, ALPHA))

    rebills = [
        t for c in (best_single, self_fusion, het_r1) for t in c.session_rebill if (t or 0) > 0
    ]
    print(f"\n--- COST-METER: sessions that re-billed tokens (should be 0): {len(rebills)} ---")

    print("\n--- PER-ITEM (where fusion helped vs hurt: het-R1 vs self-fusion) ---")
    for i, it in enumerate(items):
        if self_fusion.skipped[i] or het_r1.skipped[i]:
            print(f"  {it['id']}: (skipped)")
            continue
        sf_ok, het_ok = self_fusion.correct[i], het_r1.correct[i]
        tag = (
            "= both pass"
            if sf_ok and het_ok
            else (
                "= both fail"
                if not sf_ok and not het_ok
                else ("+ R1 WIN" if het_ok else "- R1 LOSS")
            )
        )
        print(
            f"  {it['id']} ({it['headroom']}): self-fusion={'P' if sf_ok else 'f'} het-R1={'P' if het_ok else 'f'}  {tag}"
        )

    print("\n" + "=" * 80)
    if primary:
        h = holm[primary.name]
        beats = primary.delta > 0 and h["reject"]
        past_mde = primary.ci_low >= (MDE_PP / 100.0)
        cost_mult = (
            het_r1.mean_cost_uusd() / best_single.mean_cost_uusd()
            if best_single.mean_cost_uusd() > 0
            else float("nan")
        )
        print(
            "HONEST VERDICT (heterogeneous-R1 vs self-fusion-of-best-single, EQUAL fan-in, CODING):"
        )
        print(
            f"  Δpass@1 = {primary.delta:+.3f}  (95% CI [{primary.ci_low:+.3f}, {primary.ci_high:+.3f}], n={primary.n})"
        )
        print(f"  significant after Holm@{ALPHA}? {h['reject']} (p_holm={h['p_holm']:.4g})")
        print(
            f"  past MDE ({MDE_PP:.0f}pp)? {past_mde}  (CI lower {primary.ci_low:+.3f} vs {MDE_PP / 100:.2f})"
        )
        print(f"  cost: R1 is {cost_mult:.1f}x best-single's $/task")
        if beats and past_mde:
            verdict = "R1 BEATS the matched-fan-in baseline past the MDE on coding — fusion earns its cost."
        elif beats:
            verdict = (
                "R1 beats self-fusion significantly but BELOW the MDE — gain too small for Nx cost."
            )
        elif primary.delta > 0:
            verdict = "R1 numerically ahead but NOT significant after Holm — no established win at this n."
        elif primary.delta == 0:
            verdict = (
                "R1 TIES self-fusion at equal fan-in — heterogeneity adds no measured gain here."
            )
        else:
            verdict = "R1 LOSES to self-fusion at equal fan-in — heterogeneity HURT on this corpus."
        print(f"  >> {verdict}")
    else:
        verdict = "no usable paired items — corpus or scorer needs attention"
        print(f"  >> {verdict}")
    print("=" * 80)

    payload = {
        "tier": "coding_aios",
        "best_single_key": best_key,
        "best_single_model": best_model,
        "mde_pp": MDE_PP,
        "alpha": ALPHA,
        "r1_roles": R1_ROLES,
        "conditions": {
            c.name: {
                "pass_at_1": c.accuracy(),
                "n_scored": len(c.scored_idx()),
                "mean_cost_usd_per_task": c.mean_cost_uusd() / 1e6,
                "bind": c.bind,
                "correct": c.correct,
                "skipped": c.skipped,
                "detail": c.detail,
                "cost_uusd": c.cost_uusd,
                "run_ids": c.run_ids,
            }
            for c in (best_single, self_fusion, het_r1)
        },
        "comparisons": {c.name: {**asdict(c), "holm": holm.get(c.name)} for c in comps},
        "session_rebill_count": len(rebills),
        "verdict": verdict,
        "items": [{"id": it["id"], "pr": it["pr"], "headroom": it["headroom"]} for it in items],
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
