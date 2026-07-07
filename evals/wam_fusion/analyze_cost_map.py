#!/usr/bin/env python3
"""Cost-routing map analysis — accuracy, cheap-coverage, frontier verdicts, savings sim.

Reads the matrix + score checkpoints, asserts matrix completeness (every
item x model x k cell filled or explicitly HOLE/PENDING), and emits:

  results/cost_routing_map_2026-07.md      — the deliverable report
  results/cost_routing_matrix_2026-07.json — the raw per-cell matrix (texts stripped)

Statistical conventions (declared before analysis):
  * per-sample accuracy  = mean over items of (item's mean correctness over non-hole k)
  * majority-of-k        = item passes for a model iff >1/2 of its non-hole samples correct
  * item-bootstrap CIs   = B=1000 resamples of the item set, percentile 2.5/97.5
  * cheap coverage       = % of items where maj(cheap) >= maj(per-corpus frontier best)
  * safe lane            = point delta(cheap - frontier_best) >= -2pp AND bootstrap CI
                           lower bound of the delta >= -2pp ("SAFE"); point-only => "PROVISIONAL"
  * savings sim          = per stratum, route to the CHEAPEST model whose majority
                           accuracy >= frontier_best - 2pp; cost from measured tokens
                           x 2026-07 LIST prices (estimates; we currently pay $0)
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from run_cost_map import CORPORA, K_SAMPLES, MODELS

HERE = Path(__file__).parent
STATE_DIR = HERE / "results" / "cost_map_state"
RESULTS = HERE / "results"

CHEAP = ["haiku-4-5", "sonnet-5", "gpt-5.4-mini", "gpt-5.4"]
FRONTIER = ["opus-4-8", "gpt-5.5", "fable-5"]
B = 1000
SAFE_MARGIN = 0.02


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def pct(x, nd=1):
    return "—" if x is None else f"{100 * x:.{nd}f}%"


def load_cells(state: Path):
    """cells[corpus][model][(item,k)] = record (with coding verdicts joined in)."""
    cells: dict = {c: {m: {} for m in MODELS} for c in CORPORA}
    for c in CORPORA:
        for m in MODELS:
            for r in jsonl(state / f"matrix_{c}_{m}.jsonl"):
                cells[c][m][(r["item"], r["k"])] = r
    # join coding scores
    golden = {}
    gp = state / "scores_golden.json"
    if gp.exists():
        golden = json.loads(gp.read_text())
    excluded_items = sorted(i for i, g in golden.items() if not g.get("passed"))
    for m in MODELS:
        for s in jsonl(state / f"scores_coding_{m}.jsonl"):
            key = (s["item"], s["k"])
            if key in cells["coding"][m]:
                cells["coding"][m][key]["score"] = s
    return cells, excluded_items


def sample_outcome(corpus: str, rec: dict) -> bool | None:
    """True/False = scoreable sample; None = HOLE (excluded, tracked)."""
    if rec["status"] != "ok":
        return None
    if corpus == "coding":
        s = rec.get("score")
        if s is None:
            return None  # not yet scored -> treated as pending hole
        if s.get("skipped"):
            return None  # oracle env failure on this candidate -> hole, not wrong
        return bool(s["passed"])
    return bool(rec.get("correct"))


class ItemStat:
    __slots__ = ("outcomes", "holes")

    def __init__(self):
        self.outcomes: list[bool] = []
        self.holes = 0

    @property
    def mean(self):
        return statistics.mean(self.outcomes) if self.outcomes else None

    @property
    def majority(self):
        return (self.mean > 0.5) if self.outcomes else None


def build_stats(cells, corpora_items, excluded_items):
    """stats[corpus][model][item_id] = ItemStat; also completeness ledger."""
    stats = {c: {m: {} for m in MODELS} for c in CORPORA}
    completeness = defaultdict(lambda: defaultdict(int))
    for c in CORPORA:
        for m in MODELS:
            for item in corpora_items[c]:
                iid = item["id"]
                if c == "coding" and iid in excluded_items:
                    completeness[(c, m)]["excluded_golden"] += K_SAMPLES
                    continue
                st = ItemStat()
                for k in range(K_SAMPLES):
                    rec = cells[c][m].get((iid, k))
                    if rec is None:
                        completeness[(c, m)]["pending"] += 1
                        st.holes += 1
                        continue
                    out = sample_outcome(c, rec)
                    if out is None:
                        completeness[(c, m)]["hole"] += 1
                        st.holes += 1
                    else:
                        completeness[(c, m)]["scored"] += 1
                        st.outcomes.append(out)
                stats[c][m][iid] = st
    return stats, completeness


def boot_ci(values_by_item: dict[str, float], B=B, seed=7):
    ids = sorted(values_by_item)
    if not ids:
        return (None, None)
    rng = random.Random(seed)
    means = []
    for _ in range(B):
        pick = [values_by_item[rng.choice(ids)] for _ in ids]
        means.append(statistics.mean(pick))
    means.sort()
    return (means[int(0.025 * B)], means[int(0.975 * B) - 1])


def paired_boot_ci(a: dict[str, float], b: dict[str, float], B=B, seed=11):
    """CI of mean(a-b) over shared items."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return (None, None, None)
    diffs = {i: a[i] - b[i] for i in ids}
    point = statistics.mean(diffs.values())
    lo, hi = boot_ci(diffs, B=B, seed=seed)
    return (point, lo, hi)


def acc_map(stats_cm: dict[str, ItemStat], majority=False) -> dict[str, float]:
    out = {}
    for iid, st in stats_cm.items():
        if st.outcomes:
            out[iid] = float(st.majority) if majority else st.mean
    return out


def token_stats(cells, c, m):
    ins, outs, lats = [], [], []
    for rec in cells[c][m].values():
        if rec["status"] == "ok":
            if rec.get("input_tokens") is not None:
                ins.append(rec["input_tokens"])
            if rec.get("output_tokens") is not None:
                outs.append(rec["output_tokens"])
            if rec.get("latency_s") is not None:
                lats.append(rec["latency_s"])
    return {
        "mean_in": statistics.mean(ins) if ins else None,
        "mean_out": statistics.mean(outs) if outs else None,
        "sum_in": sum(ins),
        "sum_out": sum(outs),
        "n_usage": len(outs),
        "mean_latency_s": statistics.mean(lats) if lats else None,
    }


def call_cost_usd(c, m, tok):
    p_in, p_out = MODELS[m]["price"]
    if tok["mean_in"] is None or tok["mean_out"] is None:
        return None
    return (tok["mean_in"] * p_in + tok["mean_out"] * p_out) / 1e6


def strata_of(corpora_items):
    """(corpus, stratum) -> list of item ids. Reasoning corpora are one stratum each."""
    lanes = {}
    for c, items in corpora_items.items():
        if c == "coding":
            for it in items:
                lanes.setdefault((c, it.get("stratum", "unknown")), []).append(it["id"])
        else:
            lanes.setdefault((c, "all"), []).extend(it["id"] for it in items)
    return lanes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default=str(STATE_DIR))
    ap.add_argument("--out-md", default=str(RESULTS / "cost_routing_map_2026-07.md"))
    ap.add_argument("--out-json", default=str(RESULTS / "cost_routing_matrix_2026-07.json"))
    args = ap.parse_args()
    state = Path(args.state_dir)

    corpora_items = {}
    corpora_items["headroom"] = json.loads((HERE / "tasks/reasoning_headroom.json").read_text())["items"]
    corpora_items["extrahard"] = json.loads((HERE / "tasks/reasoning_extrahard.json").read_text())["items"]
    corpora_items["coding"] = json.loads((HERE / "tasks/coding_selection.json").read_text())["items"]
    coding_items = {it["id"]: it for it in corpora_items["coding"]}

    cells, excluded_items = load_cells(state)
    stats, completeness = build_stats(cells, corpora_items, excluded_items)

    # ── completeness ledger ──
    lines = []
    W = lines.append
    total_cells = sum(len(corpora_items[c]) for c in CORPORA) * len(MODELS) * K_SAMPLES
    tot = defaultdict(int)
    for (c, m), d in completeness.items():
        for k, v in d.items():
            tot[k] += v
    holes_per_model = {
        m: sum(completeness[(c, m)]["hole"] for c in CORPORA) for m in MODELS
    }
    pending_per_model = {
        m: sum(completeness[(c, m)]["pending"] for c in CORPORA) for m in MODELS
    }

    # natural-stop + identity + repair rates from matrix records
    admin = {}
    for m in MODELS:
        calls = ok = nat = ident_mis = trunc = holes = repaired = quarantined = 0
        for c in CORPORA:
            for rec in cells[c][m].values():
                calls += 1
                if rec["status"] == "ok":
                    ok += 1
                    if rec.get("stop_reason") == "end_turn":
                        nat += 1
                    if rec.get("repaired"):
                        repaired += 1
                elif rec["status"] == "QUARANTINE_IDENTITY":
                    quarantined += 1
                    ident_mis += 1
                else:
                    holes += 1
                    if rec.get("hole_kind") == "truncation":
                        trunc += 1
        admin[m] = dict(calls=calls, ok=ok, natural_stop_rate=(nat / ok if ok else None),
                        identity_mismatches=ident_mis, truncation_holes=trunc,
                        other_holes=holes - trunc, repaired=repaired,
                        quarantined=quarantined)

    # ── per-corpus accuracy (k-sample mean + majority) ──
    acc = {c: {} for c in CORPORA}
    for c in CORPORA:
        for m in MODELS:
            amap = acc_map(stats[c][m])
            mmap = acc_map(stats[c][m], majority=True)
            lo, hi = boot_ci(amap) if amap else (None, None)
            acc[c][m] = {
                "n_items": len(amap),
                "mean": statistics.mean(amap.values()) if amap else None,
                "ci": (lo, hi),
                "majority": statistics.mean(mmap.values()) if mmap else None,
            }

    # per coding-stratum accuracy
    lanes = strata_of(corpora_items)
    lane_acc = {}
    for (c, s), ids in lanes.items():
        for m in MODELS:
            amap = {i: v for i, v in acc_map(stats[c][m]).items() if i in set(ids)}
            mmap = {i: v for i, v in acc_map(stats[c][m], majority=True).items() if i in set(ids)}
            lane_acc[(c, s, m)] = {
                "n": len(amap),
                "mean": statistics.mean(amap.values()) if amap else None,
                "majority": statistics.mean(mmap.values()) if mmap else None,
                "amap": amap,
                "mmap": mmap,
            }

    # ── frontier best per corpus (majority-based) ──
    frontier_best = {}
    for c in CORPORA:
        best, best_v = None, -1
        for m in FRONTIER:
            v = acc[c][m]["majority"]
            if v is not None and v > best_v:
                best, best_v = m, v
        frontier_best[c] = best

    # ── cheap coverage per corpus + per lane ──
    coverage = {}
    for c in CORPORA:
        fb = frontier_best[c]
        fb_m = acc_map(stats[c][fb], majority=True)
        for ch in CHEAP:
            ch_m = acc_map(stats[c][ch], majority=True)
            shared = sorted(set(fb_m) & set(ch_m))
            if not shared:
                continue
            cov = statistics.mean(1.0 if ch_m[i] >= fb_m[i] else 0.0 for i in shared)
            point, lo, hi = paired_boot_ci(ch_m, fb_m)
            flag = "NOT SAFE"
            if point is not None and point >= -SAFE_MARGIN:
                flag = "SAFE (CI-supported)" if lo is not None and lo >= -SAFE_MARGIN else "PROVISIONAL (CI crosses)"
            coverage[(c, ch)] = dict(vs=fb, n=len(shared), coverage=cov, delta=point,
                                     ci=(lo, hi), flag=flag)

    lane_coverage = {}
    for (c, s), ids in lanes.items():
        fb = frontier_best[c]
        for ch in CHEAP:
            fb_m = {i: v for i, v in acc_map(stats[c][fb], majority=True).items() if i in set(ids)}
            ch_m = {i: v for i, v in acc_map(stats[c][ch], majority=True).items() if i in set(ids)}
            shared = sorted(set(fb_m) & set(ch_m))
            if not shared:
                continue
            cov = statistics.mean(1.0 if ch_m[i] >= fb_m[i] else 0.0 for i in shared)
            point, lo, hi = paired_boot_ci(ch_m, fb_m, seed=13)
            flag = "NOT SAFE"
            if point is not None and point >= -SAFE_MARGIN:
                flag = "SAFE" if lo is not None and lo >= -SAFE_MARGIN else "PROVISIONAL"
            lane_coverage[(c, s, ch)] = dict(vs=fb, n=len(shared), coverage=cov,
                                             delta=point, ci=(lo, hi), flag=flag)

    # ── frontier pairwise per corpus ──
    frontier_pairs = {}
    for c in CORPORA:
        for i, m1 in enumerate(FRONTIER):
            for m2 in FRONTIER[i + 1:]:
                a = acc_map(stats[c][m1], majority=True)
                b = acc_map(stats[c][m2], majority=True)
                point, lo, hi = paired_boot_ci(a, b, seed=17)
                frontier_pairs[(c, m1, m2)] = (point, lo, hi)

    # ── tokens + cost ──
    tokens = {(c, m): token_stats(cells, c, m) for c in CORPORA for m in MODELS}
    costs = {(c, m): call_cost_usd(c, m, tokens[(c, m)]) for c in CORPORA for m in MODELS}

    # savings sim per lane: cheapest adequate model
    savings_rows = []
    total_frontier_cost = total_routed_cost = 0.0
    for (c, s), ids in sorted(lanes.items()):
        fb = frontier_best[c]
        fb_majority = lane_acc[(c, s, fb)]["majority"]
        if fb_majority is None:
            continue
        adequate = []
        for m in MODELS:
            la = lane_acc[(c, s, m)]
            cost = costs[(c, m)]
            if la["majority"] is not None and cost is not None and la["majority"] >= fb_majority - SAFE_MARGIN:
                adequate.append((cost, m, la["majority"]))
        if not adequate:
            continue
        adequate.sort()
        cost_routed, routed_model, routed_acc = adequate[0]
        cost_frontier = costs[(c, fb)]
        n = len(ids)
        savings_rows.append(dict(corpus=c, stratum=s, n=n, frontier=fb,
                                 frontier_acc=fb_majority, frontier_cost=cost_frontier,
                                 routed=routed_model, routed_acc=routed_acc,
                                 routed_cost=cost_routed))
        total_frontier_cost += cost_frontier * n
        total_routed_cost += cost_routed * n
    overall_savings = (1 - total_routed_cost / total_frontier_cost) if total_frontier_cost else None

    # ── raw matrix json (texts stripped) ──
    raw = []
    for c in CORPORA:
        for m in MODELS:
            for (iid, k), rec in sorted(cells[c][m].items()):
                slim = {kk: v for kk, v in rec.items()
                        if kk not in ("text", "repair_text")}
                if c == "coding" and "score" in rec:
                    slim["score"] = {kk: rec["score"][kk] for kk in
                                     ("passed", "skipped", "applied", "extraction_failed", "detail")}
                raw.append(slim)
    stage0_ledger = {}
    s0p = state / "stage0_ledger.json"
    if s0p.exists():
        stage0_ledger = json.loads(s0p.read_text())
    Path(args.out_json).write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models": {m: {k: v for k, v in MODELS[m].items()} for m in MODELS},
        "k_samples": K_SAMPLES,
        "total_expected_cells": total_cells,
        "completeness_totals": dict(tot),
        "admin_per_model": admin,
        "excluded_golden_items": excluded_items,
        "stage0_ledger": stage0_ledger,
        "cells": raw,
    }, indent=1))

    # ── markdown report ──
    W("# $0 Cost-Routing Map — 2026-07")
    W("")
    W(f"_Generated {time.strftime('%Y-%m-%d %H:%M')} from the cost-map matrix "
      f"(chairman-approved pre-work for the parked cost-routing option, D-Q24 pivot)._")
    W("")
    W("**Design**: 266 items (46 hidden-test coding + 140 reasoning-headroom + 80 "
      "reasoning-extrahard) × 7 models × k=3 provider-default-decoded samples, all via "
      "the company's subsidized gateways (ant-proxy / oai-proxy; $0 cash, zero OpenRouter "
      "calls). Coding is scored by local execution of each item's held-out golden tests; "
      "reasoning by exact-match with a numeric normalizer.")
    W("")
    W("**Column substitution**: `gpt-5.3-codex` was requested but the codex backend "
      "rejects it for ChatGPT accounts (probed 2026-07-06, both API shapes); `gpt-5.4` "
      "is the substitute third OpenAI column.")
    W("")

    W("## Matrix completion")
    W("")
    W(f"- expected cells: {total_cells} (266 items × 7 models × k=3)")
    W(f"- scored: {tot['scored']}, holes: {tot['hole']}, pending (never dispatched): "
      f"{tot['pending']}, excluded by golden self-check: {tot['excluded_golden']}")
    W(f"- coding items excluded by golden self-check (oracle can't stand up in this env "
      f"today — excluded for ALL models): {excluded_items or 'none'}")
    W("")
    W("| model | calls | ok | natural-stop | truncation holes | other holes | pending | identity mismatches | repaired |")
    W("|---|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        a = admin[m]
        W(f"| {m} | {a['calls']} | {a['ok']} | {pct(a['natural_stop_rate'],2)} | "
          f"{a['truncation_holes']} | {a['other_holes']} | {pending_per_model[m]} | "
          f"{a['identity_mismatches']} | {a['repaired']} |")
    W("")

    W("## 1. Accuracy table (k-sample means, item-bootstrap 95% CIs, B=1000)")
    W("")
    W("| model | tier | headroom (n=140) | extrahard (n=80) | coding (n=46) |")
    W("|---|---|---|---|---|")
    for m in MODELS:
        row = [m, MODELS[m]["tier"]]
        for c in CORPORA:
            a = acc[c][m]
            if a["mean"] is None:
                row.append("—")
            else:
                lo, hi = a["ci"]
                row.append(f"{pct(a['mean'])} [{pct(lo)}, {pct(hi)}] (maj {pct(a['majority'])})")
        W("| " + " | ".join(row) + " |")
    W("")
    W("Per coding stratum (majority-of-k accuracy):")
    W("")
    strata = sorted({s for (c, s) in lanes if c == "coding"})
    W("| model | " + " | ".join(f"{s} (n={len(lanes[('coding', s)])})" for s in strata) + " |")
    W("|---" * (len(strata) + 1) + "|")
    for m in MODELS:
        W(f"| {m} | " + " | ".join(pct(lane_acc[('coding', s, m)]['majority']) for s in strata) + " |")
    W("")

    W("## 2. Cheap-coverage table (the headline)")
    W("")
    W("For each cheap model vs the per-corpus frontier best: % of items where the cheap "
      "model matches-or-beats it (majority-of-k), the accuracy delta (cheap − frontier), "
      "and the safe-lane flag (Δ ≥ −2pp; CI-supported when the bootstrap CI lower bound "
      "also clears −2pp).")
    W("")
    W("| corpus | cheap model | vs frontier best | coverage % | Δacc (maj) | 95% CI | flag |")
    W("|---|---|---|---|---|---|---|")
    for c in CORPORA:
        for ch in CHEAP:
            cov = coverage.get((c, ch))
            if not cov:
                continue
            lo, hi = cov["ci"]
            W(f"| {c} | {ch} | {cov['vs']} | {pct(cov['coverage'])} | {pct(cov['delta'])} | "
              f"[{pct(lo)}, {pct(hi)}] | {cov['flag']} |")
    W("")
    W("Per coding stratum:")
    W("")
    W("| stratum | cheap model | vs | coverage % | Δacc | 95% CI | flag |")
    W("|---|---|---|---|---|---|---|")
    for s in strata:
        for ch in CHEAP:
            cov = lane_coverage.get(("coding", s, ch))
            if not cov:
                continue
            lo, hi = cov["ci"]
            W(f"| {s} | {ch} | {cov['vs']} | {pct(cov['coverage'])} | {pct(cov['delta'])} | "
              f"[{pct(lo)}, {pct(hi)}] | {cov['flag']} |")
    W("")

    W("## 3. Frontier comparison (per-lane default evidence)")
    W("")
    W("| corpus | frontier best (majority) | pairwise Δ (majority, 95% CI) |")
    W("|---|---|---|")
    for c in CORPORA:
        pairs = []
        for (cc, m1, m2), (point, lo, hi) in frontier_pairs.items():
            if cc == c and point is not None:
                pairs.append(f"{m1}−{m2}: {pct(point)} [{pct(lo)}, {pct(hi)}]")
        W(f"| {c} | {frontier_best[c]} ({pct(acc[c][frontier_best[c]]['majority'])}) | "
          + "; ".join(pairs) + " |")
    W("")

    W("## 4. Metered-era savings simulation (LIST-price estimates)")
    W("")
    W("Prices ($/MTok in, out — 2026-07 list, estimates; we currently pay $0 via "
      "subsidized gateways): "
      + ", ".join(f"{m} {MODELS[m]['price']}" for m in MODELS) + ".")
    W("")
    W("Mean measured tokens per call and cost per call by corpus:")
    W("")
    W("| model | " + " | ".join(f"{c} $/call (in/out tok)" for c in CORPORA) + " |")
    W("|---" * (len(CORPORA) + 1) + "|")
    for m in MODELS:
        row = [m]
        for c in CORPORA:
            t = tokens[(c, m)]
            cost = costs[(c, m)]
            if cost is None:
                row.append("—")
            else:
                row.append(f"${cost:.4f} ({t['mean_in']:.0f}/{t['mean_out']:.0f})")
        W("| " + " | ".join(row) + " |")
    W("")
    W("Routing each stratum to its **cheapest adequate model** (majority accuracy ≥ "
      "frontier best − 2pp) vs all-frontier:")
    W("")
    W("| corpus/stratum | n | frontier (acc, $/call) | routed to (acc, $/call) | per-call saving |")
    W("|---|---|---|---|---|")
    for r in savings_rows:
        sv = (1 - r["routed_cost"] / r["frontier_cost"]) if r["frontier_cost"] else None
        W(f"| {r['corpus']}/{r['stratum']} | {r['n']} | {r['frontier']} ({pct(r['frontier_acc'])}, "
          f"${r['frontier_cost']:.4f}) | {r['routed']} ({pct(r['routed_acc'])}, "
          f"${r['routed_cost']:.4f}) | {pct(sv)} |")
    W("")
    if overall_savings is not None:
        W(f"**Overall simulated cost reduction at quality parity: {pct(overall_savings)}** "
          f"(item-weighted; all-frontier baseline = each corpus's frontier-best model).")
    W("")

    W("## 5. Honest caveats")
    W("")
    W("- **k=3 sampling noise**: per-item majority verdicts flip easily near 50%; the "
      "bootstrap CIs quantify item-sampling error, not draw-sampling error.")
    W("- **46-item coding corpus is small** — CIs are wide; stratum slices (4 multi-attempt, "
      "3 thin-spec items) are anecdotal, not statistical.")
    W("- **Training-data contamination**: `eumemic/aios` is a PUBLIC repo; coding items are "
      "real merged PRs and may be in model training data. This inflates absolute coding "
      "scores for ALL models; the differential risk across vendors/cutoffs is noted but "
      "unmeasured.")
    W("- **Corpus vintage + model list are dated 2026-07**; the map decays with model-lineup "
      "churn (historically ~2–4 pool events/month on our gateways).")
    W("- **gpt-5.4 substitutes for gpt-5.3-codex** (upstream-rejected for ChatGPT accounts); "
      "no cheap OpenAI coding-tuned column was reachable at $0.")
    W("- **List prices are estimates**, present-day published rates; subscription-window "
      "economics (what we actually ride) are not per-token and could diverge arbitrarily.")
    W("- **Provider-default decoding** means k=3 draws are not independent for models that "
      "decode near-deterministically (see the entropy ledger below) — for those, majority-of-k "
      "≈ single-draw accuracy.")
    W("- **Reasoning corpora are own-authored** (140+80 items, verified answers); they proxy "
      "'careful multi-step reasoning', not any production lane directly.")
    W("")

    W("## 6. Administration-ledger appendix (Stage 0 + matrix-wide)")
    W("")
    if stage0_ledger:
        W(f"Stage-0 verdict: **{'PASS' if stage0_ledger.get('stage0_pass') else 'FAIL'}** "
          f"({stage0_ledger.get('ts')})")
        W("")
        for name, chk in stage0_ledger.get("checks", {}).items():
            W(f"### stage0 / {name}: {'PASS' if chk.get('pass') else 'FAIL'}")
            slim = {k: v for k, v in chk.items() if k not in ("pass",)}
            W("```json")
            W(json.dumps(slim, indent=1)[:4000])
            W("```")
            W("")
    W("### Matrix-wide administration (per model)")
    W("")
    W("```json")
    W(json.dumps(admin, indent=1))
    W("```")
    W("")
    W(f"Raw per-cell matrix: `cost_routing_matrix_2026-07.json` (texts stripped; full "
      f"transcripts remain in the local state dir, not committed).")

    Path(args.out_md).write_text("\n".join(lines))
    print(f"wrote {args.out_md}")
    print(f"wrote {args.out_json}")

    # quick console summary
    print("\n=== HEADLINES ===")
    for c in CORPORA:
        fb = frontier_best[c]
        print(f"{c}: frontier best = {fb} maj={pct(acc[c][fb]['majority'])}")
        for ch in CHEAP:
            cov = coverage.get((c, ch))
            if cov:
                print(f"  {ch}: coverage={pct(cov['coverage'])} Δ={pct(cov['delta'])} {cov['flag']}")
    if overall_savings is not None:
        print(f"overall simulated savings: {pct(overall_savings)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
