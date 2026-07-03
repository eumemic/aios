#!/usr/bin/env python3
"""Execution-verified parallel selection — the P0 harness (shared pool, four arms).

Design of record: eumemic-company `architecture/execution-verified-selection-eval.md`.
Extends the Phase B coding machinery (run_coding.py / coding_scorer.py / coding_prompt.py)
— same corpus recipe, same hidden-test oracle, same aios prod-runtime path — with:

  * ONE shared pool of N=4 temperature-sampled candidates per item (all-Opus; §2),
    so every arm comparison is matched-compute by construction;
  * four derived arms: (a) index-0, (b) seeded-random, (c) execution-verified pick
    via the hidden suite (tie-break fewest changed lines), (d) LLM-judge pick
    (sees issue + diffs, NEVER the tests);
  * the leak canary (§3): candidate transcripts grepped for hidden-test paths and
    test identifiers — a hit FAILS the item;
  * per-item oracle-soundness pre-flight: golden passes (2×, flake check), the
    empty patch (base source unchanged) fails;
  * per-item disk checkpointing of every PAID artifact (the killed-run lesson) and
    per-item cost metering with a hard budget stop.

Statistics (§4, pre-registered): primary = exact ONE-SIDED sign test on c-vs-b
discordants at full alpha=.05; secondaries (c-vs-a, d-vs-b, c-vs-d) Holm-corrected among
themselves; effect = Δpass with paired bootstrap 95% CI; win = CI-lower > 0 AND
Δ >= MDE(+15pp); null → TOST equivalence at ±10pp.

Sampling temperature (pre-registered, §2): the harness's existing default for
anthropic/claude-opus-4-8 — the provider REJECTS the `temperature` param ("temperature
is deprecated for this model"), so the param is omitted and sampling runs at the
provider default. Recorded in the results payload; run-to-run variance at this default
is nonzero on this corpus (the c07/c08 flips), which is the raw material of selection.

REDO (2026-07-03, §10 of the design doc): the 07-02 confirmatory was an INVALID
TREATMENT ADMINISTRATION (pool_disagreement 0.125; opus-4-8 removed temperature from
its API surface, so provider-default pools were near-clones). The redo administers a
VERIFIED-diverse pool via per-candidate archetype system prompts
(selection_archetypes.py; candidate idx -> archetype is pre-registered), gated by a
Stage-0 bake-off (proceed only if pool_disagreement >= 0.3 on the bake-off items).
Stage-1 fix folded in: the ~19%% "extraction failures" were a HARVEST bug (assistant
text present on the session but past the first events page) — aios_client now
paginates, and paid_turn keeps polling the SAME session instead of re-paying.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

from aios_client import AiosClient, AiosError
from coding_prompt import build_prompt, extract_sources
from selection_archetypes import ARCHETYPE_NAMES, ARCHETYPES
from coding_scorer import file_at, score_candidate, self_check, source_at_base
from recipes import build_best_single_script
from selection_arms import (
    GLOBAL_SEED,
    JUDGE_SYSTEM,
    N_CANDIDATES,
    build_judge_prompt,
    canary_tokens,
    changed_lines,
    leak_scan,
    one_sided_sign_test,
    parse_judge_pick,
    pick_exec_verified,
    pick_index0,
    pick_seeded_random,
    unified_diff_text,
)
from stats_fusion import mcnemar, holm_correction, mde_note
from tost import paired_tost

HERE = Path(__file__).parent
STATE_DIR = HERE / "selection_state"

BEST_MODEL = "anthropic/claude-opus-4-8"  # pre-registered incumbent (D-Q20 settled)
MDE_PP = 15.0  # §1: pre-declared minimum detectable effect
TOST_MARGIN = 0.10  # §1: equivalence bound if null
ALPHA = 0.05
SAMPLING_NOTE = (
    "provider-default sampling (opus-4-8 removed temperature/top_p/top_k from its API "
    "surface — 400 on every provider); candidate diversity = per-candidate archetype "
    "system prompts: " + "/".join(ARCHETYPE_NAMES)
)
DISAGREEMENT_GATE = 0.30  # §10 Stage-0 hard gate on pool_disagreement_rate

ARM_NAMES = ("a_index0", "b_random", "c_exec", "d_judge")


class BudgetStop(RuntimeError):
    """Raised when the hard per-run budget would be exceeded (checkpoint intact)."""


# ── tiny JSON state helpers (append-only, crash-safe) ────────────────────────
def _jsonl_load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _jsonl_append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _json_store(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


# ── provisioning ─────────────────────────────────────────────────────────────
def ensure_agent(client: AiosClient, name: str, model: str, system: str) -> str:
    """create_agent, or reuse the agent already named ``name`` (resume-idempotent)."""
    try:
        return client.create_agent(name, model, system, {})["id"]
    except AiosError as exc:
        if "409" not in str(exc):
            raise
        resp = client._request("GET", "/v1/agents?limit=200")
        items = resp.get("data", resp) if isinstance(resp, dict) else resp
        for a in items or []:
            if a.get("name") == name:
                return a["id"]
        raise


def provision(client: AiosClient) -> tuple[list[str], str, dict]:
    """Register the generation + judge passthrough workflows (version-pinned) and
    their bound agents. All are R0-shaped all-Opus passthroughs; only the agent
    system prompts differ. The REDO provisions FOUR generation agents — one per
    archetype (selection_archetypes.py) — so pool candidate idx i is produced by
    archetype i's worker. The judge is unchanged (never sees the tests)."""
    script = build_best_single_script(BEST_MODEL)
    gen_wf = client.ensure_workflow("selgen-opus", script, "EVS shared-pool generation (passthrough)")
    gen_wf = gen_wf.get("data", gen_wf) if isinstance(gen_wf, dict) else gen_wf
    gen_bind = f"workflow:{gen_wf['id']}@{gen_wf['version']}" if gen_wf.get("version") else f"workflow:{gen_wf['id']}"
    judge_wf = client.ensure_workflow("seljudge-opus", script, "EVS arm-d judge (passthrough)")
    judge_wf = judge_wf.get("data", judge_wf) if isinstance(judge_wf, dict) else judge_wf
    judge_bind = f"workflow:{judge_wf['id']}@{judge_wf['version']}" if judge_wf.get("version") else f"workflow:{judge_wf['id']}"
    gen_agents = [
        ensure_agent(client, f"selgen-opus-arch{i}-{name}", gen_bind, system)
        for i, (name, system) in enumerate(ARCHETYPES)
    ]
    judge_agent = ensure_agent(client, "seljudge-opus-fixed", judge_bind, JUDGE_SYSTEM)
    binds = {"generation": gen_bind, "judge": judge_bind, "archetypes": ARCHETYPE_NAMES}
    return gen_agents, judge_agent, binds


# ── paid call with the run_coding transient-retry contract ───────────────────
def paid_turn(
    client: AiosClient, agent_id: str, env_id: str, prompt: str, timeout_s: float,
    harvest_extra_s: float = 600.0,
) -> dict:
    """One session turn through a bound agent. Returns {text, cost_uusd, run_id,
    session_id}. One retry on a TRANSIENT inference error (empty text AND zero cost
    == the provider call errored — not a wrong answer; must not bias an arm).

    2026-07-03 hardening (§10 Stage 1): if the timeout elapses with no text but the
    session shows a park (inference dispatched) or the run shows cost (inference
    PAID), keep polling the SAME session for up to ``harvest_extra_s`` more — a
    long full-file generation outlives short timeouts, and re-creating the session
    would pay twice for a turn that is still (or already) materializing. Never
    re-pay when cost has been charged."""
    text = cost = run_id = sid = None
    for _attempt in range(2):
        sess = client.create_session(agent_id, env_id, prompt)
        sid = sess["id"]
        asst = client.wait_for_assistant(sid, timeout_s=timeout_s)
        text = asst.get("content") if asst else None
        run_id = client.park_run_id(sid)
        cost = client.get_run(run_id).get("call_llm_cost_microusd") if run_id else None
        if not text and (run_id or cost):
            # Inference in flight or already paid: extend the harvest window.
            asst = client.wait_for_assistant(sid, timeout_s=harvest_extra_s, poll_s=5.0)
            text = asst.get("content") if asst else None
            run_id = run_id or client.park_run_id(sid)
            cost = client.get_run(run_id).get("call_llm_cost_microusd") if run_id else cost
        if text or cost:
            break
    return {"text": text, "cost_uusd": cost, "run_id": run_id, "session_id": sid}


class Meter:
    """Cumulative paid spend for THIS invocation + replayed prior spend, with a hard
    stop BEFORE the next paid call once the budget is reached."""

    def __init__(self, budget_usd: float) -> None:
        self.budget_usd = budget_usd
        self.new_uusd = 0
        self.replayed_uusd = 0

    def replay(self, cost_uusd: int | None) -> None:
        self.replayed_uusd += cost_uusd or 0

    def charge(self, cost_uusd: int | None) -> None:
        self.new_uusd += cost_uusd or 0

    def gate(self, about_to: str) -> None:
        if self.new_uusd / 1e6 >= self.budget_usd:
            raise BudgetStop(
                f"budget ${self.budget_usd:.2f} reached (new spend ${self.new_uusd / 1e6:.3f}) "
                f"before: {about_to}. Checkpoints are intact — re-run with a higher --budget-usd to continue."
            )


# ── per-item phases (each checkpointed) ──────────────────────────────────────
def item_repo(item: dict, default_repo: str) -> str:
    return item.get("repo") or default_repo


def soundness_preflight(item: dict, repo: str, test_timeout_s: int, state: Path) -> dict:
    """§3 oracle soundness: golden passes (2×, flake check); the EMPTY PATCH — the
    base source unchanged — must FAIL (kills vacuous tests: a suite the unfixed code
    already passes cannot discriminate). Cached to disk (free but slow)."""
    path = state / f"soundness_{item['id']}.json"
    if path.exists():
        return json.loads(path.read_text())
    kw = {"venv_python": os.environ.get("AIOS_VENV_PYTHON"), "timeout_s": test_timeout_s}
    g1 = self_check(item, repo, **kw)
    g2 = self_check(item, repo, **kw)
    base_src = source_at_base(repo, item["base_parent_sha"], item["src_files"])
    b = score_candidate(item, repo, base_src, **kw)
    rec = {
        "item": item["id"],
        "golden1_passed": g1.passed,
        "golden2_passed": g2.passed,
        "flake": g1.passed != g2.passed,
        "base_passed": b.passed,
        "base_detail": b.detail,
        "ok": g1.passed and g2.passed and not b.passed,
        "detail": f"golden={g1.passed}/{g2.passed} base={b.passed} ({b.detail[:60]})",
    }
    _json_store(path, rec)
    return rec


def generate_pool(
    client: AiosClient, gen_agents: list[str], env_id: str, item: dict, repo: str,
    n: int, timeout_s: float, throttle_s: float, meter: Meter, state: Path,
) -> list[dict]:
    """The shared candidate pool: n independent sessions, SAME user prompt, one
    archetype worker per candidate index (the pre-registered diversity mechanism).
    Each candidate is flushed to disk the moment it exists — a mid-run crash must
    not lose paid results."""
    path = state / f"pool_{item['id']}.jsonl"
    pool = {r["idx"]: r for r in _jsonl_load(path)}
    for r in pool.values():
        meter.replay(r.get("cost_uusd"))
    base_src = source_at_base(repo, item["base_parent_sha"], item["src_files"])
    # expected_paths keeps the prompt well-posed for new-file-creation items
    # (empty pre-PR snapshot): the model is told WHICH file(s) to create, mirroring
    # how an existing file's path is shown alongside its content.
    prompt = build_prompt(item["task"], base_src, expected_paths=item["src_files"])
    for idx in range(n):
        if idx in pool:
            continue
        meter.gate(f"pool candidate {idx} of {item['id']}")
        agent_id = gen_agents[idx % len(gen_agents)]
        r = paid_turn(client, agent_id, env_id, prompt, timeout_s)
        rec = {"idx": idx, "archetype": ARCHETYPE_NAMES[idx % len(ARCHETYPE_NAMES)], **r}
        _jsonl_append(path, rec)
        pool[idx] = rec
        meter.charge(r["cost_uusd"])
        print(
            f"    pool[{idx}:{rec['archetype']}] cost_uusd={r['cost_uusd']} run={r['run_id']} "
            f"text={'yes' if r['text'] else 'NONE'}"
        )
        time.sleep(throttle_s)
    return [pool[i] for i in range(n)]


def exec_score_pool(
    item: dict, repo: str, pool: list[dict], test_timeout_s: int, state: Path
) -> list[dict]:
    """Run the hidden suite against every candidate (local compute, $0) — arm (c)'s
    selector AND every arm's outcome oracle. Checkpointed per candidate.

    Post-preflight semantics: the oracle is PROVEN to stand up on this item (golden
    passed 2×), so a scorer 'ENV-SKIP' here is almost surely candidate-caused (e.g.
    the candidate imports a nonexistent module → collection error) and is reclassified
    as FAIL. Only a harness infra exception skips the item."""
    path = state / f"exec_{item['id']}.jsonl"
    scored = {r["idx"]: r for r in _jsonl_load(path)}
    base_src = source_at_base(repo, item["base_parent_sha"], item["src_files"])
    for idx, cand in enumerate(pool):
        if idx in scored:
            continue
        sources = extract_sources(cand["text"], item["src_files"])
        out = score_candidate(
            item, repo, sources,
            venv_python=os.environ.get("AIOS_VENV_PYTHON"), timeout_s=test_timeout_s,
        )
        infra = out.skipped and out.detail.startswith("infra:")
        passed = bool(out.passed)
        detail = out.detail
        if out.skipped and not infra:
            detail = f"reclassified FAIL (env-skip post-preflight): {out.detail[:70]}"
        rec = {
            "idx": idx,
            "passed": passed,
            "infra_skip": infra,
            "applied": out.applied and bool(sources),
            "extraction_failed": not sources,
            "changed_lines": changed_lines(base_src, sources),
            "detail": detail[:160],
        }
        _jsonl_append(path, rec)
        scored[idx] = rec
    return [scored[i] for i in range(len(pool))]


def judge_pick(
    client: AiosClient, judge_agent: str, env_id: str, item: dict, repo: str,
    pool: list[dict], timeout_s: float, meter: Meter, state: Path,
) -> dict:
    """Arm (d): one Opus judge over issue + diffs (NEVER the tests). Checkpointed."""
    path = state / f"judge_{item['id']}.json"
    if path.exists():
        rec = json.loads(path.read_text())
        meter.replay(rec.get("cost_uusd"))
        return rec
    base_src = source_at_base(repo, item["base_parent_sha"], item["src_files"])
    diffs = [
        unified_diff_text(base_src, extract_sources(c["text"], item["src_files"]))
        for c in pool
    ]
    if not any(d.strip() for d in diffs):
        rec = {"pick": 0, "detail": "all candidates unusable; judge skipped; fallback 0",
               "parse_failed": False, "cost_uusd": 0, "run_id": None, "text": None}
        _json_store(path, rec)
        return rec
    meter.gate(f"judge of {item['id']}")
    prompt = build_judge_prompt(item["task"], diffs)
    r = paid_turn(client, judge_agent, env_id, prompt, timeout_s)
    meter.charge(r["cost_uusd"])
    pick, detail = parse_judge_pick(r["text"], len(pool))
    rec = {
        "pick": pick if pick is not None else 0,
        "parse_failed": pick is None,
        "detail": detail,
        "cost_uusd": r["cost_uusd"],
        "run_id": r["run_id"],
        "text": r["text"],
    }
    _json_store(path, rec)
    return rec


# ── the per-item pipeline ────────────────────────────────────────────────────
def process_item(
    client, gen_agents, judge_agent, env_id, item, default_repo,
    n_cands, timeout_s, test_timeout_s, throttle_s, meter, state,
    pool_only: bool = False,
) -> dict:
    iid = item["id"]
    repo = item_repo(item, default_repo)
    rec: dict = {"item": iid, "pr": item.get("pr"), "stratum": item.get("stratum", "unknown")}

    snd = soundness_preflight(item, repo, test_timeout_s, state)
    rec["soundness"] = snd
    if not snd["ok"]:
        rec["status"] = "REJECT_SOUNDNESS"
        print(f"  [{iid}] REJECT (soundness): {snd['detail']}")
        return rec

    pool = generate_pool(client, gen_agents, env_id, item, repo, n_cands,
                         timeout_s, throttle_s, meter, state)
    rec["no_text_candidates"] = sum(1 for c in pool if not c.get("text"))
    rec["pool_cost_uusd"] = sum(c.get("cost_uusd") or 0 for c in pool)
    rec["pool_run_ids"] = [c.get("run_id") for c in pool]

    # Leak canary (§3): transcript contains the diff (fenced blocks are extracted
    # from it), so scanning the full transcript covers both.
    base_src = source_at_base(repo, item["base_parent_sha"], item["src_files"])
    oracle_contents = [file_at(repo, item["merge_sha"], t) or "" for t in item["test_files"]]
    tokens = canary_tokens(item["test_files"], oracle_contents, base_src, item["task"])
    hits = {i: leak_scan(c["text"], tokens) for i, c in enumerate(pool)}
    hits = {i: h for i, h in hits.items() if h}
    rec["leak_hits"] = hits
    if hits:
        rec["status"] = "FAIL_LEAK_CANARY"
        print(f"  [{iid}] FAIL leak canary: {hits}")
        return rec

    scored = exec_score_pool(item, repo, pool, test_timeout_s, state)
    if any(s["infra_skip"] for s in scored):
        rec["status"] = "SKIP_INFRA"
        rec["exec"] = scored
        print(f"  [{iid}] SKIP (scorer infra)")
        return rec

    if pool_only:
        # Stage-0 bake-off: measure the diversity mechanism, spend nothing on the
        # judge. The pool/exec checkpoints are shared with Stage 2 (same state
        # dir), so this spend is replayed — not duplicated — by the full run.
        passed = [s["passed"] for s in scored]
        rec["exec"] = [
            {k: s[k] for k in ("idx", "passed", "changed_lines", "extraction_failed", "detail")}
            for s in scored
        ]
        rec["pool_disagreement"] = len(set(passed)) > 1
        texts = [c["text"] for c in pool]
        rec["duplicate_texts"] = len(texts) - len(set(t for t in texts if t)) if any(texts) else 0
        rec["pool_pass_rate"] = sum(passed) / len(passed) if passed else None
        rec["status"] = "BAKEOFF"
        marks = "".join("P" if p else "f" for p in passed)
        print(
            f"  [{iid}] BAKEOFF pool={marks} disagree={rec['pool_disagreement']} "
            f"dup={rec['duplicate_texts']} no_text={rec['no_text_candidates']} "
            f"${sum(c.get('cost_uusd') or 0 for c in pool) / 1e6:.3f}"
        )
        return rec

    judge = judge_pick(client, judge_agent, env_id, item, repo, pool,
                       timeout_s, meter, state)
    rec["judge_cost_uusd"] = judge.get("cost_uusd") or 0

    passed = [s["passed"] for s in scored]
    lines = [s["changed_lines"] for s in scored]
    c_pick, c_detail = pick_exec_verified(passed, lines)
    picks = {
        "a_index0": pick_index0(len(pool)),
        "b_random": pick_seeded_random(iid, len(pool), GLOBAL_SEED),
        "c_exec": c_pick,
        "d_judge": judge["pick"],
    }
    rec["picks"] = picks
    rec["pick_details"] = {"c_exec": c_detail, "d_judge": judge["detail"]}
    rec["arm_passed"] = {arm: passed[k] for arm, k in picks.items()}
    rec["judge_parse_failed"] = judge.get("parse_failed", False)
    rec["exec"] = [
        {k: s[k] for k in ("idx", "passed", "changed_lines", "extraction_failed", "detail")}
        for s in scored
    ]
    rec["pool_disagreement"] = len(set(passed)) > 1
    texts = [c["text"] for c in pool]
    rec["duplicate_texts"] = len(texts) - len(set(t for t in texts if t)) if any(texts) else 0
    rec["status"] = "SCORED"
    marks = "".join("P" if p else "f" for p in passed)
    print(
        f"  [{iid}] pool={marks} picks a={picks['a_index0']} b={picks['b_random']} "
        f"c={picks['c_exec']} d={picks['d_judge']} -> "
        + " ".join(f"{a}={'P' if rec['arm_passed'][a] else 'f'}" for a in ARM_NAMES)
        + f"  ${(rec['pool_cost_uusd'] + rec['judge_cost_uusd']) / 1e6:.3f}"
    )
    return rec


# ── Stage-0 bake-off report (§10: the diversity-mechanism HARD GATE) ─────────
def bakeoff_report(records: list[dict], meter: Meter, out_path: str, binds: dict) -> None:
    done = [r for r in records if r.get("status") == "BAKEOFF"]
    n = len(done)
    dis = sum(1 for r in done if r.get("pool_disagreement"))
    rate = dis / n if n else 0.0
    mean_pass = sum(r.get("pool_pass_rate") or 0.0 for r in done) / n if n else 0.0
    n_cand = sum(len(r.get("exec") or []) for r in done)
    extr = sum(1 for r in done for s in (r.get("exec") or []) if s.get("extraction_failed"))
    no_text = sum(r.get("no_text_candidates", 0) for r in done)
    dup = sum(r.get("duplicate_texts", 0) for r in done)
    gate = rate >= DISAGREEMENT_GATE
    print("\n" + "=" * 80)
    print("STAGE-0 BAKE-OFF — mechanism: per-candidate archetype system prompts")
    print("=" * 80)
    print(f"  items measured:            {n} ({[r['item'] for r in done]})")
    print(f"  pool_disagreement_rate:    {rate:.3f}  ({dis}/{n})   GATE >= {DISAGREEMENT_GATE}: "
          f"{'PASS' if gate else 'FAIL'}")
    print(f"  mean pool pass-rate:       {mean_pass:.3f} (quality sanity)")
    print(f"  extraction failures:       {extr}/{n_cand} candidates ({extr / n_cand:.1%})" if n_cand else "")
    print(f"  no-text (harvest) fails:   {no_text}/{n_cand}")
    print(f"  duplicate candidate texts: {dup}")
    print(f"  spend: new ${meter.new_uusd / 1e6:.3f} (+ ${meter.replayed_uusd / 1e6:.3f} replayed)")
    payload = {
        "stage": "bakeoff",
        "mechanism": "archetype-prompt-jitter",
        "archetypes": ARCHETYPE_NAMES,
        "binds": binds,
        "gate_threshold": DISAGREEMENT_GATE,
        "pool_disagreement_rate": rate,
        "gate_pass": gate,
        "mean_pool_pass_rate": mean_pass,
        "extraction_failures": extr,
        "no_text_candidates": no_text,
        "n_candidates_measured": n_cand,
        "duplicate_candidate_texts": dup,
        "items": records,
        "spend": {"new_usd": meter.new_uusd / 1e6, "replayed_usd": meter.replayed_uusd / 1e6},
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")


# ── stats + report (§4) ──────────────────────────────────────────────────────
def report(records: list[dict], meter: Meter, out_path: str, n_cands: int, binds: dict,
           sample_seed: int | None = None) -> None:
    scored = [r for r in records if r.get("status") == "SCORED"]
    arms = {a: [r["arm_passed"][a] for r in scored] for a in ARM_NAMES}
    n = len(scored)

    print("\n" + "=" * 80)
    print("EXECUTION-VERIFIED SELECTION — shared-pool arms (all-Opus, N=%d)" % n_cands)
    print(f"sampling: {SAMPLING_NOTE}; seed(b)={GLOBAL_SEED}; MDE={MDE_PP}pp; alpha={ALPHA}")
    print("=" * 80)
    for a in ARM_NAMES:
        acc = sum(arms[a]) / n if n else 0.0
        print(f"  {a:9} pass={acc:.3f} (n={n})")

    payload: dict = {
        "design": "architecture/execution-verified-selection-eval.md",
        "model": BEST_MODEL,
        "n_candidates": n_cands,
        "sampling": SAMPLING_NOTE,
        "seed_b": GLOBAL_SEED,
        "item_sample_seed": sample_seed,
        "mde_pp": MDE_PP,
        "tost_margin": TOST_MARGIN,
        "alpha": ALPHA,
        "binds": binds,
        "n_scored": n,
        "arm_pass": {a: (sum(arms[a]) / n if n else None) for a in ARM_NAMES},
        "items": records,
    }

    # per-stratum breakdown (§3: report per-stratum; headline = intake-weighted mean)
    strata = sorted({r["stratum"] for r in scored})
    payload["per_stratum"] = {}
    for s in strata:
        sub = [r for r in scored if r["stratum"] == s]
        payload["per_stratum"][s] = {
            "n": len(sub),
            **{a: sum(r["arm_passed"][a] for r in sub) / len(sub) for a in ARM_NAMES},
        }
    if strata:
        print("  per-stratum: " + json.dumps(payload["per_stratum"]))

    if n >= 1:
        primary = one_sided_sign_test("c_vs_b(primary)", arms["b_random"], arms["c_exec"])
        eff = mcnemar("c_vs_b_effect", arms["b_random"], arms["c_exec"])
        secondaries = [
            mcnemar("c_vs_a", arms["a_index0"], arms["c_exec"]),
            mcnemar("d_vs_b", arms["b_random"], arms["d_judge"]),
            mcnemar("c_vs_d", arms["d_judge"], arms["c_exec"]),
        ]
        holm = holm_correction([(s.name, s.p_exact) for s in secondaries], alpha=ALPHA)
        print("\n--- PRIMARY: exact one-sided sign test, c (exec-verified) vs b (random pick) ---")
        print(
            f"  discordants={primary.n_discordant} (c-only={primary.c_only}, "
            f"b-only={primary.b_only} [oracle-flakiness indicator]) p={primary.p_one_sided:.4g}"
        )
        print(f"  Δpass(c-b)={eff.delta:+.3f} 95% CI [{eff.ci_low:+.3f}, {eff.ci_high:+.3f}]")
        print(f"  {mde_note(n, eff.acc_a, MDE_PP, ALPHA)}")
        print("\n--- SECONDARIES (Holm-corrected among themselves) ---")
        for s in secondaries:
            h = holm[s.name]
            print(f"  {s.name}: Δ={s.delta:+.3f} p_raw={h['p_raw']:.4g} p_holm={h['p_holm']:.4g} reject={h['reject']}")
        win = primary.p_one_sided < ALPHA and eff.ci_low > 0 and eff.delta >= MDE_PP / 100
        verdict = (
            "WIN: exec-verified selection beats random-pick past the MDE"
            if win
            else "NO established win at this n"
        )
        if not win and n >= 3:
            diffs = [int(c) - int(b) for b, c in zip(arms["b_random"], arms["c_exec"])]
            t = paired_tost(diffs, margin=TOST_MARGIN, alpha=ALPHA)
            equiv = bool(getattr(t, "equivalent", False))
            verdict += f"; TOST(±{TOST_MARGIN:.0%}) equivalent={equiv}"
            payload["tost_c_vs_b"] = {"equivalent": equiv, "detail": t.summary()}
        print(f"\n  >> {verdict}")
        payload["primary"] = {
            "test": "one_sided_sign_c_vs_b",
            "n_discordant": primary.n_discordant,
            "c_only": primary.c_only,
            "b_only": primary.b_only,
            "p": primary.p_one_sided,
            "delta": eff.delta,
            "ci": [eff.ci_low, eff.ci_high],
        }
        payload["secondaries"] = {
            s.name: {"delta": s.delta, **holm[s.name]} for s in secondaries
        }
        payload["verdict"] = verdict

    # integrity block (§P2 report contract)
    rejects = [r for r in records if r.get("status") == "REJECT_SOUNDNESS"]
    leaks = [r for r in records if r.get("status") == "FAIL_LEAK_CANARY"]
    flakes = [r for r in records if r.get("soundness", {}).get("flake")]
    extraction_by_arm = {
        a: sum(
            1 for r in scored if r["exec"][r["picks"][a]]["extraction_failed"]
        )
        for a in ARM_NAMES
    }
    integrity = {
        "soundness_rejects": [r["item"] for r in rejects],
        "oracle_flakes": [r["item"] for r in flakes],
        "leak_canary_failures": {r["item"]: r["leak_hits"] for r in leaks},
        "judge_parse_failures": [r["item"] for r in scored if r.get("judge_parse_failed")],
        "b_only_discordants(oracle_flakiness)": sum(
            1 for r in scored if r["arm_passed"]["b_random"] and not r["arm_passed"]["c_exec"]
        ),
        "extraction_failures_of_picked_candidate_per_arm": extraction_by_arm,
        "extraction_failures_any_candidate": sum(
            1 for r in scored for s in r["exec"] if s.get("extraction_failed")
        ),
        "no_text_candidates_total": sum(r.get("no_text_candidates", 0) for r in scored),
        "pool_disagreement_rate": (
            sum(1 for r in scored if r.get("pool_disagreement")) / n if n else None
        ),
        "duplicate_candidate_texts_total": sum(r.get("duplicate_texts", 0) for r in scored),
    }
    payload["integrity"] = integrity
    payload["spend"] = {
        "new_usd_this_invocation": meter.new_uusd / 1e6,
        "replayed_usd_from_checkpoints": meter.replayed_uusd / 1e6,
        "budget_usd": meter.budget_usd,
    }
    print("\n--- INTEGRITY ---")
    for k, v in integrity.items():
        print(f"  {k}: {v}")
    pdr = integrity["pool_disagreement_rate"]
    if pdr is not None and pdr < DISAGREEMENT_GATE:
        print(
            f"  !! TREATMENT-ADMINISTRATION WARNING: full-run pool_disagreement_rate "
            f"{pdr:.3f} < {DISAGREEMENT_GATE} (the Stage-0 gate) — flag prominently."
        )
    print(
        f"\n--- SPEND: new ${meter.new_uusd / 1e6:.3f} this invocation "
        f"(+ ${meter.replayed_uusd / 1e6:.3f} replayed from checkpoints; budget ${meter.budget_usd:.2f}) ---"
    )
    Path(out_path).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Execution-verified selection eval (P0 harness).")
    ap.add_argument("--tasks", default="coding_selection.json", help="corpus in tasks/")
    ap.add_argument("--n", type=int, default=12, help="max items this run (pilot n=12)")
    ap.add_argument("--items", default=None, help="comma-separated item ids (overrides --n slice)")
    ap.add_argument(
        "--sample-seed", type=int, default=None,
        help="seeded shuffle of the corpus before the --n slice (pre-registered pilot "
        "sampling — avoids 'first n' bias toward the seed items; recorded in results)",
    )
    ap.add_argument("--n-candidates", type=int, default=N_CANDIDATES)
    ap.add_argument("--budget-usd", type=float, default=16.0, help="hard stop on NEW paid spend")
    ap.add_argument("--timeout-s", type=float, default=900.0)
    ap.add_argument("--test-timeout-s", type=int, default=120)
    ap.add_argument("--throttle-s", type=float, default=1.0)
    ap.add_argument("--env-id", default=None)
    ap.add_argument("--state-dir", default=str(STATE_DIR))
    ap.add_argument("--out", default=str(HERE / "selection_results.json"))
    ap.add_argument("--single-file-only", action="store_true", default=True)
    ap.add_argument(
        "--stage", choices=("bakeoff", "full"), default="full",
        help="bakeoff = Stage-0 diversity-mechanism measurement (pool + exec only, "
        "no judge, no arms); full = the confirmatory run (§10). Same state dir => "
        "bake-off pools are replayed by the full run at $0.",
    )
    args = ap.parse_args()

    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(line_buffering=True)

    corpus = json.loads((HERE / "tasks" / args.tasks).read_text())
    default_repo = corpus["repo"]
    items = corpus["items"]
    if args.single_file_only:
        items = [it for it in items if len(it["src_files"]) == 1]
    if args.items:
        want = set(args.items.split(","))
        items = [it for it in items if it["id"] in want]
    else:
        if args.sample_seed is not None:
            import random as _random

            _random.Random(args.sample_seed).shuffle(items)
        items = items[: args.n]

    client = AiosClient()
    acct = client.whoami()
    print(f"# account {acct['id']} spend_limit_usd={acct.get('config', {}).get('spend_limit_usd')}")
    env_id = args.env_id or client.ensure_environment("wam-eval-env")["id"]
    gen_agents, judge_agent, binds = provision(client)
    print(f"# generation bind {binds['generation']} x{len(gen_agents)} archetypes | judge bind {binds['judge']}")
    print(f"# stage={args.stage}; {len(items)} items; N={args.n_candidates}; budget ${args.budget_usd}; {SAMPLING_NOTE}")

    state = Path(args.state_dir)
    meter = Meter(args.budget_usd)
    records: list[dict] = []
    try:
        for it in items:
            records.append(
                process_item(
                    client, gen_agents, judge_agent, env_id, it, default_repo,
                    args.n_candidates, args.timeout_s, args.test_timeout_s,
                    args.throttle_s, meter, state,
                    pool_only=(args.stage == "bakeoff"),
                )
            )
    except BudgetStop as stop:
        print(f"\n!! BUDGET STOP: {stop}")
    if args.stage == "bakeoff":
        bakeoff_report(records, meter, args.out, binds)
    else:
        report(records, meter, args.out, args.n_candidates, binds, sample_seed=args.sample_seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
