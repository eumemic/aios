#!/usr/bin/env python3
"""$0 cost-routing map — matrix runner (Stage 0 administration ledger + full matrix).

Measures cheap-vs-frontier model coverage on OUR OWN task corpora, using ONLY the
company's subsidized gateways (ant-proxy / oai-proxy; zero cash, ZERO OpenRouter):

  corpora:  tasks/coding_selection.json      (46 hidden-test coding items)
            tasks/reasoning_headroom.json    (140 checkable reasoning items)
            tasks/reasoning_extrahard.json   (80 checkable reasoning items)
  columns:  4 Anthropic models via ant-proxy + 3 OpenAI models via oai-proxy
            (gpt-5.4 SUBSTITUTES for the requested gpt-5.3-codex, which the codex
            backend rejects for ChatGPT accounts: "model is not supported when using
            Codex with a ChatGPT account" — probed 2026-07-06 on both API shapes)
  design:   every item x 7 models x k=3 provider-default-decoded samples,
            per-cell disk checkpoints (JSONL, resume-safe, paid work never lost).

STAGE 0 (must PASS before matrix spend; defends the param-drop / wrong-model-served
fraud class that voided a prior company experiment):
  1. provider-default decoding everywhere (enforced in cost_map_gateways).
  2. headroom probe — 10 hard items/model at matrix max_tokens; plus a coding
     max_tokens acceptance probe (1 coding item/model).
  3. identity assertion on EVERY call (mismatch => quarantine + alarm).
  4. extraction calibration — 20 reasoning items/model with the corpus ANSWER: contract
     + one deterministic repair re-ask; >=98% parseable/model; cross-model spread <=2pp.
  5. error != wrong — bounded-backoff retries then HOLE (tracked, reported per model).
  6. entropy probe — 8 draws x 3 fixed items/model; unique-output rate recorded.

Concurrency: <=3 concurrent TOTAL across the GPT models (single ChatGPT-Pro account);
<=6 concurrent across ant-proxy models (7-account pool). The oai lane polls
/admin/api/pool between batches and HALTS (checkpoint + report resume time) when the
subscription window is exhausted — no busy-waiting.

Usage (run from evals/wam_fusion, with the aios venv python for `requests`):
  OAI_CLIENT_KEY=... python run_cost_map.py --stage stage0
  OAI_CLIENT_KEY=... python run_cost_map.py --stage matrix
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

from cost_map_gateways import (
    CallResult,
    Hole,
    QuotaExhausted,
    call_with_retry,
    oai_pool_state,
)
from coding_prompt import CODING_SYSTEM, build_prompt
from coding_scorer import source_at_base
from scoring import extract_answer, is_correct

HERE = Path(__file__).parent
STATE_DIR = HERE / "results" / "cost_map_state"

# ── model columns (ALL subsidized; prices = 2026-07 LIST standard rates, $/MTok,
#    verified 2026-07-06: Anthropic platform docs / OpenAI pricing table; ESTIMATES for
#    the metered-era simulation only — we currently pay $0 via the subsidized gateways) ──
MODELS: dict[str, dict] = {
    "opus-4-8": {"gw": "ant", "req": "claude-opus-4-8", "tier": "frontier", "price": (5.0, 25.0)},
    "sonnet-5": {"gw": "ant", "req": "claude-sonnet-5", "tier": "cheap", "price": (3.0, 15.0)},
    "haiku-4-5": {"gw": "ant", "req": "claude-haiku-4-5-20251001", "tier": "cheap", "price": (1.0, 5.0)},
    "fable-5": {"gw": "ant", "req": "claude-fable-5", "tier": "frontier", "price": (10.0, 50.0)},
    "gpt-5.5": {"gw": "oai", "req": "gpt-5.5", "tier": "frontier", "price": (5.0, 30.0)},
    "gpt-5.4-mini": {"gw": "oai", "req": "gpt-5.4-mini", "tier": "cheap", "price": (0.75, 4.5)},
    # SUBSTITUTE column: gpt-5.3-codex is unavailable via the ChatGPT-account codex
    # backend (probed 2026-07-06); gpt-5.4 is the closest available cheaper-than-5.5
    # OpenAI coding-capable model on this gateway.
    "gpt-5.4": {"gw": "oai", "req": "gpt-5.4", "tier": "cheap-sub", "price": (2.5, 15.0)},
}
ANT_CONCURRENCY = 6
OAI_CONCURRENCY = 3
K_SAMPLES = 3

# max_tokens: generous by design (truncation = HOLE, and holes must stay <1%).
# Coding needs to re-emit a whole source file (largest corpus file ~18.5k tokens);
# oai reasoning models spend reasoning tokens INSIDE max_output_tokens.
MAX_TOKENS = {
    ("ant", "reasoning"): 16000,
    ("ant", "coding"): 30000,
    ("oai", "reasoning"): 32000,
    ("oai", "coding"): 64000,
}
TIMEOUT_S = {"ant": 900.0, "oai": 640.0}

REASONING_SYSTEM = "You are a careful problem-solver. Show brief reasoning, then the ANSWER: line."
REPAIR_INSTR = (
    "Your reply above did not end with the required final line. Output ONLY one line of "
    "EXACTLY the form:  ANSWER: <value>   — your final answer to the problem, no other text."
)

CORPORA = ("headroom", "extrahard", "coding")
OAI_POOL_CHECK_EVERY = 20  # oai cells between /admin/api/pool checks
OAI_HALT_UTILIZATION = 97.0  # % of the primary window


# ── corpus loading ────────────────────────────────────────────────────────────
def load_corpora() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name, fname in (
        ("headroom", "reasoning_headroom.json"),
        ("extrahard", "reasoning_extrahard.json"),
    ):
        d = json.loads((HERE / "tasks" / fname).read_text())
        out[name] = {
            "kind": "reasoning",
            "fmt": d["answer_format_instruction"],
            "items": d["items"],
        }
    d = json.loads((HERE / "tasks" / "coding_selection.json").read_text())
    out["coding"] = {"kind": "coding", "repo": d["repo"], "items": d["items"]}
    return out


def build_user_prompt(corpus: dict, item: dict) -> str:
    if corpus["kind"] == "reasoning":
        return f"{item['prompt']}\n\n{corpus['fmt']}"
    base_src = source_at_base(corpus["repo"], item["base_parent_sha"], item["src_files"])
    return build_prompt(item["task"], base_src, expected_paths=item["src_files"])


def system_prompt(corpus: dict) -> str:
    return REASONING_SYSTEM if corpus["kind"] == "reasoning" else CODING_SYSTEM


# ── checkpoint store (thread-safe JSONL append, one file per model x corpus x phase) ──
class Store:
    def __init__(self, state_dir: Path):
        self.dir = state_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._done: dict[str, dict] = {}
        self._meta_lock = threading.Lock()

    def _path(self, phase: str, corpus: str, model: str) -> Path:
        return self.dir / f"{phase}_{corpus}_{model}.jsonl"

    def load_done(self, phase: str, corpus: str, model: str) -> dict:
        key = f"{phase}_{corpus}_{model}"
        with self._meta_lock:
            if key not in self._done:
                recs = {}
                p = self._path(phase, corpus, model)
                if p.exists():
                    for line in p.read_text().splitlines():
                        if line.strip():
                            r = json.loads(line)
                            recs[(r["item"], r["k"])] = r
                self._done[key] = recs
            return self._done[key]

    def append(self, phase: str, corpus: str, model: str, rec: dict) -> None:
        key = f"{phase}_{corpus}_{model}"
        with self._locks[key]:
            with self._path(phase, corpus, model).open("a") as f:
                f.write(json.dumps(rec) + "\n")
        with self._meta_lock:
            self._done.setdefault(key, {})[(rec["item"], rec["k"])] = rec


# ── one cell = one paid sample ────────────────────────────────────────────────
def run_cell(
    corpus_name: str,
    corpus: dict,
    item: dict,
    model_name: str,
    k: int,
    repair: bool = True,
) -> dict:
    m = MODELS[model_name]
    kind = corpus["kind"]
    max_tokens = MAX_TOKENS[(m["gw"], kind)]
    sys_p = system_prompt(corpus)
    user = build_user_prompt(corpus, item)
    iid = item["id"]
    rec: dict = {
        "item": iid,
        "k": k,
        "model": model_name,
        "corpus": corpus_name,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        r = call_with_retry(m["gw"], m["req"], sys_p, [{"role": "user", "content": user}],
                            max_tokens, TIMEOUT_S[m["gw"]])
    except Hole as h:
        rec.update(status="HOLE", hole_kind=h.kind, hole_detail=h.detail, attempts=h.attempts)
        return rec

    rec.update(
        served_model=r.served_model,
        stop_reason=r.stop_reason,
        input_tokens=r.input_tokens,
        output_tokens=r.output_tokens,
        reasoning_tokens=r.reasoning_tokens,
        latency_s=r.latency_s,
        attempts=r.attempts,
        identity_ok=r.identity_ok,
    )
    if not r.identity_ok:
        # wrong model served: QUARANTINE — never a scoreable sample.
        rec.update(status="QUARANTINE_IDENTITY", text=r.text[:2000])
        return rec
    if r.stop_reason == "max_tokens":
        rec.update(status="HOLE", hole_kind="truncation", hole_detail=f"max_tokens={max_tokens}")
        return rec

    rec["status"] = "ok"
    rec["text"] = r.text
    if kind == "reasoning":
        ext = extract_answer(r.text)
        # The corpus contract puts ANSWER: on the final line; extract_answer's fallback
        # (last line) can misfire on prose. Count "parsed" as: an explicit marker found.
        has_marker = bool(r.text) and "answer:" in r.text.lower()
        rec["parsed"] = has_marker
        rec["repaired"] = False
        if not has_marker and repair:
            tail = (r.text or "")[-3000:]
            try:
                r2 = call_with_retry(
                    m["gw"], m["req"], sys_p,
                    [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": tail or "(empty reply)"},
                        {"role": "user", "content": REPAIR_INSTR},
                    ],
                    1000, TIMEOUT_S[m["gw"]],
                )
                rec["repaired"] = True
                rec["repair_text"] = r2.text
                rec["repair_tokens"] = (r2.input_tokens, r2.output_tokens)
                if r2.text and "answer:" in r2.text.lower():
                    ext = extract_answer(r2.text)
                    rec["parsed"] = True
            except (Hole, QuotaExhausted):
                pass  # keep the un-repaired extraction (or None)
        rec["extracted"] = ext
        rec["correct"] = is_correct(ext, item["answer"])
    return rec


# ── worker pools ──────────────────────────────────────────────────────────────
class Runner:
    def __init__(self, store: Store, phase: str):
        self.store = store
        self.phase = phase
        self.oai_halt = threading.Event()
        self.oai_halt_reason: str | None = None
        self.counters = defaultdict(int)
        self.count_lock = threading.Lock()
        self.oai_cells_since_check = 0
        self.total = 0

    def bump(self, key: str) -> int:
        with self.count_lock:
            self.counters[key] += 1
            self.counters["done"] += 1
            return self.counters["done"]

    def check_oai_pool(self) -> None:
        with self.count_lock:
            self.oai_cells_since_check += 1
            due = self.oai_cells_since_check >= OAI_POOL_CHECK_EVERY
            if due:
                self.oai_cells_since_check = 0
        if not due:
            return
        pool = oai_pool_state()
        if not pool:
            return
        for acct in pool.get("accounts", []):
            prim = (acct.get("windows") or {}).get("primary") or {}
            util = prim.get("utilization")
            if not acct.get("available") or (util is not None and util >= OAI_HALT_UTILIZATION):
                self.oai_halt_reason = (
                    f"pool window: available={acct.get('available')} "
                    f"primary_utilization={util} resets_at={prim.get('resets_at')}"
                )
                self.oai_halt.set()
                print(f"!! OAI_HALT {self.oai_halt_reason}", flush=True)

    def worker(self, gw: str, queue: list, corpora: dict) -> None:
        while True:
            try:
                cell = queue.pop(0)
            except IndexError:
                return
            corpus_name, item, model_name, k = cell
            if gw == "oai" and self.oai_halt.is_set():
                self.bump("skipped_oai_halt")
                continue
            corpus = corpora[corpus_name]
            try:
                rec = run_cell(corpus_name, corpus, item, model_name, k)
            except QuotaExhausted as q:
                self.oai_halt_reason = f"429 quota: {q}"
                self.oai_halt.set()
                print(f"!! OAI_HALT {self.oai_halt_reason}", flush=True)
                self.bump("skipped_oai_halt")
                continue
            except Exception as exc:  # never wedge the pool on one cell
                rec = {
                    "item": item["id"], "k": k, "model": model_name, "corpus": corpus_name,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "status": "HOLE", "hole_kind": "harness_exception",
                    "hole_detail": f"{type(exc).__name__}: {exc}"[:300], "attempts": 1,
                }
            self.store.append(self.phase, corpus_name, model_name, rec)
            done = self.bump(rec["status"] if rec["status"] != "ok" else "ok")
            st = rec["status"]
            mark = {"ok": ".", "HOLE": "H"}.get(st, "!")
            if st != "ok":
                print(
                    f"[{done}/{self.total}] {corpus_name}/{item['id']}/{model_name}/k{k} "
                    f"{st} {rec.get('hole_kind','')} {str(rec.get('hole_detail',''))[:120]}",
                    flush=True,
                )
            elif done % 25 == 0 or done == self.total:
                with self.count_lock:
                    summary = dict(self.counters)
                print(f"PROGRESS {json.dumps(summary)} total={self.total}", flush=True)
            if gw == "oai":
                self.check_oai_pool()

    def run_cells(self, cells: list, corpora: dict) -> None:
        ant_q = [c for c in cells if MODELS[c[2]]["gw"] == "ant"]
        oai_q = [c for c in cells if MODELS[c[2]]["gw"] == "oai"]
        self.total = len(ant_q) + len(oai_q)
        if self.total == 0:
            print("# nothing to do (all cells checkpointed)")
            return
        threads = [
            threading.Thread(target=self.worker, args=("ant", ant_q, corpora), daemon=True)
            for _ in range(min(ANT_CONCURRENCY, len(ant_q)))
        ] + [
            threading.Thread(target=self.worker, args=("oai", oai_q, corpora), daemon=True)
            for _ in range(min(OAI_CONCURRENCY, len(oai_q)))
        ]
        print(f"# dispatching {len(ant_q)} ant + {len(oai_q)} oai cells "
              f"({ANT_CONCURRENCY}/{OAI_CONCURRENCY} concurrent)", flush=True)
        for t in threads:
            t.start()
        for t in threads:
            t.join()


def pending_cells(store: Store, phase: str, corpora: dict, k_samples: int,
                  corpus_names=CORPORA, models=None, item_slice=None) -> list:
    """Item-major cell ordering: a partially-completed run still yields complete
    item rows (all models x k) for as many items as were reached."""
    cells = []
    models = models or list(MODELS)
    for cname in corpus_names:
        corpus = corpora[cname]
        items = corpus["items"] if item_slice is None else corpus["items"][item_slice]
        for item in items:
            for mname in models:
                done = store.load_done(phase, cname, mname)
                for k in range(k_samples):
                    if (item["id"], k) not in done:
                        cells.append((cname, item, mname, k))
    return cells


# ── Stage 0 ───────────────────────────────────────────────────────────────────
def stage0(store: Store, corpora: dict) -> int:
    runner = Runner(store, "stage0")
    ledger: dict = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "checks": {}}

    # (2) headroom probe: first 10 extrahard items, matrix max_tokens; + 1 coding item.
    probe_cells = pending_cells(store, "stage0", corpora, 1,
                                corpus_names=("extrahard",), item_slice=slice(0, 10))
    probe_cells += pending_cells(store, "stage0", corpora, 1,
                                 corpus_names=("coding",), item_slice=slice(0, 1))
    # (4) calibration: first 20 headroom items.
    calib_cells = pending_cells(store, "stage0", corpora, 1,
                                corpus_names=("headroom",), item_slice=slice(0, 20))
    # (6) entropy probe: 8 draws on 3 fixed items (k = draw index).
    entropy_items = [corpora["headroom"]["items"][9], corpora["headroom"]["items"][69],
                     corpora["extrahard"]["items"][39]]
    entropy_cells = []
    for item in entropy_items:
        for mname in MODELS:
            done = store.load_done("stage0", "entropy", mname)
            for k in range(8):
                if (item["id"], k) not in done:
                    entropy_cells.append(("entropy", item, mname, k))
    # entropy runs against the headroom/extrahard prompts — register a synthetic corpus
    ent_corpus = {"kind": "reasoning", "fmt": corpora["headroom"]["fmt"],
                  "items": entropy_items}
    corpora_all = dict(corpora)
    corpora_all["entropy"] = ent_corpus

    runner.run_cells(probe_cells + calib_cells + entropy_cells, corpora_all)

    # ── evaluate the ledger ──
    def recs(corpus_names, models=MODELS):
        out = []
        for cname in corpus_names:
            for mname in models:
                out += list(store.load_done("stage0", cname, mname).values())
        return out

    all_recs = recs(("extrahard", "coding", "headroom", "entropy"))
    per_model: dict[str, dict] = {m: defaultdict(int) for m in MODELS}
    for r in all_recs:
        pm = per_model[r["model"]]
        pm["calls"] += 1
        if r["status"] == "ok":
            pm["ok"] += 1
            if r.get("stop_reason") == "end_turn":
                pm["natural_stop"] += 1
            if r.get("identity_ok"):
                pm["identity_ok"] += 1
        elif r["status"] == "QUARANTINE_IDENTITY":
            pm["identity_mismatch"] += 1
        else:
            pm["holes"] += 1
            if r.get("hole_kind") == "truncation":
                pm["truncations"] += 1

    # check: identity
    mism = {m: c["identity_mismatch"] for m, c in per_model.items() if c["identity_mismatch"]}
    ledger["checks"]["identity"] = {"pass": not mism, "mismatches": mism}

    # check: natural stop (probe + all stage0 ok-calls)
    stop_rates = {
        m: (c["natural_stop"] / c["ok"] if c["ok"] else None) for m, c in per_model.items()
    }
    ledger["checks"]["natural_stop"] = {
        "pass": all(v is not None and v >= 0.99 for v in stop_rates.values()),
        "rates": stop_rates,
        "note": "matrix-wide rate re-checked at analysis; this is the stage0 sample",
    }

    # check: calibration parse rate (with repair) per model on the 20 headroom items
    parse_rates = {}
    for m in MODELS:
        rs = [r for r in store.load_done("stage0", "headroom", m).values()
              if r["status"] == "ok"]
        n = len(rs)
        parse_rates[m] = (sum(1 for r in rs if r.get("parsed")) / n) if n else None
    vals = [v for v in parse_rates.values() if v is not None]
    spread = (max(vals) - min(vals)) if vals else None
    ledger["checks"]["extraction_calibration"] = {
        "pass": all(v is not None and v >= 0.98 for v in parse_rates.values())
        and (spread is not None and spread <= 0.02),
        "parse_rates": parse_rates,
        "cross_model_spread": spread,
        "contract": "corpus ANSWER: <value> final line + one deterministic repair re-ask",
    }

    # check: holes
    holes = {m: c["holes"] for m, c in per_model.items()}
    ledger["checks"]["holes"] = {
        "pass": True,  # holes don't fail stage0; they are tracked (error != wrong)
        "per_model": holes,
        "truncations": {m: c["truncations"] for m, c in per_model.items()},
    }

    # check: entropy
    entropy = {}
    for m in MODELS:
        rows = {}
        for item in entropy_items:
            draws = [r for r in store.load_done("stage0", "entropy", m).values()
                     if r["item"] == item["id"] and r["status"] == "ok"]
            texts = [r.get("text") for r in draws if r.get("text")]
            answers = [r.get("extracted") for r in draws]
            rows[item["id"]] = {
                "draws": len(draws),
                "unique_texts": len(set(texts)),
                "unique_answers": len(set(a for a in answers if a is not None)),
                "answer_agreement": (
                    max((answers.count(a) for a in set(answers)), default=0) / len(answers)
                    if answers else None
                ),
            }
        entropy[m] = rows
    ledger["checks"]["entropy_probe"] = {"pass": True, "per_model": entropy,
                                         "note": "ledger only: determinism is valid, just recorded"}

    ledger["per_model_call_stats"] = {m: dict(c) for m, c in per_model.items()}
    ledger["decoding"] = "provider-default everywhere (no temperature/top_p/top_k sent)"
    ledger["substitution"] = "gpt-5.4 substitutes for gpt-5.3-codex (upstream-rejected for ChatGPT accounts)"

    hard_pass = all(
        ledger["checks"][c]["pass"] for c in ("identity", "natural_stop", "extraction_calibration")
    )
    ledger["stage0_pass"] = hard_pass
    out = store.dir / "stage0_ledger.json"
    out.write_text(json.dumps(ledger, indent=2))
    print("\n" + "=" * 78)
    for name, chk in ledger["checks"].items():
        print(f"STAGE0 {name}: {'PASS' if chk['pass'] else 'FAIL'}")
    print(f"STAGE0 VERDICT: {'PASS — matrix spend authorized' if hard_pass else 'FAIL — fix before matrix'}")
    print(f"wrote {out}")
    print("=" * 78, flush=True)
    return 0 if hard_pass else 1


# ── matrix ────────────────────────────────────────────────────────────────────
def matrix(store: Store, corpora: dict, args) -> int:
    gate = store.dir / "stage0_ledger.json"
    if not gate.exists() or not json.loads(gate.read_text()).get("stage0_pass"):
        print("REFUSING matrix: stage0_ledger.json missing or not PASS", flush=True)
        return 1
    corpus_names = tuple(args.corpora.split(",")) if args.corpora else CORPORA
    models = args.models.split(",") if args.models else None
    if args.retry_truncations:
        # Re-administer cells whose LAST record is a truncation HOLE, at a scaled
        # cap. The cap is administration, not construct (generous-by-design;
        # truncation = an administration failure, never a model verdict) — leaving
        # systematic holes on a verbose model's longest items would bias its column.
        # Store appends are last-record-wins on reload, so a fresh ok record
        # supersedes the HOLE in analysis.
        for key in list(MAX_TOKENS):
            MAX_TOKENS[key] = int(MAX_TOKENS[key] * args.cap_scale)
        cells = []
        for cname in corpus_names:
            items_by_id = {it["id"]: it for it in corpora[cname]["items"]}
            for mname in (models or list(MODELS)):
                done = store.load_done("matrix", cname, mname)
                for (iid, k), r in sorted(done.items()):
                    if r["status"] == "HOLE" and r.get("hole_kind") == "truncation":
                        cells.append((cname, items_by_id[iid], mname, k))
        print(f"# retry-truncations: {len(cells)} truncation-HOLE cells at "
              f"cap x{args.cap_scale} => {MAX_TOKENS}", flush=True)
    else:
        cells = pending_cells(store, "matrix", corpora, K_SAMPLES,
                              corpus_names=corpus_names, models=models)
    runner = Runner(store, "matrix")
    t0 = time.time()
    runner.run_cells(cells, corpora)
    dt = time.time() - t0
    print(f"\nMATRIX PASS COMPLETE in {dt/60:.1f} min: {json.dumps(dict(runner.counters))}")
    if runner.oai_halt.is_set():
        print(f"OAI LANE HALTED: {runner.oai_halt_reason}")
        print("Re-run the same command after the window resets — checkpoints are intact.")
        return 2
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="$0 cost-routing map runner")
    ap.add_argument("--stage", choices=("stage0", "matrix"), required=True)
    ap.add_argument("--state-dir", default=str(STATE_DIR))
    ap.add_argument("--models", default=None, help="comma-separated column subset")
    ap.add_argument("--corpora", default=None, help="comma-separated corpus subset")
    ap.add_argument("--retry-truncations", action="store_true",
                    help="re-run truncation-HOLE cells at a scaled max_tokens cap")
    ap.add_argument("--cap-scale", type=float, default=2.0,
                    help="max_tokens multiplier for --retry-truncations (default 2x)")
    args = ap.parse_args()

    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(line_buffering=True)

    store = Store(Path(args.state_dir))
    corpora = load_corpora()
    n_items = {c: len(corpora[c]["items"]) for c in corpora}
    print(f"# corpora: {n_items}; models: {list(MODELS)}; k={K_SAMPLES}")
    print(f"# decoding: provider-default (never temperature/top_p/top_k)")
    if args.stage == "stage0":
        return stage0(store, corpora)
    return matrix(store, corpora, args)


if __name__ == "__main__":
    sys.exit(main())
