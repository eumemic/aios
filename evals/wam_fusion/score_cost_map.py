#!/usr/bin/env python3
"""Cost-map coding scorer pass — local pytest execution of every coding candidate.

Reads the matrix_coding_<model>.jsonl checkpoints produced by run_cost_map.py, runs
each candidate through the validated hidden-test oracle (coding_scorer.score_candidate
— the same scorer whose ground-truth self-check was 14/14 at corpus admission), and
appends verdicts to scores_coding_<model>.jsonl (checkpointed, resume-safe).

Also runs a ONE-PASS golden self-check per item (env-drift guard): if the oracle
cannot stand up on THIS machine today, the item is excluded for ALL models (fairness),
not scored as a fail for any of them.

$0: all local compute.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import json
import os
import sys
import threading
import time
from pathlib import Path

from coding_prompt import extract_sources
from coding_scorer import score_candidate, self_check

HERE = Path(__file__).parent
STATE_DIR = HERE / "results" / "cost_map_state"

from run_cost_map import MODELS  # single source of truth for the columns


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default=str(STATE_DIR))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--test-timeout-s", type=int, default=120)
    args = ap.parse_args()
    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(line_buffering=True)

    state = Path(args.state_dir)
    corpus = json.loads((HERE / "tasks" / "coding_selection.json").read_text())
    repo = corpus["repo"]
    items = {it["id"]: it for it in corpus["items"]}
    venv = os.environ.get("AIOS_VENV_PYTHON") or str(Path.home() / "code/aios/.venv/bin/python")

    # ── golden self-check (one pass; env-drift guard) ──
    gold_path = state / "scores_golden.json"
    golden = json.loads(gold_path.read_text()) if gold_path.exists() else {}
    todo_gold = [iid for iid in items if iid not in golden]
    if todo_gold:
        print(f"# golden self-check on {len(todo_gold)} items...", flush=True)
        lock = threading.Lock()

        def _gold(iid: str):
            r = self_check(items[iid], repo, venv_python=venv, timeout_s=args.test_timeout_s)
            with lock:
                golden[iid] = {"passed": r.passed, "skipped": r.skipped, "detail": r.detail[:120]}
                gold_path.write_text(json.dumps(golden, indent=1))
            if not r.passed:
                print(f"  GOLDEN-FAIL {iid}: {r.detail[:100]}", flush=True)

        with cf.ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(_gold, todo_gold))
    bad_items = sorted(iid for iid, g in golden.items() if not g["passed"])
    print(f"# golden: {sum(1 for g in golden.values() if g['passed'])}/{len(golden)} pass; "
          f"excluded items: {bad_items}", flush=True)

    # ── candidate scoring ──
    t0 = time.time()
    locks: dict[str, threading.Lock] = {m: threading.Lock() for m in MODELS}
    jobs = []
    for m in MODELS:
        recs = jsonl(state / f"matrix_coding_{m}.jsonl")
        done = {(r["item"], r["k"]) for r in jsonl(state / f"scores_coding_{m}.jsonl")}
        for r in recs:
            if r["status"] != "ok" or (r["item"], r["k"]) in done or r["item"] in bad_items:
                continue
            jobs.append((m, r))
    print(f"# scoring {len(jobs)} candidates ({args.workers} workers)...", flush=True)
    counter = {"done": 0}
    clock = threading.Lock()

    def _score(job):
        m, r = job
        item = items[r["item"]]
        sources = extract_sources(r.get("text"), item["src_files"])
        out = score_candidate(item, repo, sources, venv_python=venv,
                              timeout_s=args.test_timeout_s)
        rec = {
            "item": r["item"], "k": r["k"], "model": m,
            "passed": bool(out.passed),
            "skipped": bool(out.skipped),
            "applied": out.applied and bool(sources),
            "extraction_failed": not sources,
            "detail": out.detail[:140],
        }
        with locks[m]:
            with (state / f"scores_coding_{m}.jsonl").open("a") as f:
                f.write(json.dumps(rec) + "\n")
        with clock:
            counter["done"] += 1
            if counter["done"] % 50 == 0:
                print(f"PROGRESS scored={counter['done']}/{len(jobs)}", flush=True)

    with cf.ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(_score, jobs))
    print(f"SCORING COMPLETE: {counter['done']} candidates in {(time.time()-t0)/60:.1f} min",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
