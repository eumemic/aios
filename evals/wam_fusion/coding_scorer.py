"""Hidden-test pass@1 scorer for the CODING tier — sandboxed, robust, isolatable.

The Phase A scorer compared a model's checkable answer to a canonical value. Coding
needs a held-out ORACLE: a real merged aios PR's test, run against the candidate's
patch. pass@1 = the held-out test goes green.

PER-ITEM PROTOCOL (robust to a flaky/non-applying/hostile-ish candidate):
  1. Materialize aios at the item's ``base_parent_sha`` into a fresh temp dir via
     ``git archive`` (no worktree lock, no mutation of the canonical clone).
  2. Overwrite the item's ``src_files`` with the CANDIDATE's produced source.
  3. Drop in the held-out ``test_files`` from the item's ``merge_sha`` (the oracle —
     the candidate never sees or edits these).
  4. Run ONLY those test files with ``pytest``, in a subprocess, with:
       * ``PYTHONPATH=<tmp>/src`` FIRST so the patched tree shadows any editable
         install of aios (the .pth trap the corpus recon flagged — without this the
         test silently imports the canonical checkout and scores the WRONG code), and
       * a per-item wall-clock timeout (a candidate that hangs a test = fail, not a
         wedge of the whole run).
  4. Classify: green => pass; red/collection-error => fail; the test ENV itself failing
     to stand up (import error in the oracle, missing dep) => SKIP (logged, excluded
     from the paired sample — one bad item must never wedge or bias the run).

This deliberately runs candidate code. Containment here is process-level (subprocess +
timeout + temp dir), NOT a security boundary — the corpus is OUR repo's real PRs and the
candidates are model-written patches to known-small source files, so the threat is a
runaway/expensive test, which the timeout bounds. A true adversarial-code sandbox (the
aios microVM) is overkill for this and out of scope.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScoreOutcome:
    item_id: str
    passed: bool
    skipped: bool  # env couldn't stand up — exclude from the paired sample
    detail: str  # short diagnosis
    applied: bool  # did the candidate produce usable source for all src_files?


def _git(repo: str, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def materialize_base(repo: str, sha: str, dest: Path) -> None:
    """Export the repo tree at ``sha`` into ``dest`` (no .git, no worktree lock)."""
    dest.mkdir(parents=True, exist_ok=True)
    # git archive <sha> | tar -x -C dest  — a clean snapshot of the tree at base.
    archive = subprocess.run(
        ["git", "-C", repo, "archive", sha],
        capture_output=True,
        timeout=120,
    )
    if archive.returncode != 0:
        raise RuntimeError(f"git archive {sha[:10]} failed: {archive.stderr.decode()[:200]}")
    untar = subprocess.run(
        ["tar", "-x", "-C", str(dest)],
        input=archive.stdout,
        capture_output=True,
        timeout=120,
    )
    if untar.returncode != 0:
        raise RuntimeError(f"tar extract failed: {untar.stderr.decode()[:200]}")


def file_at(repo: str, sha: str, path: str) -> str | None:
    """Return the content of ``path`` at ``sha`` (the held-out test), or None."""
    p = _git(repo, "show", f"{sha}:{path}")
    return p.stdout if p.returncode == 0 else None


def source_at_base(repo: str, base_sha: str, src_files: list[str]) -> dict[str, str]:
    """The source file(s) at base — what the candidate is shown and asked to correct."""
    out: dict[str, str] = {}
    for f in src_files:
        c = file_at(repo, base_sha, f)
        if c is not None:
            out[f] = c
    return out


def score_candidate(
    item: dict,
    repo: str,
    candidate_sources: dict[str, str],
    *,
    timeout_s: int = 120,
    venv_python: str | None = None,
    oracle_verified: bool = False,
) -> ScoreOutcome:
    """Run the held-out test against the candidate's produced source. pass@1 = green.

    ``candidate_sources`` maps each ``src_file`` path -> the candidate's full file
    content. A missing/empty file for any required src_file => not applied => fail
    (the candidate didn't produce a usable patch).

    ``oracle_verified``: pass True when a golden self-check has ALREADY proven (on
    this machine, with the true merged sources) that this item's oracle test stands
    up. Under that guarantee, a collection/import error on a candidate run cannot be
    the environment's fault — it is the candidate breaking the module interface the
    oracle imports (e.g. omitting a helper/constant the test imports by name) — so
    it is classified FAIL, not ENV-SKIP. Without the guarantee (default), the old
    conservative ENV-SKIP behavior is kept. (Found 2026-07-07: ENV-SKIP was
    laundering real interface failures into holes, differentially by model.)
    """
    iid = item["id"]
    src_files = item["src_files"]
    test_files = item["test_files"]
    base = item["base_parent_sha"]
    merge = item["merge_sha"]
    python = venv_python or os.environ.get("AIOS_VENV_PYTHON") or "python3"

    # 1. candidate must have produced every required source file
    missing = [f for f in src_files if not (candidate_sources.get(f) or "").strip()]
    if missing:
        return ScoreOutcome(iid, False, False, f"candidate omitted {missing}", applied=False)

    tmp = Path(tempfile.mkdtemp(prefix=f"wamb_{iid}_"))
    try:
        materialize_base(repo, base, tmp)
        # 2. overwrite source files with the candidate's version
        for f, content in candidate_sources.items():
            if f not in src_files:
                continue
            target = tmp / f
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        # 3. drop in the held-out test file(s) from the merge commit (the oracle)
        for tf in test_files:
            content = file_at(repo, merge, tf)
            if content is None:
                return ScoreOutcome(
                    iid, False, True, f"oracle test {tf} missing at merge", applied=True
                )
            target = tmp / tf
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        # 4. run ONLY the held-out test(s), patched tree FIRST on sys.path
        env = dict(os.environ)
        env["PYTHONPATH"] = str(tmp / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            proc = subprocess.run(
                [python, "-m", "pytest", *test_files, "-q", "-p", "no:cacheprovider"],
                cwd=str(tmp),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return ScoreOutcome(iid, False, False, f"test timed out >{timeout_s}s", applied=True)

        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        tail = out.strip().splitlines()[-1] if out.strip() else ""
        # pytest exit codes: 0 pass, 1 tests failed, 2 interrupted, 3 internal,
        # 4 usage, 5 no tests collected. A collection/import error in the ORACLE
        # (not the candidate) means the env couldn't stand up -> SKIP it.
        if proc.returncode == 0:
            return ScoreOutcome(iid, True, False, f"PASS ({tail})", applied=True)
        env_broke = (
            (
                "ModuleNotFoundError" in out
                or "ImportError" in out
                or "errors during collection" in out
                or proc.returncode in (3, 4, 5)
            )
            and "FAILED" not in out
            and "assert" not in out.lower()
        )
        if env_broke:
            if oracle_verified:
                # golden proved the oracle stands up with true sources => the
                # candidate broke the interface the test imports. A real failure.
                return ScoreOutcome(
                    iid, False, False, f"INTERFACE-FAIL ({tail[:80]})", applied=True
                )
            return ScoreOutcome(iid, False, True, f"ENV-SKIP ({tail[:80]})", applied=True)
        return ScoreOutcome(iid, False, False, f"FAIL ({tail[:80]})", applied=True)
    except Exception as exc:  # any infra hiccup on this item => SKIP, never wedge
        return ScoreOutcome(
            iid, False, True, f"infra: {type(exc).__name__}: {exc}"[:120], applied=True
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def self_check(item: dict, repo: str, **kw) -> ScoreOutcome:
    """Sanity oracle: score the GROUND-TRUTH merge-commit source against the held-out
    test. MUST pass — if it doesn't, the item's provenance (base/merge SHAs, file
    paths, or test isolatability) is wrong and the item must be dropped before any
    model is scored on it. This is the harness's own correctness guard."""
    gt = {f: file_at(repo, item["merge_sha"], f) or "" for f in item["src_files"]}
    return score_candidate(item, repo, gt, **kw)


if __name__ == "__main__":
    # Self-check every corpus item against its ground truth (no model calls).
    import sys

    corpus = json.loads(
        Path(
            sys.argv[1]
            if len(sys.argv) > 1
            else Path(__file__).parent / "tasks" / "coding_aios.json"
        ).read_text()
    )
    repo = corpus["repo"]
    py = os.environ.get("AIOS_VENV_PYTHON", "python3")
    ok = bad = 0
    for it in corpus["items"]:
        r = self_check(it, repo, venv_python=py, timeout_s=120)
        flag = "OK " if (r.passed) else ("SKIP" if r.skipped else "BAD ")
        if r.passed:
            ok += 1
        else:
            bad += 1
        print(f"  [{flag}] {it['id']} pr#{it['pr']} {r.detail}")
    print(f"\nground-truth self-check: {ok} pass, {bad} not-pass (BAD/SKIP items must be dropped)")
    sys.exit(1 if bad else 0)
