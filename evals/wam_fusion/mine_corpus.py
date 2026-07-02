#!/usr/bin/env python3
"""P1 corpus miner: extend the hidden-test coding corpus 9 → ~40 single-file items.

Recipe (design §3, identical to the Phase B corpus): mine MERGED PRs where
  * exactly ONE source file changed (single-file-only, scorer comparability), and
  * >= one tests/unit/ test file changed (the held-out oracle), and
  * nothing else changed (so base-tree + merged-src == the golden tree for the oracle),
  * a linked issue with a real body exists (task = issue title + body, verbatim —
    the pre-PR knowledge a dev_pipeline dispatch would see).

Provenance per item: merge_sha = the PR's (squash-)merge commit; base_parent_sha =
merge_sha^ (the pre-PR repo snapshot). Oracle = the golden PR's test files AT merge_sha,
held out from generation.

Intake stratification (§3): each item is tagged
  * multi-attempt — >=2 PRs reference the linked issue (re-dispatch/retry signature),
  * thin-spec    — issue body < 500 chars,
  * standard     — the rest,
and the achieved per-stratum proportions are reported (the headline of the eval is the
intake-weighted mean; the run harness reports per-stratum).

EVERY candidate item is independently soundness-verified before admission (§3):
golden patch passes the hidden suite (run 2x — a flip = flake = reject) and the empty
patch (base source unchanged) FAILS it (kills vacuous/non-discriminating tests).
Rejects are logged with reasons in the mining report.

Usage:
  python3 mine_corpus.py --repo-path ~/code/aios --limit 800 --target 45 \
      --out tasks/coding_selection.json
Requires: gh CLI authed, AIOS_VENV_PYTHON pointing at a venv that can run the repo's
unit tests. No model calls, no spend.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from coding_scorer import score_candidate, self_check, source_at_base

HERE = Path(__file__).parent

CLOSES = re.compile(r"(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)[:\s]+#(\d+)", re.IGNORECASE)
REFS = re.compile(r"#(\d+)")
THIN_SPEC_CHARS = 500


def sh(*args: str, timeout: int = 120) -> str:
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:4])} -> {p.returncode}: {p.stderr[:200]}")
    return p.stdout


def gh_json(*args: str) -> object:
    return json.loads(sh("gh", *args, timeout=180))


def list_merged_prs(repo: str, limit: int) -> list[dict]:
    return gh_json(
        "pr", "list", "--repo", repo, "--state", "merged", "--limit", str(limit),
        "--json", "number,title,body,files,mergeCommit,mergedAt",
    )


def list_all_pr_bodies(repo: str, limit: int) -> list[dict]:
    """number+body+state of ALL PRs (merged AND closed-unmerged) — the multi-attempt
    stratum needs the failed attempts too."""
    return gh_json(
        "pr", "list", "--repo", repo, "--state", "all", "--limit", str(limit),
        "--json", "number,body,state",
    )


def single_file_shape(files: list[dict]) -> tuple[str, list[str]] | None:
    """(src_file, test_files) if the PR is exactly 1 src file + >=1 tests/unit file,
    plus at most IGNORABLE extras; None otherwise.

    Ignorable extras = changed files the item neither applies nor uses as oracle:
    docs (*.md) and non-unit test files. This is SOUND because admission (§3) still
    requires the golden tree — base + merged src + unit oracle ONLY — to pass: if the
    oracle secretly depends on an ignored file (a conftest, a fixture), golden fails
    and the item is rejected."""
    paths = [f["path"] for f in files]
    src = [p for p in paths if p.startswith("src/") and p.endswith(".py")]
    tests = [p for p in paths if p.startswith("tests/unit/") and p.endswith(".py")]
    ignorable = [
        p for p in paths
        if p not in src and p not in tests
        and (p.endswith(".md") or (p.startswith("tests/") and p.endswith(".py")))
    ]
    other = [p for p in paths if p not in src and p not in tests and p not in ignorable]
    if len(src) == 1 and len(tests) >= 1 and not other:
        return src[0], sorted(tests)
    return None


def linked_issue(body: str | None) -> int | None:
    if not body:
        return None
    m = CLOSES.search(body)
    return int(m.group(1)) if m else None


def issue_attempt_counts(all_prs: list[dict]) -> dict[int, int]:
    """issue# -> number of PRs whose body claims to close/fix it (any PR state).
    Approximate by the closing-keyword convention (autodev/dev_pipeline PRs always
    carry it); recorded as the stratum method."""
    counts: dict[int, int] = {}
    for pr in all_prs:
        n = linked_issue(pr.get("body"))
        if n is not None:
            counts[n] = counts.get(n, 0) + 1
    return counts


def stratum_of(issue_body: str, attempts: int) -> str:
    if attempts >= 2:
        return "multi-attempt"
    if len(issue_body or "") < THIN_SPEC_CHARS:
        return "thin-spec"
    return "standard"


def git(repo_path: str, *args: str) -> str:
    return sh("git", "-C", repo_path, *args)


def verify_soundness(item: dict, repo_path: str, test_timeout_s: int) -> tuple[bool, str]:
    """§3 per-item admission gate: golden passes 2x (flake check); base source fails."""
    kw = {"venv_python": os.environ.get("AIOS_VENV_PYTHON"), "timeout_s": test_timeout_s}
    g1 = self_check(item, repo_path, **kw)
    if not g1.passed:
        return False, f"golden-fail: {g1.detail[:90]}"
    g2 = self_check(item, repo_path, **kw)
    if not g2.passed:
        return False, f"golden-FLAKE (pass then {g2.detail[:80]})"
    base_src = source_at_base(repo_path, item["base_parent_sha"], item["src_files"])
    b = score_candidate(item, repo_path, base_src, **kw)
    if b.passed:
        return False, "vacuous: base (empty patch) PASSES the oracle"
    # Golden passed 2x, so the env is PROVEN able to stand the oracle up; a base run
    # classified env-broke (e.g. the merged test imports a symbol the fix adds ->
    # collection error) is a DISCRIMINATING failure of the unfixed tree, not a broken
    # env. Same semantics as run_selection.soundness_preflight (not base_passed).
    mode = "env-broke-at-base (collection/import)" if b.skipped else "test-fail"
    return True, f"golden 2x pass; base fails [{mode}] ({b.detail[:60]})"


def main() -> int:
    ap = argparse.ArgumentParser(description="Mine merged PRs into hidden-test eval items.")
    ap.add_argument("--repo", default="eumemic/aios")
    ap.add_argument("--repo-path", default=str(Path.home() / "code" / "aios"))
    ap.add_argument("--limit", type=int, default=900, help="how many merged PRs to scan")
    ap.add_argument("--target", type=int, default=45, help="stop once this many items are SOUND")
    ap.add_argument("--test-timeout-s", type=int, default=180)
    ap.add_argument("--min-issue-body", type=int, default=80)
    ap.add_argument("--exclude-prs", default="", help="comma-separated PR numbers to skip")
    ap.add_argument("--seed-corpus", default="coding_aios.json",
                    help="existing corpus whose single-file items are re-admitted first")
    ap.add_argument("--out", default=str(HERE / "tasks" / "coding_selection.json"))
    ap.add_argument("--report", default=str(HERE / "tasks" / "coding_selection_mining_report.json"))
    ap.add_argument("--id-prefix", default="s")
    args = ap.parse_args()

    repo_path = str(Path(args.repo_path).expanduser())
    exclude = {int(x) for x in args.exclude_prs.split(",") if x.strip()}

    print(f"# scanning merged PRs in {args.repo} (limit {args.limit}) ...")
    merged = list_merged_prs(args.repo, args.limit)
    all_prs = list_all_pr_bodies(args.repo, max(args.limit * 2, 1000))
    attempts = issue_attempt_counts(all_prs)
    print(f"# {len(merged)} merged PRs fetched; {len(all_prs)} total PRs for attempt-counts")

    # ── seed: re-admit the existing single-file items through the SAME gate ──
    accepted: list[dict] = []
    rejected: list[dict] = []
    seen_issue_src: set[tuple[int | None, str]] = set()
    seed_path = HERE / "tasks" / args.seed_corpus
    if seed_path.exists():
        seed = json.loads(seed_path.read_text())
        for it in seed["items"]:
            if len(it["src_files"]) != 1:
                continue
            it = dict(it)
            it.setdefault("task_source", "distilled-2026-06-30")
            it.setdefault("runner", "pytest")
            issue_body = ""
            if it.get("issue"):
                try:
                    issue_body = gh_json(
                        "issue", "view", str(it["issue"]), "--repo", args.repo, "--json", "body"
                    ).get("body") or ""
                except RuntimeError:
                    issue_body = it.get("task", "")
            it["stratum"] = stratum_of(issue_body, attempts.get(it.get("issue"), 1))
            ok, why = verify_soundness(it, repo_path, args.test_timeout_s)
            (accepted if ok else rejected).append({**it, "soundness": why} if ok else
                                                  {"id": it["id"], "pr": it["pr"], "reason": why})
            print(f"  [seed {'OK ' if ok else 'REJ'}] {it['id']} pr#{it['pr']} {why[:70]}")
            seen_issue_src.add((it.get("issue"), it["src_files"][0]))
            exclude.add(it["pr"])

    # ── mine new ones (newest first = closest to today's intake distribution) ──
    idx = 0
    skip_counts: dict[str, int] = {}

    def skip(reason: str) -> None:
        skip_counts[reason] = skip_counts.get(reason, 0) + 1

    for pr in merged:
        if len(accepted) >= args.target:
            break
        num = pr["number"]
        if num in exclude:
            skip("already-in-seed/excluded")
            continue
        shape = single_file_shape(pr.get("files") or [])
        if not shape:
            skip("shape (not exactly 1 src + tests/unit)")
            continue
        src_file, test_files = shape
        issue_no = linked_issue(pr.get("body"))
        if issue_no is None:
            # Fall back to GitHub's formal linkage (UI-linked "Development" issues
            # don't need a body keyword).
            try:
                refs = gh_json("pr", "view", str(num), "--repo", args.repo,
                               "--json", "closingIssuesReferences")
                nums = [x["number"] for x in refs.get("closingIssuesReferences") or []]
                issue_no = nums[0] if nums else None
            except RuntimeError:
                issue_no = None
        if issue_no is None:
            skip("no linked issue")
            continue
        if (issue_no, src_file) in seen_issue_src:
            skip("duplicate issue+src")
            continue
        merge_sha = (pr.get("mergeCommit") or {}).get("oid")
        if not merge_sha:
            skip("no merge commit")
            continue
        try:
            issue = gh_json("issue", "view", str(issue_no), "--repo", args.repo,
                            "--json", "title,body")
        except RuntimeError:
            skip("issue fetch failed")
            continue
        body = issue.get("body") or ""
        if len(body) < args.min_issue_body:
            skip("issue body too short to be a task")
            continue
        try:
            git(repo_path, "cat-file", "-e", merge_sha)
        except RuntimeError:
            skip("merge sha not in local clone")
            continue
        base_parent = git(repo_path, "rev-parse", f"{merge_sha}^").strip()
        idx += 1
        item = {
            "id": f"{args.id_prefix}{idx:02d}",
            "pr": num,
            "issue": issue_no,
            "task": f"{issue.get('title', '')}\n\n{body}".strip(),
            "task_source": "issue-title+body-verbatim",
            "base_parent_sha": base_parent,
            "merge_sha": merge_sha,
            "src_files": [src_file],
            "test_files": test_files,
            "headroom": "UNKNOWN",
            "stratum": stratum_of(body, attempts.get(issue_no, 1)),
            "runner": "pytest",
        }
        ok, why = verify_soundness(item, repo_path, args.test_timeout_s)
        if ok:
            accepted.append({**item, "soundness": why})
            seen_issue_src.add((issue_no, src_file))
            print(f"  [OK  {len(accepted):>2}] {item['id']} pr#{num} ({item['stratum']}) {src_file}")
        else:
            rejected.append({"id": item["id"], "pr": num, "issue": issue_no,
                             "src": src_file, "reason": why})
            print(f"  [REJ    ] pr#{num} {why[:80]}")

    # ── emit corpus + report ──
    strata: dict[str, int] = {}
    for it in accepted:
        strata[it["stratum"]] = strata.get(it["stratum"], 0) + 1
    corpus = {
        "repo": repo_path,
        "recipe": {
            "source": f"merged {args.repo} PRs, newest-first",
            "task": "linked issue title+body verbatim (seed items keep their distilled tasks)",
            "oracle": "golden PR's tests/unit files at merge_sha, held out",
            "provenance": "merge_sha = PR merge commit; base_parent_sha = merge_sha^",
            "admission": "golden passes hidden suite 2x (flake=reject); base source (empty patch) fails",
            "strata_method": (
                f"multi-attempt: >=2 PRs body-reference the issue via closing keywords; "
                f"thin-spec: issue body < {THIN_SPEC_CHARS} chars; else standard"
            ),
        },
        "items": accepted,
    }
    Path(args.out).write_text(json.dumps(corpus, indent=1))
    report = {
        "accepted": len(accepted),
        "rejected_on_soundness": [r for r in rejected if "shape" not in r.get("reason", "")],
        "strata": strata,
        "scan_skip_counts": skip_counts,
        "merged_prs_scanned": len(merged),
    }
    Path(args.report).write_text(json.dumps(report, indent=1))
    print(f"\n# ACCEPTED {len(accepted)} items  strata={strata}")
    print(f"# rejected on soundness: {len(rejected)}  (details in {args.report})")
    print(f"# wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
