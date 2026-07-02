"""Shared-pool arm derivation for the execution-verified-selection eval (P0).

Design of record: eumemic-company `architecture/execution-verified-selection-eval.md`.

Per task we draw ONE pool of N=4 independent candidate patches from the best single
model, then derive all four arms from that single pool so every pairwise comparison is
matched-compute *by construction* (§2):

  (a) best-single 1-shot        — candidate index 0 (pre-registered, not post-hoc)
  (b) N-shot + random-pick      — seeded-uniform pick among the N (the compute control)
  (c) N-shot + execution pick   — run the hidden suite per candidate; keep a passer
                                  (tie-break: fewest changed lines vs base)
  (d) N-shot + LLM-judge pick   — one judge ranks the candidates; sees issue + diffs,
                                  NEVER the tests

This module is the PURE part (no I/O, no network): picks, diffs, the leak canary, and
the judge prompt/parse. Everything here is deterministic given its inputs, so it is
unit-testable without spend (test_selection.py).
"""

from __future__ import annotations

import difflib
import random
import re
from dataclasses import dataclass

# ── pre-registered constants (recorded into the results payload) ─────────────
N_CANDIDATES = 4
GLOBAL_SEED = 20260702  # arm-(b) pick seed; per-item stream = f"{seed}:{item_id}"

# A candidate that produced no usable source can never win a fewest-changed-lines
# tie-break (and scores fail regardless).
_NO_SOURCE_CHANGED_LINES = 10**9


# ── arm (a): index-0 ─────────────────────────────────────────────────────────
def pick_index0(n: int) -> int:
    """Pre-registered production status quo: the first sample drawn."""
    if n < 1:
        raise ValueError("empty pool")
    return 0


# ── arm (b): seeded-random ───────────────────────────────────────────────────
def pick_seeded_random(item_id: str, n: int, seed: int = GLOBAL_SEED) -> int:
    """Uniform pick among the pool, deterministic per (seed, item): the compute
    control. The seed is pre-registered and recorded; per-item streams keep one
    item's pool size from perturbing another item's pick."""
    if n < 1:
        raise ValueError("empty pool")
    return random.Random(f"{seed}:{item_id}").randrange(n)


# ── arm (c): execution-verified ──────────────────────────────────────────────
def changed_lines(base_sources: dict[str, str], cand_sources: dict[str, str]) -> int:
    """Total +/- line count of the candidate's diff vs base, across src files.

    A candidate missing any required file gets a sentinel so it never wins the
    tie-break. This is the pre-registered tie-break metric (smallest diff = the
    most conservative passer)."""
    if not cand_sources or any(not (cand_sources.get(f) or "").strip() for f in base_sources):
        return _NO_SOURCE_CHANGED_LINES
    total = 0
    for path, base in base_sources.items():
        cand = cand_sources.get(path) or ""
        for line in difflib.unified_diff(
            base.splitlines(), cand.splitlines(), lineterm="", n=0
        ):
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                total += 1
    return total


def pick_exec_verified(
    passed: list[bool], n_changed_lines: list[int]
) -> tuple[int, str]:
    """Keep a passer; tie-break fewest changed lines, then lowest index.

    No passer → the pool fails regardless of pick; we still return a deterministic
    pick (fewest changed lines) so the arm's per-item record is complete."""
    if not passed or len(passed) != len(n_changed_lines):
        raise ValueError("pool score vectors malformed")
    idxs = range(len(passed))
    passers = [i for i in idxs if passed[i]]
    if passers:
        best = min(passers, key=lambda i: (n_changed_lines[i], i))
        return best, f"passer {best} of {passers} (changed_lines={n_changed_lines[best]})"
    best = min(idxs, key=lambda i: (n_changed_lines[i], i))
    return best, f"NO PASSER; deterministic fallback {best}"


# ── arm (d): LLM judge ───────────────────────────────────────────────────────
JUDGE_SYSTEM = (
    "You are an expert code reviewer selecting the best of several candidate patches "
    "for a bug-fix task. You are given the task and each candidate's diff against the "
    "current source. You canNOT run tests and you are NOT shown any tests — judge by "
    "reading the code: correctness for the stated task, edge cases, and minimality. "
    "Think briefly, then reply with a final line of EXACTLY the form:\n"
    "WINNER: <k>\n"
    "where <k> is the number of the best candidate."
)


def build_judge_prompt(task: str, diffs: list[str]) -> str:
    """Issue + per-candidate diffs. NEVER include tests (the design's FATAL leak)."""
    parts = [f"TASK:\n{task}\n", f"There are {len(diffs)} candidate patches (numbered 0..{len(diffs) - 1})."]
    for i, d in enumerate(diffs):
        shown = d if d.strip() else "(candidate produced no usable patch)"
        parts.append(f"\n===== CANDIDATE {i} DIFF =====\n{shown}")
    parts.append(
        f"\nPick the single best candidate for the task. End with EXACTLY: WINNER: <k> (0..{len(diffs) - 1})."
    )
    return "\n".join(parts)


def unified_diff_text(base_sources: dict[str, str], cand_sources: dict[str, str]) -> str:
    """The candidate's diff vs base (what the judge sees — src files only)."""
    chunks = []
    for path, base in base_sources.items():
        cand = cand_sources.get(path) or ""
        diff = "\n".join(
            difflib.unified_diff(
                base.splitlines(), cand.splitlines(),
                fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="",
            )
        )
        if diff:
            chunks.append(diff)
    return "\n".join(chunks)


_WINNER = re.compile(r"WINNER:\s*(\d+)")


def parse_judge_pick(text: str | None, n: int) -> tuple[int | None, str]:
    """Last 'WINNER: k' line wins (models sometimes restate). None on parse failure
    or out-of-range (caller falls back to index 0 and flags integrity)."""
    if not text:
        return None, "judge returned no text"
    hits = _WINNER.findall(text)
    if not hits:
        return None, "no WINNER: line"
    k = int(hits[-1])
    if not (0 <= k < n):
        return None, f"WINNER {k} out of range 0..{n - 1}"
    return k, f"judge picked {k}"


# ── leak canary (§3: the eval's FATAL-class fraud) ───────────────────────────
_TEST_DEF = re.compile(r"^\s*(?:async\s+)?def\s+(test_\w+)", re.MULTILINE)
_TEST_CLASS = re.compile(r"^\s*class\s+(Test\w+)", re.MULTILINE)
# vitest/jest oracle identifiers: it("...")/test("...")/describe("...") string titles.
_TS_TEST_TITLE = re.compile(r"""(?:\bit|\btest|\bdescribe)\s*\(\s*['"`]([^'"`\n]{8,80})['"`]""")


def canary_tokens(
    test_paths: list[str],
    oracle_test_contents: list[str],
    base_sources: dict[str, str],
    task: str,
) -> set[str]:
    """Tokens whose presence in a candidate transcript/diff means the hidden tests
    leaked: the hidden-test file paths (+ basenames) and the test identifiers
    defined in the oracle.

    Tokens that already appear in the model's INPUT (the base sources or the task
    text) are excluded — the model legitimately saw those, so their presence in the
    output is not evidence of a leak (e.g. an issue body that says "add
    tests/unit/test_x.py", or a test name that collides with a source symbol)."""
    tokens: set[str] = set()
    for p in test_paths:
        tokens.add(p)
        base = p.rsplit("/", 1)[-1]
        if base:
            tokens.add(base)
    for content in oracle_test_contents:
        tokens.update(_TEST_DEF.findall(content or ""))
        tokens.update(_TEST_CLASS.findall(content or ""))
        tokens.update(_TS_TEST_TITLE.findall(content or ""))
    visible = task + "".join(base_sources.values())
    return {t for t in tokens if t not in visible}


def leak_scan(candidate_text: str | None, tokens: set[str]) -> list[str]:
    """The canary check: grep the candidate's full transcript (which contains its
    diff — the fenced file blocks are extracted from this text) for hidden-test
    tokens. Any hit FAILS the item."""
    if not candidate_text:
        return []
    return sorted(t for t in tokens if t in candidate_text)


# ── one-sided exact sign test (the §4 primary) ───────────────────────────────
@dataclass(frozen=True)
class SignTestResult:
    name: str
    n_discordant: int
    c_only: int  # discordants favoring the treatment (arm c)
    b_only: int  # discordants favoring the baseline — oracle-flakiness indicator
    p_one_sided: float


def one_sided_sign_test(name: str, baseline: list[bool], treatment: list[bool]) -> SignTestResult:
    """Exact one-sided McNemar/sign test on the discordant pairs, testing
    H1: treatment > baseline. With a shared pool and a sound oracle the treatment
    (execution-verified pick) dominates item-wise, so all discordants should favor
    it; baseline-only wins can arise only from oracle flakiness (tracked)."""
    import math

    if len(baseline) != len(treatment):
        raise ValueError("paired arms must have equal length")
    c_only = sum(1 for b, t in zip(baseline, treatment) if t and not b)
    b_only = sum(1 for b, t in zip(baseline, treatment) if b and not t)
    n = c_only + b_only
    if n == 0:
        return SignTestResult(name, 0, 0, 0, 1.0)
    p = sum(math.comb(n, k) for k in range(c_only, n + 1)) * (0.5**n)
    return SignTestResult(name, n, c_only, b_only, min(1.0, p))
