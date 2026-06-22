"""Fixture corpus for the phantom-forward-ref scanner (``aios._drift.forward_refs``).

The scanner finds promissory issue references ("until #N lands") so the
phantom-ref canary can flag the ones whose issue has since closed. Its whole
value rests on *precision*: it must flag forward-looking phrases and must NOT
flag the far more common historical citation of the same number — otherwise the
canary files a false incident every week.

This is the guard-for-the-guard. The POSITIVE corpus is every promissory shape
seen in the tree (drawn from the real phantoms #332/#877/#878/#1127/#1128/#1131
and the open #1335). The NEGATIVE corpus is the citation/structural shapes that
share a number with a forward-ref but must stay silent — the exact false-positive
modes the design rejected. Extend a pattern only by first extending a corpus here.
"""

from __future__ import annotations

from pathlib import Path

from aios._drift.forward_refs import scan_text, scan_tree

# ── Positive: promissory phrases the scanner MUST flag ───────────────────────
# (source line, the issue numbers it must yield)
_POSITIVE: list[tuple[str, set[int]]] = [
    # connective whose subject is the ref — the dominant shape
    ("# When #332 lands (bash writes persisted), drop the bash-bypass clause.", {332}),
    ("materialized env-var set until #877's drift handling lands. Layer 2 ...", {877}),
    # the #878 case the literal verb-list missed: a domain verb ("redirects")
    ("the proxy serves hostname egress only. Once #878 redirects host traffic here", {878}),
    # two-issue slash form — findall must enumerate BOTH
    ("children, and — once #1127/#1128 land their call sites — session→session", {1127, 1128}),
    ("legacy run path until #1131 retires it). Both halves are partial-indexed", {1131}),
    ("    # ── live FK half (until #1131) ──────────────────────────────────", {1131}),
    # connective-free idioms
    ("    # TODO(#1131): this point-reads sessions.parent_run_id directly.", {1131}),
    ("    No in-tree consumer yet (the read site is deferred to #1335) — a seam.", {1335}),
    ("    # blocked on #1234 — cannot land the read path until then.", {1234}),
]

# ── Negative: shapes that share a number with a forward-ref but must stay silent
_NEGATIVE: list[str] = [
    # historical citation of a (closed) issue — the dangerous false-positive class
    "applies the #1132 cap to bound the recovery sweep (see the #1132 discussion).",
    "the #1131-proof structural invariant holds across the dual-write era.",
    "blocking the host is #878's job once the proxy owns egress.",  # apposition, no connective
    "this mirrors the git_proxy precedent. (#878)",  # bare parenthetical
    "recycle-on-rotation are follow-ups (#877/#878).",  # parenthetical list
    "deduped once (#725) per the dedupe pass, not per call.",  # temporal adverb + paren
    "on master today (#1131 unmerged, dual-write era) neither half alone is complete.",
    "after #205 was removed the 'connectors' queue subscription is dead.",  # wrong connective
    "When ``connection_ids`` is non-``None`` (#350) the stream filters.",  # ref not the subject
    "needs CLI; tracked in #1446 (memory-stores group)",  # the NEEDS_CLI_TRACKED citation form
    "see #1294 for the http_request truncation flag.",  # bare 'see #N'
    # structural ordinals — never matched: the ref isn't the immediate subject of a
    # forward connective. `until phase #200` doesn't match even at a high number
    # (the connective isn't directly before `#N`), so no structural exclusion needed.
    "invariant #6 (gapless seq per session) — every append locks the row.",
    "Layer #2's triggers stay off the connection tables.",
    "until phase #200 lands, the ladder stays linear.",
    # below the issue-number floor
    "the placeholder stayed until #42 lands, long ago.",
]


def _numbers(text: str) -> set[int]:
    return {f.number for f in scan_text("sample.py", text)}


def test_positive_corpus_is_flagged() -> None:
    for text, expected in _POSITIVE:
        assert _numbers(text) == expected, f"expected {expected} from: {text!r}"


def test_negative_corpus_is_silent() -> None:
    for text in _NEGATIVE:
        found = scan_text("sample.py", text)
        assert found == [], f"false positive {[(f.number, f.text) for f in found]} from: {text!r}"


def test_one_line_two_patterns_dedupes() -> None:
    """A line matched by both ``TODO(#N)`` and ``When #N`` yields one finding."""
    line = "# TODO(#1131): point-read parent_run_id. When #1131 subsumes it, drop this."
    findings = scan_text("sample.py", line)
    assert [f.number for f in findings] == [1131], findings


def test_finding_carries_path_line_and_text() -> None:
    findings = scan_text("src/aios/foo.py", "x = 1\nuntil #1234 lands here\n")
    assert len(findings) == 1
    f = findings[0]
    assert (f.path, f.line, f.number) == ("src/aios/foo.py", 2, 1234)
    assert f.text == "until #1234 lands here"


def test_scan_tree_excludes_the_scanner_package(tmp_path: Path) -> None:
    """``scan_tree`` skips ``_drift/`` so the scanner's own example phrases — and
    this kind of phrasing in any future drift tooling — never self-flag."""
    (tmp_path / "_drift").mkdir()
    (tmp_path / "_drift" / "scanner.py").write_text('# example: "until #999 lands"\n')
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "real.py").write_text("# until #1234 lands, keep the shim\n")

    numbers = {f.number for f in scan_tree(tmp_path)}
    assert numbers == {1234}, numbers
