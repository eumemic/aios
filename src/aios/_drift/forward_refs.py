"""Offline scanner for phantom forward-looking issue references.

A *forward-looking* reference promises future work gated on another issue —
``"When #332 lands"``, ``"until #1131 retires it"``, ``"deferred to #1335"``,
``"TODO(#1131)"``. When that issue later closes or merges (often in a *different*
PR than the one that wrote the comment), the promise rots: the gated work is done
or abandoned, yet the prose still says ``"until it lands"``. The signal now lies.

This module finds the *candidates* — the promissory phrases — purely offline. It
does NOT decide whether a referenced issue is actually closed; that needs the
GitHub API and lives in ``.github/workflows/phantom-ref-canary.yml``, so the unit
suite stays offline. The scanner's job is precision against the dangerous class: a
*closed* issue cited **historically** (``"the #1132 cap"``, an ``"#1131-proof"``
invariant, a bare ``"(#878)"`` citation) must NOT be a candidate, or the canary
would file a false incident about it.

Precision comes from keying on the **phrase, never the bare number**: one issue
can be a forward-phantom on one line (``"until #1131 retires it"``) and a
legitimate historical citation on another (``"#1131-proof"``) in the same file.
The discriminator is that a forward reference makes the issue the grammatical
*subject* of a forward connective (``until``/``once``/``when`` immediately followed
by the ref), or uses one of three connective-free promissory idioms — whereas a
citation puts the number in parentheses or apposition.

Scope, stated so nobody mistakes this for a total drift solution: it catches
"forward-connective bound to ``#N``" plus ``TODO(#N)`` / ``deferred to #N`` /
``blocked on #N`` only. Numberless forward-state prose (``"once the prune-cron
exemption lands"``) and negated-verb idioms (``"(#953) deliberately NOT shipped"``)
are out of scope by design — they are ambiguous, and widening recall trades false
negatives for false alarms. Extend the patterns only alongside a fixture.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Issue numbers below this floor are structural ordinals ("invariant #6",
# "step #3", "Layer #2"), never issue refs. The repo's tracked issues sit far
# above it; no real forward-ref points below 100.
_MIN_ISSUE_NUMBER = 100

# (1) A forward connective whose grammatical subject IS the issue ref: the ref
# immediately follows ``until``/``once``/``when`` (optionally a ``/``-list and/or a
# possessive). This directness — not a verb list — is what separates a promissory
# clause from a bare parenthetical citation: ``"Once #878 redirects ... here"``
# matches, while ``"(#878)"``, ``"the #1132 cap"``, and ``"deduped once (#725)
# per ..."`` do not.
_CONNECTIVE_SUBJECT = re.compile(
    r"\b(?:until|once|when)\s+#\d+(?:/#\d+)*(?:'s)?\b",
    re.IGNORECASE,
)

# (2) Connective-free idioms that are promissory on their own.
_IDIOM = re.compile(
    r"TODO\(#\d+\)|\bdeferred to #\d+|\bblocked on #\d+",
    re.IGNORECASE,
)

_PATTERNS = (_CONNECTIVE_SUBJECT, _IDIOM)

# Every issue ref inside a matched span — ``findall`` so the two-issue
# ``"#1127/#1128 land"`` form enumerates both numbers.
_ISSUE_REF = re.compile(r"#(\d+)")

# The scanner's own package: its docstrings carry example forward-ref phrases by
# necessity, so scanning it would self-flag. Excluded by directory name.
_SELF_PACKAGE = "_drift"

# Default tree the canary scans (the repo runs it from the root).
_DEFAULT_ROOT = Path("src")


@dataclass(frozen=True, slots=True)
class Finding:
    """One forward-ref occurrence: an issue number used promissorily on a line."""

    path: str  # repo-relative POSIX path
    line: int  # 1-based
    number: int
    text: str  # the stripped source line, for the incident body


def _numbers_in_span(line: str, start: int, end: int) -> list[int]:
    """Issue numbers inside ``line[start:end]`` that clear the number floor.

    No structural-ordinal exclusion is needed: a number is only examined here when
    it sits inside a matched span, whose first token is the connective or idiom
    keyword — so the token immediately before ``#N`` is never ``invariant``/``step``/
    ``phase``/``Layer``. ``until phase #200 lands`` simply doesn't match (the ref
    isn't directly after the connective)."""
    numbers: list[int] = []
    for m in _ISSUE_REF.finditer(line, start, end):
        number = int(m.group(1))
        if number >= _MIN_ISSUE_NUMBER:
            numbers.append(number)
    return numbers


def scan_text(path: str, text: str) -> list[Finding]:
    """Findings for one file's text. Public so tests can drive it without disk."""
    findings: list[Finding] = []
    seen: set[tuple[int, int]] = set()  # (line, number) — dedup across patterns
    for line_no, line in enumerate(text.splitlines(), start=1):
        for pattern in _PATTERNS:
            for match in pattern.finditer(line):
                for number in _numbers_in_span(line, match.start(), match.end()):
                    if (line_no, number) not in seen:
                        seen.add((line_no, number))
                        findings.append(Finding(path, line_no, number, line.strip()))
    return findings


def scan_tree(root: Path = _DEFAULT_ROOT) -> list[Finding]:
    """Findings across every ``*.py`` under ``root``, excluding the scanner's own
    package (whose docstrings hold example phrases)."""
    findings: list[Finding] = []
    for py in sorted(root.rglob("*.py")):
        if _SELF_PACKAGE in py.parts:
            continue
        findings.extend(scan_text(py.as_posix(), py.read_text(encoding="utf-8")))
    return findings


def group_by_number(findings: list[Finding]) -> dict[int, list[Finding]]:
    """Group findings by issue number (the canary checks each number's state once)."""
    grouped: dict[int, list[Finding]] = {}
    for f in findings:
        grouped.setdefault(f.number, []).append(f)
    return grouped


def _main(argv: list[str]) -> int:
    """Emit the candidate forward-refs as JSON, grouped by number, for the canary.

    ``[{"number": 332, "occurrences": [{"path": ..., "line": ..., "text": ...}]}]``
    """
    root = Path(argv[1]) if len(argv) > 1 else _DEFAULT_ROOT
    grouped = group_by_number(scan_tree(root))
    payload = [
        {
            "number": number,
            "occurrences": [{"path": f.path, "line": f.line, "text": f.text} for f in occ],
        }
        for number, occ in sorted(grouped.items())
    ]
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
