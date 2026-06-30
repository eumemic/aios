"""Programmatic answer extraction + scoring for the REASONING/CORRECTNESS tier.

No LLM-judge: each task has a canonical checkable answer, the model is instructed to
end with ``ANSWER: <value>``, and we extract that and compare after normalization. This
deliberately proves the matched-fan-in measuring machinery on an OBJECTIVE metric before
the LLM-judge (Phase B) is introduced — a judge would itself be a fusion node and
confound the "does fusion help" question.

Comparison is forgiving of representation (``13.5`` == ``$13.50`` == ``13.50``;
``2/9`` == ``0.2222``; ``Friday`` == ``friday.``) but never of the actual value.
"""

from __future__ import annotations

import re
from fractions import Fraction


def extract_answer(text: str | None) -> str | None:
    """Pull the value after the last ``ANSWER:`` marker; fallback to last number/line."""
    if not text:
        return None
    # Primary: the explicit marker (last occurrence wins — models sometimes restate).
    matches = re.findall(r"ANSWER\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip().splitlines()[0].strip()
    # Fallback: last non-empty line (a model that ignored the format).
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else None


def _to_number(s: str) -> float | None:
    """Parse a numeric answer tolerant of $, commas, %, a/b fractions, and a trailing
    unit word (e.g. ``13.50 dollars``, ``12 feet``)."""
    s = s.strip().replace(",", "").replace("$", "").strip()
    # fraction a/b (allow a trailing unit after the fraction too)
    m = re.match(r"^(-?\d+)\s*/\s*(\d+)\b", s)
    if m:
        try:
            return float(Fraction(int(m.group(1)), int(m.group(2))))
        except (ValueError, ZeroDivisionError):
            return None
    # leading signed decimal, optionally followed by % or a unit word — take the number
    m = re.match(r"^(-?\d+(?:\.\d+)?)\s*%?", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).strip(".!? ")


def is_correct(extracted: str | None, canonical: str) -> bool:
    """True iff the extracted answer matches the canonical, value-wise.

    Numeric answers compare with a relative+absolute tolerance (handles 74.29 vs
    74.2857 and 13.5 vs 13.50); otherwise a normalized string/token compare, with a
    fraction-vs-decimal bridge.
    """
    if extracted is None:
        return False
    ext = extracted.strip()
    can = canonical.strip()

    # Try numeric (covers ints, decimals, fractions, $, %, commas).
    en, cn = _to_number(ext), _to_number(can)
    if en is not None and cn is not None:
        return abs(en - cn) <= max(1e-2, 1e-3 * abs(cn))

    # One side numeric, the other a fraction string already handled above; fall through
    # to normalized text compare (e.g. weekday names).
    nt_e, nt_c = _normalize_text(ext), _normalize_text(can)
    if nt_e == nt_c:
        return True
    # Token containment for short single-token answers (e.g. extracted "Friday." vs
    # canonical "Friday", or "the answer is Friday").
    return bool(nt_c and re.search(rf"\b{re.escape(nt_c)}\b", nt_e))


def score(text: str | None, canonical: str) -> tuple[bool, str | None]:
    """Return (correct, extracted_answer) for one model output."""
    ext = extract_answer(text)
    return is_correct(ext, canonical), ext
