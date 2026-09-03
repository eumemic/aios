"""E.164 normalizer — symmetric at store + lookup."""

from __future__ import annotations

import pytest

from aios_sms.addressing import normalize_e164


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("+18005551234", "+18005551234"),
        ("18005551234", "+18005551234"),
        ("+1 800 555-1234", "+18005551234"),
        ("  +1-800-555-1234  ", "+18005551234"),
        ("1-800-555-1234", "+18005551234"),
    ],
)
def test_normalize_collapses_formatting(raw: str, expected: str) -> None:
    assert normalize_e164(raw) == expected


def test_normalize_is_symmetric_idempotent() -> None:
    """Store side and lookup side must produce byte-identical output, or
    operator formatting drift silently misroutes (design §3.3)."""
    stored = normalize_e164("+1 (800) 555 1234".replace("(", "").replace(")", ""))
    looked_up = normalize_e164("18005551234")
    assert stored == looked_up
    # idempotent: normalizing an already-normalized value is a no-op
    assert normalize_e164(stored) == stored


def test_normalize_empty_stays_empty() -> None:
    assert normalize_e164("") == ""
    assert normalize_e164("   ") == ""
