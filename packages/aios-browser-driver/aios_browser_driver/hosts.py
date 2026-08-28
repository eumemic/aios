"""The signed-in-hosts heuristic (jarbot#106 §5.5).

A host counts as "signed in" when the profile holds at least one unexpired
``HttpOnly`` cookie for it — real session cookies are overwhelmingly HttpOnly,
and pure-JS state (analytics, consent banners) overwhelmingly is not. A
heuristic, deliberately: it feeds the owner-facing "where is this browser
signed in" list and the takeover handback delta, never an enforcement
decision.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def normalize_host(host: str) -> str:
    """The canonical host key: no leading dot, lowercased.

    One definition shared by the signed-in-hosts scan and ``revoke_site`` —
    revoke-then-report only works because the two agree on the key."""
    return host.strip().lstrip(".").lower()


def signed_in_hosts(cookies: Iterable[Mapping[str, Any]], *, now: float) -> list[str]:
    """Hosts holding ≥1 unexpired HttpOnly cookie, sorted and deduplicated.

    ``cookies`` are playwright ``context.cookies()`` records; ``expires`` is a
    unix timestamp, with ``-1`` marking a session cookie (unexpired by
    definition — it lives as long as the profile's session store does).
    """
    hosts: set[str] = set()
    for cookie in cookies:
        if not cookie.get("httpOnly"):
            continue
        expires = float(cookie.get("expires") or -1)
        if 0 < expires <= now:
            continue
        domain = normalize_host(str(cookie.get("domain") or ""))
        if domain:
            hosts.add(domain)
    return sorted(hosts)
