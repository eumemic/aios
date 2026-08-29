"""The signed-in-hosts heuristic over a fixture cookie jar."""

from __future__ import annotations

from typing import Any

from aios_browser_driver.hosts import signed_in_hosts

_NOW = 1_700_000_000.0


def _cookie(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "sid",
        "value": "x",
        "domain": "example.com",
        "path": "/",
        "expires": _NOW + 3600,
        "httpOnly": True,
        "secure": True,
        "sameSite": "Lax",
    }
    base.update(overrides)
    return base


def test_unexpired_httponly_cookies_count() -> None:
    jar = [
        _cookie(domain="github.com"),
        _cookie(domain=".google.com"),  # leading dot normalized
        _cookie(domain="expired.example", expires=_NOW - 10),  # expired: out
        _cookie(domain="analytics.example", httpOnly=False),  # JS cookie: out
        _cookie(domain="session.example", expires=-1),  # session cookie: in
        _cookie(domain="github.com", name="second"),  # deduped
        _cookie(domain=""),  # no domain: out
    ]
    assert signed_in_hosts(jar, now=_NOW) == ["github.com", "google.com", "session.example"]


def test_empty_jar() -> None:
    assert signed_in_hosts([], now=_NOW) == []


def test_host_with_only_non_httponly_cookies_is_absent() -> None:
    jar = [_cookie(domain="tracker.example", httpOnly=False)]
    assert signed_in_hosts(jar, now=_NOW) == []


def test_domains_are_lowercased() -> None:
    jar = [_cookie(domain=".GitHub.COM")]
    assert signed_in_hosts(jar, now=_NOW) == ["github.com"]
