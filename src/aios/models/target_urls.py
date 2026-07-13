"""Validation shared by credential-bearing outbound target declarations."""

from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlsplit


def validate_outbound_target_url(url: str) -> str:
    """Reject runtime-local and non-public literal targets unless allowlisted.

    ``AIOS_TARGET_URL_ALLOW_HOSTS`` is the explicit operator escape hatch for
    deliberately internal MCP deployments. Entries are comma-separated bare
    hosts or ``host:port`` authorities. ``AIOS_URL`` identifies this runtime's
    public API origin; path differences do not make that origin safe.
    """
    parsed = urlsplit(url)
    host = parsed.hostname
    if parsed.scheme not in {"http", "https"} or not host or parsed.username is not None:
        raise ValueError("target URL must be an absolute http(s) URL without userinfo")

    normalized_host = host.rstrip(".").lower()
    authority = parsed.netloc.lower()
    allow_hosts = {
        entry.strip().lower()
        for entry in os.getenv("AIOS_TARGET_URL_ALLOW_HOSTS", "").split(",")
        if entry.strip()
    }
    if normalized_host in allow_hosts or authority in allow_hosts:
        return url

    runtime_url = os.getenv("AIOS_URL")
    parsed_runtime_host = urlsplit(runtime_url).hostname if runtime_url else None
    runtime_host = parsed_runtime_host.rstrip(".").lower() if parsed_runtime_host else None
    blocked = (
        normalized_host == runtime_host
        or normalized_host == "localhost"
        or normalized_host.endswith(".localhost")
    )
    try:
        address = ipaddress.ip_address(normalized_host)
    except ValueError:
        address = None
    if address is not None:
        blocked = blocked or not address.is_global

    if blocked:
        raise ValueError(
            "target URL resolves to a private or runtime-local host; explicitly allowlist "
            "the host with AIOS_TARGET_URL_ALLOW_HOSTS only for an informed internal deployment"
        )
    return url
