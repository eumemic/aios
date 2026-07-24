"""Write-boundary validation for credential-bearing outbound target URLs (#861).

One canonical SSRF policy, two enforcement points:

* **Declaration time** (this module) — ingress bodies that persist an outbound
  target (``mcp_servers[].url`` on agent/workflow create/update,
  ``VaultCredentialCreate.target_url``) reject runtime-local / private /
  internal targets up front, so a bad declaration fails loudly at the write
  instead of latently at execution.
* **Connection time** (:class:`aios.pinned_transport.PinnedTransport`, wired
  into every MCP/OAuth httpx client) — re-resolves every request's host,
  validates all returned IPs against the same classifiers, and pins the
  connection; this is the fail-closed, DNS-rebinding-resistant authority.

Both points classify with the SAME primitives
(:func:`aios.tools.url_safety.is_blocked_ip` /
:func:`~aios.tools.url_safety.is_blocked_hostname`) and honor the SAME
operator escape hatch (``Settings.oauth_allow_insecure_host_set``, i.e.
``AIOS_OAUTH_ALLOW_INSECURE_HOSTS`` — the allowlist ``PinnedTransport`` and the
cleartext-credential guard already use), so declaration-time and
execution-time policy cannot drift: a URL that validates here is connectable
under exactly the same rules at execution, and a host allowlisted for one is
allowlisted for both.

Failure-mode split (deliberate): host strings that are *numeric* IPv4/IPv6
forms — including the decimal (``2130706433``), hex (``0x7f000001``) and
short-form (``127.1``) encodings of loopback — are normalized via
``AI_NUMERICHOST`` and rejected deterministically with no DNS traffic. For
DNS *names*, resolution here is best-effort defense-in-depth: a name that
resolves to a blocked range is rejected now; a name that does not resolve at
declaration time is admitted (there is nothing reachable to protect, and any
later resolution — including a rebind to a private address — is re-checked
fail-closed by ``PinnedTransport`` on every connection attempt).

Posture note (#1153 reconciliation): this validator governs only
*worker-originated* target URLs (MCP connect, vault-credential targets),
whose execution path runs through ``PinnedTransport`` and therefore blocks
private/self targets unconditionally regardless of any environment's sandbox
networking config — matching that unconditional stance here is what keeps the
write and execution boundaries coherent. The Limited-vs-Unrestricted
"permit-with-warning" posture of #1153 applies to *sandbox-egress*
``environment_variable`` credentials, which carry no ``target_url`` and are
untouched by this module (their Limited-only host-containment check lives in
``services/vaults.py``).
"""

from __future__ import annotations

import os
import socket
from urllib.parse import urlsplit


def _operator_allow_hosts() -> frozenset[str]:
    """The single operator allowlist shared with ``PinnedTransport``.

    Union of a direct read of ``AIOS_OAUTH_ALLOW_INSECURE_HOSTS`` (always
    fresh — validators run at arbitrary points relative to the
    ``get_settings()`` singleton's construction, and a write-boundary check
    must see the same env a just-(re)started process would) and the parsed
    ``Settings`` field (which additionally layers the ``.env``-file sources
    the env read alone would miss). Both spellings are the same operator
    variable, so this can only widen toward the identical allowlist
    ``PinnedTransport`` enforces at connection time — never drift from it.

    Lazy import: ``aios.config`` imports ``aios.models.vaults`` which imports
    this module. Settings construction failing (bare CLI contexts without the
    required env) falls back to the env read alone.
    """
    from aios.config import get_settings

    env_hosts = frozenset(
        h.strip().lower()
        for h in os.getenv("AIOS_OAUTH_ALLOW_INSECURE_HOSTS", "").split(",")
        if h.strip()
    )
    try:
        return env_hosts | get_settings().oauth_allow_insecure_host_set
    except Exception:
        return env_hosts


def _numeric_host_ips(host: str) -> list[str]:
    """Canonical IPs for a *numeric* host literal; ``[]`` for DNS names.

    ``AI_NUMERICHOST`` normalizes every IPv4/IPv6 literal encoding the socket
    layer itself accepts — decimal (``2130706433``), hex (``0x7f000001``),
    short-form (``127.1``), dotted quad, bracketed/plain IPv6 — with no DNS
    traffic, so a loopback-equivalent encoding cannot slip past a
    string-equality check.
    """
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM, flags=socket.AI_NUMERICHOST)
    except OSError:
        return []
    return [str(info[4][0]) for info in infos]


def _resolve_host_ips(host: str) -> list[str] | None:
    """Best-effort DNS resolution seam (monkeypatched in tests).

    ``None`` means the name does not currently resolve — see the module note
    on the deliberate declaration-time fail-open for unresolvable names.
    """
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return None
    return [str(info[4][0]) for info in infos]


def validate_outbound_target_url(url: str) -> str:
    """Validate one outbound target URL at the write boundary; return it unchanged.

    Raises ``ValueError`` (surfacing as a pydantic ``ValidationError`` on the
    ingress bodies that call this) when the target is not an absolute http(s)
    URL, carries userinfo, or addresses a private / internal / runtime-local
    host — unless the host is operator-allowlisted via
    ``AIOS_OAUTH_ALLOW_INSECURE_HOSTS`` (the same allowlist the connection-time
    ``PinnedTransport`` honors). ``AIOS_URL`` identifies this runtime's own
    public API origin; path differences do not make that origin safe.
    """
    # Lazy import — ``aios.tools``'s eager package __init__ would otherwise
    # join this module's import graph and cycle (same precedent as
    # ``pinned_transport`` / the egress proxy).
    from aios.tools.url_safety import is_blocked_hostname, is_blocked_ip

    parsed = urlsplit(url)
    host = parsed.hostname
    if parsed.scheme not in {"http", "https"} or not host or parsed.username is not None:
        raise ValueError("target URL must be an absolute http(s) URL without userinfo")

    normalized_host = host.rstrip(".").lower()
    try:
        port = parsed.port
    except ValueError as err:
        raise ValueError("target URL has an invalid port") from err
    effective_port = port if port is not None else (443 if parsed.scheme == "https" else 80)

    allow_hosts = _operator_allow_hosts()
    if (
        normalized_host in allow_hosts
        or f"{normalized_host}:{effective_port}" in allow_hosts
        or parsed.netloc.lower() in allow_hosts
    ):
        return url

    runtime_url = os.getenv("AIOS_URL")
    parsed_runtime_host = urlsplit(runtime_url).hostname if runtime_url else None
    runtime_host = parsed_runtime_host.rstrip(".").lower() if parsed_runtime_host else None

    blocked = (
        normalized_host == runtime_host
        or normalized_host == "localhost"
        or normalized_host.endswith(".localhost")
        or is_blocked_hostname(normalized_host)
    )

    if not blocked:
        numeric_ips = _numeric_host_ips(normalized_host)
        if numeric_ips:
            blocked = any(is_blocked_ip(ip) for ip in numeric_ips)
        else:
            resolved_ips = _resolve_host_ips(normalized_host)
            if resolved_ips is not None:
                blocked = any(is_blocked_ip(ip) for ip in resolved_ips)
            # else: unresolvable name — admitted here by design; the
            # fail-closed PinnedTransport gate re-resolves at every
            # connection attempt.

    if blocked:
        raise ValueError(
            "target URL resolves to a private, internal, or runtime-local address; "
            "allowlist the host via AIOS_OAUTH_ALLOW_INSECURE_HOSTS only for a "
            "deliberately internal deployment"
        )
    return url
