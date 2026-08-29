"""The ssh tool — run a shell command on one of the agent's declared
``ssh_servers``, authenticated by a vault-held private key.

Structural sibling of :mod:`aios.tools.http_request`: the agent composes the
command, the worker resolves the ``ssh_key`` vault credential named by the
server's ``credential`` and holds it in memory only for the call; the key never
enters the sandbox or the model context. The owner-agnostic core
(:func:`_do_ssh`) takes the server list, a key resolver, and an optional
suppression hook injected — so the same gate logic can serve any owner, exactly
as http_request's split does (v1 wires only the session path).

Security posture:

* **Host-key pin (required).** The server's presented host key must be in the
  spec's ``host_keys`` set or the connection is refused. No trust-on-first-use,
  no known-hosts store, no insecure mode. This is the SSH analogue of TLS
  certificate verification that a vaulted credential requires.
* **Connect-IP pinning.** The host is resolved ONCE and the connection pinned to
  that IP (:func:`aios.pinned_transport.resolve_pinned_ip`), so DNS rebinding
  cannot redirect an allowed name after the check. Private/internal addresses
  are refused by default; an operator exempts named hosts via
  ``settings.ssh_allow_internal_hosts`` (still resolve+pin, skip the range
  check) for Tailscale / VPN / internal infra.
* **Key stays worker-side.** ``ResolvedSshKey`` is ``repr=False`` and held only
  in a local; it is never logged nor placed in a ToolBail message.

Error contract (mirrors http_request): every EXPECTED failure — unknown
server_ref, missing credential, blocked host, connect/command timeout, host-key
mismatch, auth rejection, transport fault — raises :class:`ToolBail` (a clean
model-visible refusal). A raw exception would be treated as internal and evict
the sandbox. Crucially, a non-zero remote **exit code is NOT an error**: it is a
legitimate ``exit_code`` value on the success-shaped result, just as an upstream
4xx is a legitimate ``status`` for http_request. Only faults that prevented the
command from running (or reading its result) bail.

Return shape: ``{"exit_code": int, "stdout": str, "stderr": str}`` with
``stdout_truncated`` / ``stderr_truncated`` present (and ``True``) only when a
stream was capped at ``settings.ssh_max_output_chars``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import asyncssh

from aios.config import get_settings
from aios.harness import runtime
from aios.models.agents import PermissionPolicy, SshServerSpec, StepSurface, ssh_server_suppressed
from aios.pinned_transport import resolve_internal_ip, resolve_pinned_ip
from aios.services import agents as agents_service
from aios.services import outbound_suppression as outbound_suppression_service
from aios.services import sessions as sessions_service
from aios.services import vaults as vaults_service
from aios.services.vaults import ResolvedSshKey
from aios.tools.invoke import ToolBail
from aios.tools.registry import registry

SSH_DESCRIPTION = (
    "Run a shell command on one of the agent's declared ssh_servers. Specify "
    "server_ref (the name of the SSH server, as listed in the system prompt) and "
    "command; optionally timeout_seconds up to the operator ceiling. Authentication "
    "uses the server's vault-held key — you never see, supply, or handle it. The "
    "server's host key must match the pinned set or the call is refused. A non-zero "
    "exit_code is a NORMAL result (inspect stdout/stderr), not an error. stdout and "
    "stderr are each truncated at the operator cap, flagged with "
    '"stdout_truncated"/"stderr_truncated": true. No pty is allocated — compose '
    "non-interactive commands."
)

SSH_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "server_ref": {
            "type": "string",
            "description": (
                "Name of the SshServerSpec on the agent (must match an entry in ssh_servers)."
            ),
        },
        "command": {
            "type": "string",
            "description": (
                "The shell command line to run on the remote host, via the remote "
                "user's login shell."
            ),
        },
        "timeout_seconds": {
            "type": "integer",
            "description": (
                "Optional per-call time limit for the command, capped at the operator maximum."
            ),
        },
    },
    "required": ["server_ref", "command"],
    "additionalProperties": False,
}


def _find_server(servers: list[SshServerSpec], server_ref: str) -> SshServerSpec | None:
    """First enabled server matching ``server_ref`` (a disabled server is invisible)."""
    for s in servers:
        if s.name == server_ref and s.enabled:
            return s
    return None


def _classify_permission(args: dict[str, Any], surface: StepSurface) -> PermissionPolicy | None:
    """Per-server ``always_ask`` gate for the disposition classifier.

    Returns the matched server's ``permission_policy.type``; ``None`` when the
    server_ref is malformed or unmatched, so the handler runs and emits a typed,
    self-correctable error rather than the classifier guessing.
    """
    server_ref = args.get("server_ref")
    if not isinstance(server_ref, str):
        return None
    server = _find_server(surface.ssh_servers, server_ref)
    if server is None or server.permission_policy is None:
        return None
    return server.permission_policy.type


async def _load_session_agent(session_id: str) -> tuple[StepSurface, str, str]:
    """Re-derive the acting surface, account, and suppression mode from the session id
    (verbatim twin of ``http_request._load_session_agent``)."""
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    session = await sessions_service.get_session_basic(pool, session_id, account_id=account_id)
    agent = await agents_service.load_for_session(pool, session, account_id=account_id)
    return agent, account_id, session.outbound_suppression


def _resolve_timeout(arguments: dict[str, Any]) -> int:
    """The command timeout: the operator ceiling by default, an agent-supplied
    ``timeout_seconds`` clamped to it (bash's ceiling-clamp shape)."""
    ceiling = get_settings().ssh_command_timeout_seconds
    raw = arguments.get("timeout_seconds", ceiling)
    if not isinstance(raw, int) or isinstance(raw, bool) or raw <= 0:
        raise ToolBail("timeout_seconds must be a positive integer")
    return min(raw, ceiling)


def _cap(text: str, limit: int) -> tuple[str, bool]:
    """Cap a stream at ``limit`` chars, reporting whether it was truncated."""
    if len(text) <= limit:
        return text, False
    return text[:limit], True


async def _do_ssh(
    *,
    servers: list[SshServerSpec],
    arguments: dict[str, Any],
    resolve_key: Callable[[str], Awaitable[ResolvedSshKey | None]],
    on_suppress: (
        Callable[[SshServerSpec, dict[str, Any]], Awaitable[dict[str, Any] | None]] | None
    ) = None,
) -> dict[str, Any]:
    """Owner-agnostic ssh exec core. Never learns which principal it serves — the
    key resolver and server list are injected, exactly as http_request's core."""
    settings = get_settings()
    server_ref = arguments.get("server_ref", "")
    server = _find_server(servers, str(server_ref))
    if server is None:
        raise ToolBail(f"unknown server_ref {server_ref!r}; not declared on ssh_servers")

    command = arguments.get("command")
    if not isinstance(command, str) or not command:
        raise ToolBail("command must be a non-empty string")

    timeout = _resolve_timeout(arguments)

    # Suppression is consulted AFTER the server gate (so the agent sees the same
    # accept/reject surface it would in production) and BEFORE key resolution or
    # the dial (so no real connection is opened for a suppressed command).
    if on_suppress is not None:
        synthesized = await on_suppress(server, arguments)
        if synthesized is not None:
            return synthesized

    resolved: ResolvedSshKey | None = await resolve_key(server.credential)
    if resolved is None:
        raise ToolBail(
            f"no ssh_key credential named {server.credential!r} in this session's vaults"
        )

    # Resolve+pin the host IP. Operator-allowlisted internal hosts skip the
    # private-range block (still resolve+pin); everyone else is public-only and
    # rebinding-pinned. The required host-key pin authenticates whatever answers.
    allow_key = f"{server.host}:{server.port}"
    if server.host in settings.ssh_allow_internal_host_set or (
        allow_key in settings.ssh_allow_internal_host_set
    ):
        pinned_ip = await resolve_internal_ip(server.host, server.port)
    else:
        pinned_ip = await resolve_pinned_ip(server.host, server.port)
    if pinned_ip is None:
        raise ToolBail(
            f"refusing to connect to {server.host!r}: unresolvable or resolves to a "
            "private/internal address (an operator can allowlist it via "
            "AIOS_SSH_ALLOW_INTERNAL_HOSTS)"
        )

    try:
        client_key = asyncssh.import_private_key(
            resolved.private_key, passphrase=resolved.passphrase
        )
    except (asyncssh.KeyImportError, ValueError) as exc:
        # Do NOT include the exception text — it can echo key material.
        raise ToolBail(
            f"the vault ssh_key for {server.credential!r} could not be parsed "
            f"({type(exc).__name__})"
        ) from exc

    try:
        # ``*``-pattern known_hosts: verification is "server key ∈ pinned set",
        # independent of the (IP-literal) name we dialed.
        known_hosts = asyncssh.import_known_hosts(
            "\n".join(f"* {line}" for line in server.host_keys)
        )
    except ValueError as exc:
        raise ToolBail(
            f"host_keys for {server.name!r} could not be parsed ({type(exc).__name__})"
        ) from exc

    try:
        async with asyncio.timeout(settings.ssh_connect_timeout_seconds):
            conn = await asyncssh.connect(
                pinned_ip,
                port=server.port,
                username=server.username,
                client_keys=[client_key],
                known_hosts=known_hosts,
                agent_path=None,  # never consult a worker-host ssh-agent
            )
    except asyncssh.HostKeyNotVerifiable as exc:
        raise ToolBail(
            f"host key verification failed for {server.name!r}: the server presented a "
            "key not in the pinned host_keys set"
        ) from exc
    except asyncssh.PermissionDenied as exc:
        raise ToolBail(f"authentication rejected for {server.username}@{server.host}") from exc
    except TimeoutError as exc:
        raise ToolBail(f"connection to {server.host} timed out") from exc
    except (asyncssh.Error, OSError) as exc:
        raise ToolBail(f"SSH transport error: {type(exc).__name__}: {exc}") from exc

    try:
        async with conn:
            try:
                async with asyncio.timeout(timeout):
                    proc = await conn.run(command, check=False, encoding="utf-8", errors="replace")
            except TimeoutError as exc:
                conn.abort()
                raise ToolBail(
                    f"command timed out after {timeout}s; the connection was closed"
                ) from exc
    except asyncssh.Error as exc:
        raise ToolBail(f"SSH transport error: {type(exc).__name__}: {exc}") from exc

    limit = settings.ssh_max_output_chars
    stdout, stdout_trunc = _cap(str(proc.stdout or ""), limit)
    stderr, stderr_trunc = _cap(str(proc.stderr or ""), limit)
    # ``exit_status`` is None when the remote process died by signal; surface -1
    # (and note the signal in stderr is already present in ``proc.stderr``).
    exit_code = proc.exit_status if proc.exit_status is not None else -1
    result: dict[str, Any] = {"exit_code": exit_code, "stdout": stdout, "stderr": stderr}
    if stdout_trunc:
        result["stdout_truncated"] = True
    if stderr_trunc:
        result["stderr_truncated"] = True
    return result


async def ssh_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Session entry: resolve the agent's ``ssh_servers`` + a session-scoped key resolver."""
    agent, account_id, outbound_suppression = await _load_session_agent(session_id)
    pool = runtime.require_pool()
    crypto_box = runtime.require_crypto_box()

    async def resolve_key(secret_name: str) -> ResolvedSshKey | None:
        return await vaults_service.resolve_session_ssh_key(
            pool, crypto_box, session_id, secret_name, account_id=account_id
        )

    on_suppress = None
    if outbound_suppression == "on":

        async def on_suppress(server: SshServerSpec, args: dict[str, Any]) -> dict[str, Any] | None:
            if not ssh_server_suppressed(server):
                return None  # operator-attested read-only server — let it through
            await outbound_suppression_service.record_ssh_suppression(
                pool,
                session_id,
                account_id=account_id,
                server_ref=server.name,
                host=server.host,
                username=server.username,
                command=str(args.get("command", "")),
            )
            return outbound_suppression_service.ssh_synthesized_result()

    return await _do_ssh(
        servers=agent.ssh_servers,
        arguments=arguments,
        resolve_key=resolve_key,
        on_suppress=on_suppress,
    )


def _register() -> None:
    registry.register(
        name="ssh",
        description=SSH_DESCRIPTION,
        parameters_schema=SSH_PARAMETERS_SCHEMA,
        handler=ssh_handler,
        # agent_tool only: a sandbox-reachable ssh would chain sandbox bash into
        # a remote shell — keep the model the bottleneck for this outbound effect.
        transport="agent_tool",
        executes="worker",
        classify_permission=_classify_permission,
    )


_register()
