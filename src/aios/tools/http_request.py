"""The http_request tool — make an authenticated HTTP call to one of the
agent's declared ``http_servers``.

The agent composes the URL, method, headers, and body; the worker
matches the path against the ``HttpServerSpec``'s route allowlist and
authors the ``Authorization`` header from the session's vault. Secret
never enters the sandbox.

Return shape: ``{"status": int, "headers": {...}, "body": "..."}``.
On error: ``{"error": "..."}``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from aios.harness import runtime
from aios.mcp.client import resolve_auth_for_target_url
from aios.models.agents import (
    Agent,
    AgentVersion,
    HttpRouteSpec,
    HttpServerSpec,
    PermissionPolicy,
)
from aios.services import agents as agents_service
from aios.services import sessions as sessions_service
from aios.tools._glob_match import match_glob
from aios.tools.registry import registry
from aios.tools.url_safety import is_safe_url

_MAX_RESPONSE_CHARS = 100_000
_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
_ALLOWED_METHODS = ("GET", "POST", "PUT", "DELETE", "PATCH")

# Headers the worker manages on the agent's behalf. Agent-supplied values
# under any casing are stripped before dispatch so the route allowlist's
# intent (path-only + base_url-scoped) isn't bypassed.
_RESERVED_HEADERS = frozenset({"authorization", "host"})


HTTP_REQUEST_DESCRIPTION = (
    "Make an authenticated HTTP request to one of the agent's declared "
    "http_servers. Specify server_ref (the spec's name), path, method, "
    "optional headers (Authorization is set by the worker — your value "
    "is ignored), and optional body. Response body is truncated to 100k "
    "characters. Only paths matching an enabled route on the server's "
    "allowlist are permitted; the worker refuses non-matching paths."
)

HTTP_REQUEST_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "server_ref": {
            "type": "string",
            "description": (
                "Name of the HttpServerSpec on the agent. Must match an "
                "entry in agent.http_servers."
            ),
        },
        "path": {
            "type": "string",
            "description": (
                "Path appended to the server's base_url. Must match one "
                "of the server's enabled route patterns."
            ),
        },
        "method": {
            "type": "string",
            "enum": list(_ALLOWED_METHODS),
            "description": "HTTP method.",
        },
        "headers": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": (
                "Additional request headers. Authorization is reserved — "
                "any value here is dropped; the worker writes it from "
                "the session's vault credential."
            ),
        },
        "body": {
            "type": "string",
            "description": (
                "Request body string. Empty / omitted for methods that don't carry one."
            ),
        },
    },
    "required": ["server_ref", "path", "method"],
    "additionalProperties": False,
}


def _find_server(servers: list[HttpServerSpec], server_ref: str) -> HttpServerSpec | None:
    for s in servers:
        if s.name == server_ref:
            return s
    return None


def _match_route(server: HttpServerSpec, path: str, method: str) -> HttpRouteSpec | None:
    """First enabled route whose pattern matches ``path`` AND whose ``methods``
    admit ``method``. ``methods is None`` means all verbs are allowed (the
    method-dimension lattice top); a list scopes the route to exactly those
    verbs (``[]`` = deny-all). First-match-wins on launcher route order, so a
    method-disjoint earlier route does not shadow a later matching one."""
    for r in server.routes:
        if not (r.enabled and match_glob(r.path_pattern, path)):
            continue
        if r.methods is not None and method not in r.methods:
            continue
        return r
    return None


def _path_rejected_reason(path: str) -> str | None:
    """Return the reason ``path`` must be rejected at the route gate, or
    ``None`` when it's safe to glob-match.

    The route allowlist is enforced by segment-glob matching on the
    raw ``path`` string, but the final URL is composed and dispatched
    via ``httpx``. Any element that ``httpx`` mutates between match
    and wire is an allowlist bypass: ``?action=delete`` slips past
    ``/lights/*`` because httpx parses it as a query string; ``.`` /
    ``..`` segments slip past the glob because ``*`` matches the
    literal dot-segment, while httpx applies RFC 3986 §5.2.4
    dot-segment removal — ``/v1/lights/../admin`` collapses to
    ``/v1/admin`` and ``/lights/./state`` collapses to
    ``/lights/state`` on the wire. Reject up front so the gate's
    check equals the gate's effect.
    """
    if "?" in path or "#" in path:
        return (
            f"path {path!r} contains a query string or fragment, which is "
            "not allowed — pass only the path portion. Route allowlists "
            "do not extend across query parameters."
        )
    # ``.`` / ``..`` as literal segments: ``foo/../bar`` is normalised by
    # httpx to ``bar``, and ``foo/./bar`` to ``foo/bar`` — escaping any
    # allowlist that matched on the original shape. The check looks for
    # whole segments, not substrings, so paths like ``foo..bar`` (legal
    # in many APIs) are left alone.
    segments = path.split("/")
    if ".." in segments or "." in segments:
        return (
            f"path {path!r} contains a '.' or '..' segment, which is not "
            "allowed — the upstream URL would be normalised to a different "
            "path, escaping the route allowlist's intent. Pass an absolute "
            "path without dot-segments."
        )
    return None


def _classify_permission(
    args: dict[str, Any], agent: Agent | AgentVersion
) -> PermissionPolicy | None:
    """Per-route permission lookup for the dispatch gate.

    Returns the matched route's ``permission_policy`` so the harness can
    leave the call unresolved (pending confirmation via
    ``POST /sessions/:id/tool-confirmations``) if the operator marked it
    ``always_ask``. Returns ``None`` for missing server / no route match
    (including method-disallowed) / bad args / query-or-fragment-bearing /
    dot-segment-bearing path — the handler then runs and emits a typed
    error the model can self-correct from.
    """
    server_ref = args.get("server_ref")
    path = args.get("path")
    method = args.get("method")
    if not isinstance(server_ref, str) or not isinstance(path, str) or not isinstance(method, str):
        return None
    if _path_rejected_reason(path) is not None:
        return None
    server = _find_server(agent.http_servers, server_ref)
    if server is None:
        return None
    route = _match_route(server, path, method)
    if route is None or route.permission_policy is None:
        return None
    return route.permission_policy.type


async def _load_session_agent(session_id: str) -> tuple[Agent | AgentVersion, str]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    session = await sessions_service.get_session_basic(pool, session_id, account_id=account_id)
    agent = await agents_service.load_for_session(pool, session, account_id=account_id)
    return agent, account_id


def _decode_body(response: httpx.Response) -> str:
    content_type = response.headers.get("content-type", "")
    is_textish = (
        content_type.startswith("text/")
        or "json" in content_type
        or "xml" in content_type
        or "javascript" in content_type
        or content_type == ""
    )
    if is_textish:
        return response.text[:_MAX_RESPONSE_CHARS]
    return (
        f"<binary content of type {content_type or 'unknown'}, "
        f"{len(response.content)} bytes — refused>"
    )


async def _do_http_request(
    *,
    servers: list[HttpServerSpec],
    arguments: dict[str, Any],
    resolve_auth: Callable[[str], Awaitable[tuple[str | None, dict[str, str]]]],
) -> dict[str, Any]:
    """The owner-agnostic core: match a path against ``servers``' route allowlist,
    author the ``Authorization`` header via the injected ``resolve_auth``, and make the
    request. Shared by the session handler (servers = agent's) and the workflow-run
    dispatcher (servers = run's snapshot); the core never learns which principal it serves.
    """
    server_ref = arguments["server_ref"]
    path = arguments["path"]
    method = arguments["method"]
    caller_headers: dict[str, str] = arguments.get("headers") or {}
    body = arguments.get("body")

    # Route allowlists are path-only gates. Reject any path element that
    # ``httpx`` would mutate between match and wire (query/fragment,
    # ``.`` / ``..`` dot-segments) so the gate's check equals the gate's
    # effect — see ``_path_rejected_reason``.
    reason = _path_rejected_reason(path)
    if reason is not None:
        return {"error": reason}

    server = _find_server(servers, server_ref)
    if server is None:
        return {"error": (f"unknown server_ref {server_ref!r}; not declared on http_servers")}
    if _match_route(server, path, method) is None:
        return {
            "error": (
                f"{method} {path!r} does not match any enabled route on "
                f"http_server {server_ref!r} — the path may be unlisted, or the "
                f"route's allowed methods may not include this verb"
            )
        }

    full_url = server.base_url.rstrip("/") + "/" + path.lstrip("/")
    if not is_safe_url(full_url):
        return {"error": f"Blocked: URL targets a private/internal address: {full_url}"}

    _vault_id, auth_headers = await resolve_auth(server.base_url)

    # Strip headers the worker manages on the caller's behalf. ``Authorization``
    # is rendered from the bound vault by ``resolve_auth``. ``Host`` is derived
    # from the request URL by httpx; a caller override would land the request on a
    # different name-based virtual host than the operator's base_url scopes
    # to (NGINX, Cloudflare, ALB, Ingress — all route by Host header), so
    # leaving it caller-controlled effectively defeats the route allowlist.
    request_headers = {
        k: v for k, v in caller_headers.items() if k.lower() not in _RESERVED_HEADERS
    }
    request_headers.update(auth_headers)

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            kwargs: dict[str, Any] = {"headers": request_headers}
            if body:
                kwargs["content"] = body
            response = await client.request(method, full_url, **kwargs)
    except httpx.TimeoutException:
        return {"error": f"Request timed out: {method} {full_url}"}
    except httpx.HTTPError as exc:
        return {"error": f"HTTP transport error: {type(exc).__name__}: {exc}"}

    return {
        "status": response.status_code,
        "headers": dict(response.headers),
        "body": _decode_body(response),
    }


async def http_request_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Session entry: resolve the agent's ``http_servers`` + a session-scoped vault resolver."""
    agent, account_id = await _load_session_agent(session_id)
    pool = runtime.require_pool()
    crypto_box = runtime.require_crypto_box()

    async def resolve_auth(base_url: str) -> tuple[str | None, dict[str, str]]:
        return await resolve_auth_for_target_url(
            pool, crypto_box, session_id, base_url, account_id=account_id
        )

    return await _do_http_request(
        servers=agent.http_servers, arguments=arguments, resolve_auth=resolve_auth
    )


def _register() -> None:
    registry.register(
        name="http_request",
        description=HTTP_REQUEST_DESCRIPTION,
        parameters_schema=HTTP_REQUEST_PARAMETERS_SCHEMA,
        handler=http_request_handler,
        transport="both",
        classify_permission=_classify_permission,
    )


_register()
