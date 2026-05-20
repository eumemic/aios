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


def _find_server(agent: Agent | AgentVersion, server_ref: str) -> HttpServerSpec | None:
    for s in agent.http_servers:
        if s.name == server_ref:
            return s
    return None


def _match_route(server: HttpServerSpec, path: str) -> HttpRouteSpec | None:
    for r in server.routes:
        if r.enabled and match_glob(r.path_pattern, path):
            return r
    return None


_DOT_SEGMENTS = frozenset({".", ".."})


def _path_is_dispatchable(path: str) -> bool:
    """True iff ``path`` is safe to feed through the route allowlist.

    Filters paths whose textual shape lets the upstream URL diverge from
    what ``match_glob`` checked:

    * ``?`` / ``#`` — httpx parses these off as query / fragment, so the
      upstream sees a different path than the gate validated.
    * ``.`` / ``..`` segments — httpx applies RFC 3986 §5.2.4 dot-segment
      removal: ``base_url/lights/..`` collapses to ``base_url/`` (reaching
      the server root despite a ``/lights/*`` allowlist), and
      ``base_url/lights/./state`` collapses to ``base_url/lights/state``
      (escaping a ``/lights/*/state`` allowlist that expected a real id
      segment between). ``*`` matches the literal ``.`` / ``..`` segment,
      so without this filter the glob accepts what the wire then rewrites
      past the gate.

    Same posture as the ``?`` / ``#`` filters: reject before dispatch
    so the gate's intent is the gate's effect.
    """
    if "?" in path or "#" in path:
        return False
    # Split on ``/`` to inspect each segment; ``.`` / ``..`` only trigger
    # httpx's normalization when they're whole segments, not when they
    # appear inside one (e.g. ``/foo..bar`` is a literal path component).
    return not any(seg in _DOT_SEGMENTS for seg in path.split("/"))


def _classify_permission(
    args: dict[str, Any], agent: Agent | AgentVersion
) -> PermissionPolicy | None:
    """Per-route permission lookup for the dispatch gate.

    Returns the matched route's ``permission_policy`` so the harness can
    park the call in ``requires_action`` if the operator marked it
    ``always_ask``. Returns ``None`` for missing server / no route match
    / bad args / non-dispatchable path (see :func:`_path_is_dispatchable`)
    — the handler then runs and emits a typed error the model can
    self-correct from.
    """
    server_ref = args.get("server_ref")
    path = args.get("path")
    if not isinstance(server_ref, str) or not isinstance(path, str):
        return None
    if not _path_is_dispatchable(path):
        return None
    server = _find_server(agent, server_ref)
    if server is None:
        return None
    route = _match_route(server, path)
    if route is None or route.permission_policy is None:
        return None
    return route.permission_policy.type


async def _load_session_agent(session_id: str) -> tuple[Agent | AgentVersion, str]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    session = await sessions_service.get_session(pool, session_id, account_id=account_id)
    if session.agent_version is not None:
        agent: Agent | AgentVersion = await agents_service.get_agent_version(
            pool, session.agent_id, session.agent_version, account_id=account_id
        )
    else:
        agent = await agents_service.get_agent(pool, session.agent_id, account_id=account_id)
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


async def http_request_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    server_ref = arguments["server_ref"]
    path = arguments["path"]
    method = arguments["method"]
    caller_headers: dict[str, str] = arguments.get("headers") or {}
    body = arguments.get("body")

    # Route allowlists are path-only gates. Filter paths whose textual
    # shape lets the upstream URL diverge from what ``match_glob`` would
    # check — ``?`` / ``#`` (parsed off as query / fragment), and
    # ``.`` / ``..`` path segments (RFC 3986 §5.2.4 dot-segment removal:
    # httpx normalizes ``base_url/lights/..`` to ``base_url/`` post-glob).
    # Reject up front so the gate's intent is the gate's effect.
    if not _path_is_dispatchable(path):
        return {
            "error": (
                f"path {path!r} contains a query string, fragment, or "
                "'.'/'..' segment, which is not allowed — pass only a "
                "normalized path portion. Route allowlists do not extend "
                "across query parameters or dot-segment normalization."
            )
        }

    agent, account_id = await _load_session_agent(session_id)
    server = _find_server(agent, server_ref)
    if server is None:
        return {"error": (f"unknown server_ref {server_ref!r}; not declared on agent.http_servers")}
    if _match_route(server, path) is None:
        return {
            "error": (
                f"path {path!r} does not match any enabled route on http_server {server_ref!r}"
            )
        }

    full_url = server.base_url.rstrip("/") + "/" + path.lstrip("/")
    if not is_safe_url(full_url):
        return {"error": f"Blocked: URL targets a private/internal address: {full_url}"}

    pool = runtime.require_pool()
    crypto_box = runtime.require_crypto_box()
    _vault_id, auth_headers = await resolve_auth_for_target_url(
        pool, crypto_box, session_id, server.base_url, account_id=account_id
    )

    # Strip headers the worker manages on the agent's behalf. ``Authorization``
    # is rendered from the session vault below. ``Host`` is derived from the
    # request URL by httpx; an agent override would land the request on a
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


def _register() -> None:
    registry.register(
        name="http_request",
        description=HTTP_REQUEST_DESCRIPTION,
        parameters_schema=HTTP_REQUEST_PARAMETERS_SCHEMA,
        handler=http_request_handler,
        classify_permission=_classify_permission,
    )


_register()
