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


def _classify_permission(
    args: dict[str, Any], agent: Agent | AgentVersion
) -> PermissionPolicy | None:
    """Per-route permission lookup for the dispatch gate.

    Returns the matched route's ``permission_policy`` so the harness can
    park the call in ``requires_action`` if the operator marked it
    ``always_ask``. Returns ``None`` for missing server / no route match
    / bad args / query-or-fragment-bearing path — the handler then runs
    and emits a typed error the model can self-correct from.
    """
    server_ref = args.get("server_ref")
    path = args.get("path")
    if not isinstance(server_ref, str) or not isinstance(path, str):
        return None
    if "?" in path or "#" in path:
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

    # Route allowlists are path-only gates. A ``?`` or ``#`` in ``path``
    # would be matched as literal segment characters by ``match_glob``
    # and then parsed by httpx as a query/fragment on the wire — letting
    # ``/lights/1?action=delete`` slip past an `/lights/*` read-only route
    # because the upstream sees the query string and treats it as a
    # state-change request. Reject up front so the gate's intent is the
    # gate's effect.
    if "?" in path or "#" in path:
        return {
            "error": (
                f"path {path!r} contains a query string or fragment, which is "
                "not allowed — pass only the path portion. Route allowlists "
                "do not extend across query parameters."
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

    request_headers = {k: v for k, v in caller_headers.items() if k.lower() != "authorization"}
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
