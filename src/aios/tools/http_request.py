"""The http_request tool — make an authenticated HTTP call to one of the
agent's declared ``http_servers``.

The agent composes the URL, method, headers, and body; the worker
matches the path against the ``HttpServerSpec``'s route allowlist and
authors the ``Authorization`` header from the session's vault. Secret
never enters the sandbox.

Return shape: ``{"status": int, "headers": [[name, value], ...], "body": "..."}``.
``headers`` is a list of ``[name, value]`` pairs, not a dict: a dict collapses
duplicate header names, and HTTP *requires* some (notably ``Set-Cookie``) be
multi-valued — so a response setting two cookies would silently lose one. The
pair-list preserves every occurrence in wire order.
On error: ``{"error": "..."}``.
"""

from __future__ import annotations

import os
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
    http_route_suppressed,
)
from aios.services import agents as agents_service
from aios.services import outbound_suppression as outbound_suppression_service
from aios.services import sessions as sessions_service
from aios.tools._glob_match import match_glob
from aios.tools.registry import registry
from aios.tools.url_safety import is_safe_url

# The response-body character cap. Bodies longer than this are truncated — but
# NEVER silently: the result dict carries ``"truncated": True`` so the caller can
# fail loud instead of degrading to an empty/None parse (aios#1294). The default is
# ~1 MB (was 100 KB, which truncated a single page of a GitHub issue list mid-JSON
# and broke the triage scan). Operators can raise/lower it via
# ``AIOS_HTTP_RESPONSE_MAX_CHARS``; a missing / non-positive / unparseable value
# falls back to the default so a bad env never disables the cap.
_DEFAULT_MAX_RESPONSE_CHARS = 1_000_000


def _max_response_chars() -> int:
    """The response-body char cap, from ``AIOS_HTTP_RESPONSE_MAX_CHARS`` or the
    default. Resolved per call so an operator override takes effect without a
    process restart; a non-positive / unparseable value falls back to the default
    (we never want a bad env to mean ``[:0]`` or an unbounded read)."""
    raw = os.environ.get("AIOS_HTTP_RESPONSE_MAX_CHARS")
    if raw:
        try:
            n = int(raw)
        except ValueError:
            return _DEFAULT_MAX_RESPONSE_CHARS
        if n > 0:
            return n
    return _DEFAULT_MAX_RESPONSE_CHARS


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
    "is ignored), and optional body. Response body is truncated to ~1M "
    "characters (configurable via AIOS_HTTP_RESPONSE_MAX_CHARS); when a body "
    'is cut the result carries "truncated": true so callers never mistake a '
    "truncated body for a complete one. Only paths matching an enabled route "
    "on the server's allowlist are permitted; the worker refuses "
    "non-matching paths."
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


def _split_query(path: str) -> tuple[str, str]:
    """Split ``path`` into ``(path_portion, query_portion)`` at the first ``?``.

    ``httpx`` parses the query off the constructed URL, so the route gate must
    glob-match on the path portion alone. ``query_portion`` is ``""`` when there
    is no ``?``; it includes everything after the first ``?`` (the leading ``?``
    is dropped) so the caller can decide, per matched route, whether to allow it.
    """
    head, sep, tail = path.partition("?")
    return head, tail if sep else ""


def _path_rejected_reason(path: str) -> str | None:
    """Return the reason the path PORTION must be rejected at the route gate, or
    ``None`` when it's safe to glob-match. ``path`` here is already query-stripped
    (see ``_split_query``); the query allowance is decided per matched route by
    ``_query_rejected_reason``.

    The route allowlist is enforced by segment-glob matching on the path
    string, but the final URL is composed and dispatched via ``httpx``. The
    ``.`` / ``..`` dot-segments slip past the glob because ``*`` matches the
    literal dot-segment, while httpx applies RFC 3986 §5.2.4 dot-segment
    removal — ``/v1/lights/../admin`` collapses to ``/v1/admin`` and
    ``/lights/./state`` collapses to ``/lights/state`` on the wire. A ``#``
    fragment is similarly stripped by httpx with no route equivalent. Reject up
    front so the gate's check equals the gate's effect.
    """
    if "#" in path:
        return (
            f"path {path!r} contains a fragment, which is not allowed — pass "
            "only the path portion. Route allowlists do not extend across "
            "fragments."
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


def _query_rejected_reason(route: HttpRouteSpec, path: str, query: str) -> str | None:
    """Return the reason a query string on ``path`` must be rejected, or ``None``.

    A query is rejected (the #485 default) UNLESS the matched ``route`` opted in
    with ``allow_query=True``. Default-deny: ``?action=delete`` must not slip past
    a read-only ``/lights/*`` because httpx sends the query upstream verbatim.
    A ``#`` fragment is always rejected (handled in ``_path_rejected_reason`` on
    the path portion); a fragment cannot appear in ``query`` because the split is
    on the first ``?`` and httpx would carry a ``#`` in the query upstream, so we
    reject it here too for the allow_query path.
    """
    if not query:
        return None
    if "#" in query:
        return (
            f"path {path!r} contains a fragment, which is not allowed — pass "
            "only the path and query portions."
        )
    if not route.allow_query:
        return (
            f"path {path!r} contains a query string, which is not allowed on "
            "this route — pass only the path portion. Route allowlists do not "
            "extend across query parameters unless the route sets allow_query."
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
    (including method-disallowed) / bad args / fragment- or
    disallowed-query-bearing / dot-segment-bearing path — the handler then
    runs and emits a typed error the model can self-correct from.
    """
    server_ref = args.get("server_ref")
    path = args.get("path")
    method = args.get("method")
    if not isinstance(server_ref, str) or not isinstance(path, str) or not isinstance(method, str):
        return None
    path_only, query = _split_query(path)
    if _path_rejected_reason(path_only) is not None:
        return None
    server = _find_server(agent.http_servers, server_ref)
    if server is None:
        return None
    route = _match_route(server, path_only, method)
    if route is None:
        return None
    if _query_rejected_reason(route, path, query) is not None:
        return None
    if route.permission_policy is None:
        return None
    return route.permission_policy.type


async def _load_session_agent(session_id: str) -> tuple[Agent | AgentVersion, str, str]:
    pool = runtime.require_pool()
    account_id = await sessions_service.load_session_account_id(pool, session_id)
    session = await sessions_service.get_session_basic(pool, session_id, account_id=account_id)
    agent = await agents_service.load_for_session(pool, session, account_id=account_id)
    return agent, account_id, session.outbound_suppression


def _decode_body(response: httpx.Response) -> tuple[str, bool]:
    """Decode the response body to text and report whether it was truncated.

    Returns ``(body, truncated)``. ``truncated`` is ``True`` only when a textish
    body exceeded the char cap and was cut — so the caller can surface a
    ``"truncated": True`` flag and NEVER hand back a body that silently lost its
    tail (aios#1294: a 100 KB cut mid-JSON broke the triage scan). A refused
    binary body is never "truncated" in this sense — the marker IS the full,
    honest value — so it reports ``False``.
    """
    content_type = response.headers.get("content-type", "")
    is_textish = (
        content_type.startswith("text/")
        or "json" in content_type
        or "xml" in content_type
        or "javascript" in content_type
        or content_type == ""
    )
    if is_textish:
        text = response.text
        cap = _max_response_chars()
        if len(text) > cap:
            return text[:cap], True
        return text, False
    return (
        f"<binary content of type {content_type or 'unknown'}, "
        f"{len(response.content)} bytes — refused>"
    ), False


async def _do_http_request(
    *,
    servers: list[HttpServerSpec],
    arguments: dict[str, Any],
    resolve_auth: Callable[[str], Awaitable[tuple[str | None, dict[str, str]]]],
    on_suppress: (
        Callable[[HttpServerSpec, HttpRouteSpec, dict[str, Any]], Awaitable[dict[str, Any] | None]]
        | None
    ) = None,
) -> dict[str, Any]:
    """The owner-agnostic core: match a path against ``servers``' route allowlist,
    author the ``Authorization`` header via the injected ``resolve_auth``, and make the
    request. Shared by the session handler (servers = agent's) and the workflow-run
    dispatcher (servers = run's snapshot); the core never learns which principal it serves.

    ``on_suppress`` (outbound suppression, #710) is consulted AFTER the path /
    method / query gates pass but BEFORE the upstream dispatch. The session
    handler injects it when ``outbound_suppression == "on"``; it returns a
    synthesized success (and records the audit event) for a write, or ``None``
    to let the call proceed normally for a read. The workflow-run dispatcher
    leaves it ``None`` — suppression is a per-session property.
    """
    server_ref = arguments["server_ref"]
    path = arguments["path"]
    method = arguments["method"]
    caller_headers: dict[str, str] = arguments.get("headers") or {}
    body = arguments.get("body")

    # Route allowlists are path-only gates. Reject any path element that
    # ``httpx`` would mutate between match and wire (fragment, ``.`` / ``..``
    # dot-segments) so the gate's check equals the gate's effect. The query
    # string is split off and matched on the path portion alone, then the
    # query allowance is decided against the matched route (default-deny;
    # ``allow_query`` opt-in) — see ``_path_rejected_reason`` /
    # ``_query_rejected_reason``.
    path_only, query = _split_query(path)
    reason = _path_rejected_reason(path_only)
    if reason is not None:
        return {"error": reason}

    server = _find_server(servers, server_ref)
    if server is None:
        return {"error": (f"unknown server_ref {server_ref!r}; not declared on http_servers")}
    route = _match_route(server, path_only, method)
    if route is None:
        return {
            "error": (
                f"{method} {path!r} does not match any enabled route on "
                f"http_server {server_ref!r} — the path may be unlisted, or the "
                f"route's allowed methods may not include this verb"
            )
        }
    reason = _query_rejected_reason(route, path, query)
    if reason is not None:
        return {"error": reason}

    # Outbound suppression (#710): a session in suppression mode short-circuits a
    # write here — synthesized success, no upstream dispatch — AFTER the gates so
    # the agent sees the same accept/reject surface it would in production, and
    # the suppressed call is recorded against a real, allowlisted route.
    if on_suppress is not None:
        synthesized = await on_suppress(server, route, arguments)
        if synthesized is not None:
            return synthesized

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

    body_text, truncated = _decode_body(response)
    result: dict[str, Any] = {
        "status": response.status_code,
        # A list of ``[name, value]`` pairs, NOT a dict: ``dict(headers)`` collapses
        # duplicate names, silently dropping all but the last ``Set-Cookie`` (HTTP
        # requires it be multi-valued). ``multi_items()`` preserves every occurrence.
        "headers": [[k, v] for k, v in response.headers.multi_items()],
        "body": body_text,
    }
    # Signal a cut body explicitly (aios#1294). The flag is present ONLY when the
    # body was truncated, so a caller can branch on its mere presence; never cut a
    # body without saying so.
    if truncated:
        result["truncated"] = True
    return result


async def http_request_handler(session_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Session entry: resolve the agent's ``http_servers`` + a session-scoped vault resolver."""
    agent, account_id, outbound_suppression = await _load_session_agent(session_id)
    pool = runtime.require_pool()
    crypto_box = runtime.require_crypto_box()

    async def resolve_auth(base_url: str) -> tuple[str | None, dict[str, str]]:
        return await resolve_auth_for_target_url(
            pool, crypto_box, session_id, base_url, account_id=account_id
        )

    on_suppress = None
    if outbound_suppression == "on":

        async def on_suppress(
            server: HttpServerSpec, route: HttpRouteSpec, args: dict[str, Any]
        ) -> dict[str, Any] | None:
            method = str(args.get("method", ""))
            if not http_route_suppressed(route, method):
                return None  # a read — let it through against real credentials
            status = server.suppressed_response_status
            await outbound_suppression_service.record_http_suppression(
                pool,
                session_id,
                account_id=account_id,
                server_ref=server.name,
                base_url=server.base_url,
                method=method,
                path=str(args.get("path", "")),
                body=args.get("body"),
                synthesized_status=status,
            )
            return outbound_suppression_service.http_synthesized_response(status)

    return await _do_http_request(
        servers=agent.http_servers,
        arguments=arguments,
        resolve_auth=resolve_auth,
        on_suppress=on_suppress,
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
