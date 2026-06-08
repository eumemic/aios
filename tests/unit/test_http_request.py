"""Unit tests for the ``http_request`` built-in tool."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from aios.models.agents import HttpPermissionPolicy, HttpRouteSpec, HttpServerSpec, ToolSpec
from aios.tools.http_request import (
    _classify_permission,
    _decode_body,
    http_request_handler,
)

# Capture the real httpx.AsyncClient before any patch in this file replaces it.
# A stub that calls ``httpx.AsyncClient(...)`` after a patch would otherwise
# recurse into itself.
_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _agent(
    *, http_servers: list[HttpServerSpec], tools: list[ToolSpec] | None = None
) -> SimpleNamespace:
    return SimpleNamespace(http_servers=http_servers, tools=tools or [])


def _server(
    *,
    name: str = "hue",
    base_url: str = "https://api.example.com/v1",
    routes: list[HttpRouteSpec] | None = None,
    description: str | None = None,
) -> HttpServerSpec:
    return HttpServerSpec(
        name=name, base_url=base_url, routes=routes or [], description=description
    )


def _route(
    pattern: str,
    *,
    enabled: bool = True,
    policy: str | None = None,
    description: str | None = None,
) -> HttpRouteSpec:
    permission = HttpPermissionPolicy(type=policy) if policy else None  # type: ignore[arg-type]
    return HttpRouteSpec(
        path_pattern=pattern,
        enabled=enabled,
        permission_policy=permission,
        description=description,
    )


# ── _classify_permission ────────────────────────────────────────────────────


class TestClassifyPermission:
    def test_unknown_server_returns_none(self) -> None:
        agent = _agent(http_servers=[_server(name="hue", routes=[_route("/lights/*")])])
        assert _classify_permission({"server_ref": "missing", "path": "/lights/1"}, agent) is None

    def test_no_matching_route_returns_none(self) -> None:
        agent = _agent(http_servers=[_server(name="hue", routes=[_route("/lights/*")])])
        assert _classify_permission({"server_ref": "hue", "path": "/sensors/1"}, agent) is None

    def test_matched_route_with_always_ask(self) -> None:
        agent = _agent(
            http_servers=[
                _server(
                    name="hue",
                    routes=[_route("/lights/*/state", policy="always_ask")],
                )
            ]
        )
        assert (
            _classify_permission({"server_ref": "hue", "path": "/lights/1/state"}, agent)
            == "always_ask"
        )

    def test_matched_route_with_always_allow(self) -> None:
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*", policy="always_allow")])])
        assert (
            _classify_permission({"server_ref": "hue", "path": "/lights/1"}, agent)
            == "always_allow"
        )

    def test_matched_route_with_no_policy_returns_none(self) -> None:
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*")])])
        assert _classify_permission({"server_ref": "hue", "path": "/lights/1"}, agent) is None

    def test_disabled_route_does_not_match(self) -> None:
        agent = _agent(
            http_servers=[
                _server(
                    routes=[_route("/lights/*", enabled=False, policy="always_ask")],
                )
            ]
        )
        assert _classify_permission({"server_ref": "hue", "path": "/lights/1"}, agent) is None

    def test_bad_args_returns_none(self) -> None:
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*")])])
        assert _classify_permission({"server_ref": "hue"}, agent) is None
        assert _classify_permission({"server_ref": 123, "path": "/lights/1"}, agent) is None


# ── _classify_tool_call (loop.py) over http_request ──────────────────────────


class TestClassifyToolCallArguments:
    """``_classify_tool_call`` must accept both string- and dict-form
    ``function.arguments`` (providers differ on which shape they emit)."""

    @staticmethod
    def _make_agent() -> SimpleNamespace:
        return _agent(
            http_servers=[_server(routes=[_route("/lights/*", policy="always_allow")])],
            tools=[ToolSpec(type="http_request")],
        )

    def test_string_arguments_classify_as_immediate(self) -> None:
        from aios.harness.loop import _classify_tool_call

        tc = {
            "id": "c1",
            "type": "function",
            "function": {
                "name": "http_request",
                "arguments": '{"server_ref": "hue", "path": "/lights/1", "method": "GET"}',
            },
        }
        assert _classify_tool_call(tc, self._make_agent(), {}) == "immediate"

    def test_dict_arguments_classify_as_immediate(self) -> None:
        from aios.harness.loop import _classify_tool_call

        tc = {
            "id": "c1",
            "type": "function",
            "function": {
                "name": "http_request",
                "arguments": {"server_ref": "hue", "path": "/lights/1", "method": "GET"},
            },
        }
        assert _classify_tool_call(tc, self._make_agent(), {}) == "immediate"


# ── _decode_body ────────────────────────────────────────────────────────────


class TestDecodeBody:
    def _response(self, content_type: str, body: bytes) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": content_type},
            content=body,
        )

    def test_text_passthrough(self) -> None:
        r = self._response("text/plain", b"hello")
        assert _decode_body(r) == "hello"

    def test_json_passthrough(self) -> None:
        r = self._response("application/json", b'{"k": "v"}')
        assert _decode_body(r) == '{"k": "v"}'

    def test_binary_refused(self) -> None:
        r = self._response("application/octet-stream", b"\x00\x01\x02\x03")
        decoded = _decode_body(r)
        assert decoded.startswith("<binary content of type application/octet-stream")
        assert "4 bytes" in decoded

    def test_truncates_at_cap(self) -> None:
        large = b"x" * 200_000
        r = self._response("text/plain", large)
        assert len(_decode_body(r)) == 100_000


# ── http_request_handler ────────────────────────────────────────────────────


@pytest.fixture
def _stub_runtime() -> Iterator[SimpleNamespace]:
    """Patch out the runtime pool/crypto_box accessors used by the handler."""
    pool = MagicMock()
    crypto_box = MagicMock()
    with (
        patch("aios.tools.http_request.runtime.require_pool", return_value=pool),
        patch("aios.tools.http_request.runtime.require_crypto_box", return_value=crypto_box),
    ):
        yield SimpleNamespace(pool=pool, crypto_box=crypto_box)


def _patch_load_agent(agent: SimpleNamespace) -> Any:
    return patch(
        "aios.tools.http_request._load_session_agent",
        AsyncMock(return_value=(agent, "acc_test_stub")),
    )


def _patch_resolve_auth(headers: dict[str, str] | None = None) -> Any:
    return patch(
        "aios.tools.http_request.resolve_auth_for_target_url",
        AsyncMock(return_value=("vlt_x", headers or {})),
    )


def _patch_safe_url(safe: bool = True) -> Any:
    return patch("aios.tools.http_request.is_safe_url", return_value=safe)


def _make_stub_client(
    *,
    response: httpx.Response | None = None,
    exc: Exception | None = None,
    capture: dict[str, Any] | None = None,
) -> type:
    """Build a stand-in for ``httpx.AsyncClient`` that returns ``response``
    (or raises ``exc``) on ``.request``.  Uses the *real* AsyncClient class
    captured at import time so the patch doesn't recurse into itself."""

    class _Stub:
        def __init__(self, **_: Any) -> None:
            if response is not None:
                self._inner: httpx.AsyncClient | None = _REAL_ASYNC_CLIENT(
                    transport=httpx.MockTransport(lambda _req: response)
                )
            else:
                self._inner = None

        async def __aenter__(self) -> _Stub:
            return self

        async def __aexit__(self, *_: Any) -> None:
            if self._inner is not None:
                await self._inner.aclose()

        async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
            if capture is not None:
                capture.update({"method": method, "url": url, "kwargs": kwargs})
            if exc is not None:
                raise exc
            assert self._inner is not None
            return await self._inner.request(method, url, **kwargs)

    return _Stub


class TestHttpRequestHandler:
    async def test_unknown_server_returns_error(self, _stub_runtime: Any) -> None:
        agent = _agent(http_servers=[_server(name="hue", routes=[_route("/lights/*")])])
        with _patch_load_agent(agent):
            result = await http_request_handler(
                "sess_x",
                {"server_ref": "missing", "path": "/lights/1", "method": "GET"},
            )
        assert "error" in result
        assert "unknown server_ref" in result["error"]

    async def test_no_route_match_returns_error(self, _stub_runtime: Any) -> None:
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*")])])
        with _patch_load_agent(agent):
            result = await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/sensors/1", "method": "GET"},
            )
        assert "error" in result
        assert "does not match any enabled route" in result["error"]

    async def test_path_with_query_string_rejected(self, _stub_runtime: Any) -> None:
        """A ``path`` carrying a query string must be rejected even when the
        path-portion matches an enabled route. The route allowlist is a
        path-only gate; if a ``?`` segment slipped through, the operator's
        intent (e.g. an `/lights/*` read-only allowlist) would be bypassed
        because ``httpx`` parses the query off the constructed URL and sends
        it upstream, where APIs routinely interpret ``?action=delete`` /
        ``?force=true`` / ``?_method=DELETE`` as state-changing operations.

        Pre-fix ``match_glob`` treats ``?action=delete`` as literal trailing
        characters on the last segment, so ``/lights/1?action=delete`` matches
        ``/lights/*`` and the upstream receives the query string verbatim."""
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*")])])
        captured: dict[str, Any] = {}
        stub = _make_stub_client(
            response=httpx.Response(200, content=b""),
            capture=captured,
        )
        with (
            _patch_load_agent(agent),
            _patch_resolve_auth(),
            _patch_safe_url(),
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
        ):
            result = await http_request_handler(
                "sess_x",
                {
                    "server_ref": "hue",
                    "path": "/lights/1?action=delete",
                    "method": "GET",
                },
            )
        assert "error" in result, (
            f"path with query string must be rejected at the route gate; "
            f"got success {result!r}. Pre-fix symptom: the upstream URL "
            f"carried ?action=delete past an allowlist intended for "
            f"read-only operations."
        )
        # And the request must not have been dispatched.
        assert "url" not in captured, (
            f"upstream request was dispatched despite path containing a "
            f"query string; captured={captured!r}"
        )

    async def test_path_with_fragment_rejected(self, _stub_runtime: Any) -> None:
        """Same shape as query-string bypass for ``#fragment``. ``httpx``
        strips the fragment over the wire but the glob-match silently accepts
        it, leaving an inconsistency between what the route gate checks and
        what the upstream sees. Reject for consistency with ``?``."""
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*")])])
        with _patch_load_agent(agent):
            result = await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/lights/1#frag", "method": "GET"},
            )
        assert "error" in result

    async def test_successful_get(self, _stub_runtime: Any) -> None:
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*")])])
        response = httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=b'{"on": true}',
        )
        stub = _make_stub_client(response=response)
        with (
            _patch_load_agent(agent),
            _patch_resolve_auth({"Authorization": "Bearer xyz"}),
            _patch_safe_url(),
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
        ):
            result = await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/lights/1", "method": "GET"},
            )
        assert result["status"] == 200
        assert result["body"] == '{"on": true}'

    async def test_upstream_4xx_returns_status(self, _stub_runtime: Any) -> None:
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*")])])
        response = httpx.Response(
            404,
            headers={"content-type": "text/plain"},
            content=b"not found",
        )
        stub = _make_stub_client(response=response)
        with (
            _patch_load_agent(agent),
            _patch_resolve_auth(),
            _patch_safe_url(),
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
        ):
            result = await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/lights/99", "method": "GET"},
            )
        # Upstream 4xx surfaces via `status`, not the `error` envelope —
        # the model reads the status code and can react accordingly.
        assert result["status"] == 404
        assert result["body"] == "not found"
        assert "error" not in result

    async def test_caller_authorization_header_dropped(self, _stub_runtime: Any) -> None:
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*")])])
        captured: dict[str, Any] = {}
        response = httpx.Response(200, content=b"ok")
        stub = _make_stub_client(response=response, capture=captured)
        with (
            _patch_load_agent(agent),
            _patch_resolve_auth({"Authorization": "Bearer broker-token"}),
            _patch_safe_url(),
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
        ):
            await http_request_handler(
                "sess_x",
                {
                    "server_ref": "hue",
                    "path": "/lights/1",
                    "method": "GET",
                    "headers": {"Authorization": "Bearer attacker", "X-Other": "ok"},
                },
            )
        sent = captured["kwargs"]["headers"]
        assert sent["Authorization"] == "Bearer broker-token"
        assert sent["X-Other"] == "ok"

    async def test_caller_host_header_dropped(self, _stub_runtime: Any) -> None:
        """An agent-supplied ``Host`` header must not reach the dispatched
        request. The route allowlist scopes by ``base_url`` — i.e. by the
        TCP host the connection lands on — but with name-based virtual
        hosting (a near-universal production HTTP pattern: NGINX, Cloudflare,
        AWS ALB, Kubernetes Ingress) the upstream routes by the ``Host``
        header, not by the destination IP. A caller_headers passthrough of
        ``Host: admin.internal`` would land the request on an upstream
        vhost the operator never approved, bypassing the route allowlist's
        intent.

        Same shape as #485 (query string in path) and #477 (Telegram
        reaction emoji): a worker-managed protocol element (here: the
        TCP host derived from base_url) gets silently overridden by an
        agent-controlled field. Strip ``Host`` at the same site that
        strips ``Authorization``.
        """
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*")])])
        captured: dict[str, Any] = {}
        response = httpx.Response(200, content=b"ok")
        stub = _make_stub_client(response=response, capture=captured)
        with (
            _patch_load_agent(agent),
            _patch_resolve_auth({"Authorization": "Bearer broker-token"}),
            _patch_safe_url(),
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
        ):
            await http_request_handler(
                "sess_x",
                {
                    "server_ref": "hue",
                    "path": "/lights/1",
                    "method": "GET",
                    # Case-mixed to verify the strip is case-insensitive,
                    # matching the ``Authorization`` filter's semantics.
                    "headers": {"HoSt": "admin.internal", "X-Other": "ok"},
                },
            )
        sent = captured["kwargs"]["headers"]
        # The agent's Host (under any casing) must not be present.
        assert all(k.lower() != "host" for k in sent), (
            f"caller Host header must be stripped before dispatch; got {sent!r}. "
            f"Pre-fix symptom: the only stripped header was Authorization, so "
            f"a Host override slipped through to httpx and onto the wire — "
            f"bypassing the operator's base_url scoping under name-based vhosting."
        )
        # Other agent-supplied headers should still pass through.
        assert sent["X-Other"] == "ok"

    async def test_timeout_returns_error(self, _stub_runtime: Any) -> None:
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*")])])
        stub = _make_stub_client(exc=httpx.ReadTimeout("timed out"))
        with (
            _patch_load_agent(agent),
            _patch_resolve_auth(),
            _patch_safe_url(),
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
        ):
            result = await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/lights/1", "method": "GET"},
            )
        assert "error" in result
        assert "timed out" in result["error"].lower()

    async def test_path_with_dot_dot_segment_rejected(self, _stub_runtime: Any) -> None:
        """Path-traversal bypass of the route allowlist via ``..`` segments.

        The route gate uses ``match_glob``, where ``*`` matches any single
        path segment — including a literal ``..``. The constructed URL
        ``base_url/lights/..`` is then normalized by httpx (per RFC 3986
        section 5.2.4) into ``base_url/``, reaching the server root despite
        the operator's intent to scope the agent to ``/lights/*``.

        Same shape as the ``?`` and ``#`` rejection tests: a path the route
        allowlist accepts produces, after httpx normalization, an upstream
        request to a path the allowlist does NOT cover. Reject ``..`` at
        the same site so the gate's intent is the gate's effect.

        Pre-fix symptom: the request dispatches and the upstream sees the
        root path, bypassing the allowlist that scoped to ``/lights/*``.
        """
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*")])])
        captured: dict[str, Any] = {}
        stub = _make_stub_client(
            response=httpx.Response(200, content=b""),
            capture=captured,
        )
        with (
            _patch_load_agent(agent),
            _patch_resolve_auth(),
            _patch_safe_url(),
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
        ):
            result = await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/lights/..", "method": "GET"},
            )
        assert "error" in result, (
            f"path with '..' segment must be rejected at the route gate; "
            f"got success {result!r}. Pre-fix symptom: glob '/lights/*' "
            f"matches 'lights/..' (the '*' eats the '..' segment), httpx "
            f"normalizes 'base_url/lights/..' to 'base_url/', and the "
            f"upstream receives a root-path request the allowlist never "
            f"approved."
        )
        # And the request must not have been dispatched.
        assert "url" not in captured, (
            f"upstream request was dispatched despite path containing '..' "
            f"traversal; captured={captured!r}"
        )

    async def test_path_with_single_dot_segment_rejected(self, _stub_runtime: Any) -> None:
        """Same shape as ``..`` but for single-``.`` segments.

        RFC 3986 §5.2.4 also strips ``.`` segments: ``base_url/lights/./state``
        normalizes to ``base_url/lights/state``. Against a
        ``/lights/*/state`` allowlist (operator's intent: only routes that
        include a real id segment between ``lights`` and ``state``), the
        agent can send ``/lights/./state``: ``*`` matches the literal ``.``
        and the glob accepts, but the upstream then receives ``/lights/state``
        — a path the allowlist does NOT cover.

        Same root cause as ``..``; both are dot-segment normalization
        artefacts. The fix at :func:`_path_rejected_reason` covers both.
        """
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*/state")])])
        captured: dict[str, Any] = {}
        stub = _make_stub_client(
            response=httpx.Response(200, content=b""),
            capture=captured,
        )
        with (
            _patch_load_agent(agent),
            _patch_resolve_auth(),
            _patch_safe_url(),
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
        ):
            result = await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/lights/./state", "method": "GET"},
            )
        assert "error" in result, (
            f"path with single '.' segment must be rejected at the route "
            f"gate; got success {result!r}. Pre-fix symptom: glob "
            f"'/lights/*/state' matches 'lights/./state', httpx normalizes "
            f"to 'base_url/lights/state', and the upstream receives a path "
            f"the allowlist's *-segment scoping did not cover."
        )
        assert "url" not in captured, (
            f"upstream request was dispatched despite '.' segment; captured={captured!r}"
        )

    async def test_path_with_embedded_dot_dot_rejected(self, _stub_runtime: Any) -> None:
        """``..`` segments embedded mid-path also escape the allowlist.

        With a ``/lights/**`` (multi-segment) allowlist, an agent can send
        ``/lights/../admin/delete``: the glob matches because ``**`` accepts
        any segments, but httpx normalizes the URL to ``base_url/admin/delete``
        — reaching an admin endpoint the operator never granted.

        Distinct from the single-segment case because ``**`` is the only
        glob form that admits the embedded variant. Worth a separate test
        so a fix that only handles the single-``*`` case still goes red here.
        """
        agent = _agent(http_servers=[_server(routes=[_route("/lights/**")])])
        captured: dict[str, Any] = {}
        stub = _make_stub_client(
            response=httpx.Response(200, content=b""),
            capture=captured,
        )
        with (
            _patch_load_agent(agent),
            _patch_resolve_auth(),
            _patch_safe_url(),
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
        ):
            result = await http_request_handler(
                "sess_x",
                {
                    "server_ref": "hue",
                    "path": "/lights/../admin/delete",
                    "method": "GET",
                },
            )
        assert "error" in result, (
            f"path with embedded '..' must be rejected at the route gate; "
            f"got success {result!r}. Pre-fix symptom: glob '/lights/**' "
            f"matches 'lights/../admin/delete', httpx normalizes the URL "
            f"to 'base_url/admin/delete', and the upstream receives an "
            f"admin-path request the allowlist never approved."
        )
        assert "url" not in captured, (
            f"upstream request was dispatched despite embedded '..' "
            f"traversal; captured={captured!r}"
        )
