"""Unit tests for the ``http_request`` built-in tool."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from aios.models.agents import (
    GenericChildBinding,
    HttpPermissionPolicy,
    HttpRouteSpec,
    HttpServerSpec,
    StepSurface,
    ToolSpec,
)
from aios.pinned_transport import PinnedTransport
from aios.tools.http_request import (
    _classify_permission,
    _decode_body,
    _do_http_request,
    http_request_handler,
)
from aios.tools.invoke import ToolBail

# Capture the real httpx.AsyncClient before any patch in this file replaces it.
# A stub that calls ``httpx.AsyncClient(...)`` after a patch would otherwise
# recurse into itself.
_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _agent(
    *, http_servers: list[HttpServerSpec], tools: list[ToolSpec] | None = None
) -> StepSurface:
    """A minimal ``StepSurface`` for the http_request path (which reads only
    ``http_servers``/``tools``). The binding kind is irrelevant here."""
    return StepSurface(
        model="test/dummy",
        system="",
        tools=tools or [],
        skills=[],
        mcp_servers=[],
        http_servers=http_servers,
        litellm_extra={},
        window_min=1000,
        window_max=100000,
        binding=GenericChildBinding(session_id="ses_test"),
    )


def _server(
    *,
    name: str = "hue",
    base_url: str = "https://api.example.com/v1",
    routes: list[HttpRouteSpec] | None = None,
    description: str | None = None,
    suppressed_response_status: int = 200,
) -> HttpServerSpec:
    return HttpServerSpec(
        name=name,
        base_url=base_url,
        routes=routes or [],
        description=description,
        suppressed_response_status=suppressed_response_status,
    )


def _route(
    pattern: str,
    *,
    enabled: bool = True,
    policy: str | None = None,
    description: str | None = None,
    methods: list[str] | None = None,
    allow_query: bool = False,
    suppress: bool | None = None,
) -> HttpRouteSpec:
    permission = HttpPermissionPolicy(type=policy) if policy else None
    return HttpRouteSpec(
        path_pattern=pattern,
        enabled=enabled,
        permission_policy=permission,
        description=description,
        methods=cast(Any, methods),
        allow_query=allow_query,
        suppress=suppress,
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
            _classify_permission(
                {"server_ref": "hue", "path": "/lights/1/state", "method": "GET"}, agent
            )
            == "always_ask"
        )

    def test_matched_route_with_always_allow(self) -> None:
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*", policy="always_allow")])])
        assert (
            _classify_permission({"server_ref": "hue", "path": "/lights/1", "method": "GET"}, agent)
            == "always_allow"
        )

    def test_matched_route_with_no_policy_returns_none(self) -> None:
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*")])])
        assert (
            _classify_permission({"server_ref": "hue", "path": "/lights/1", "method": "GET"}, agent)
            is None
        )

    def test_disabled_route_does_not_match(self) -> None:
        agent = _agent(
            http_servers=[
                _server(
                    routes=[_route("/lights/*", enabled=False, policy="always_ask")],
                )
            ]
        )
        assert (
            _classify_permission({"server_ref": "hue", "path": "/lights/1", "method": "GET"}, agent)
            is None
        )

    def test_bad_args_returns_none(self) -> None:
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*")])])
        assert _classify_permission({"server_ref": "hue"}, agent) is None
        assert _classify_permission({"server_ref": 123, "path": "/lights/1"}, agent) is None

    def test_method_scoped_route_classifies_only_matching_method(self) -> None:
        # A route scoped to POST with always_ask classifies a POST but not a GET
        # (GET doesn't match the route at all → None → handler emits a typed error).
        agent = _agent(
            http_servers=[
                _server(routes=[_route("/lights/*", policy="always_ask", methods=["POST"])])
            ]
        )
        assert (
            _classify_permission(
                {"server_ref": "hue", "path": "/lights/1", "method": "POST"}, agent
            )
            == "always_ask"
        )
        assert (
            _classify_permission({"server_ref": "hue", "path": "/lights/1", "method": "GET"}, agent)
            is None
        )

    def test_missing_method_returns_none(self) -> None:
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*", policy="always_ask")])])
        assert _classify_permission({"server_ref": "hue", "path": "/lights/1"}, agent) is None


# ── _classify_tool_call (loop.py) over http_request ──────────────────────────


class TestClassifyToolCallArguments:
    """``_classify_tool_call`` must accept both string- and dict-form
    ``function.arguments`` (providers differ on which shape they emit)."""

    @staticmethod
    def _make_agent() -> StepSurface:
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
        assert _decode_body(r) == ("hello", False)

    def test_json_passthrough(self) -> None:
        r = self._response("application/json", b'{"k": "v"}')
        assert _decode_body(r) == ('{"k": "v"}', False)

    def test_binary_refused(self) -> None:
        r = self._response("application/octet-stream", b"\x00\x01\x02\x03")
        decoded, truncated = _decode_body(r)
        assert decoded.startswith("<binary content of type application/octet-stream")
        assert "4 bytes" in decoded
        # A refused-binary marker is the full honest value, not a truncated body.
        assert truncated is False

    def test_under_cap_not_truncated(self) -> None:
        # Just under the default ~1 MB cap: full body, no truncated flag.
        body = b"x" * 999_999
        r = self._response("text/plain", body)
        decoded, truncated = _decode_body(r)
        assert len(decoded) == 999_999
        assert truncated is False

    def test_truncates_at_cap_and_signals(self) -> None:
        # Over the default cap: body is cut AND truncated is reported True so the
        # caller can fail loud instead of silently parsing a half-body (aios#1294).
        large = b"x" * 2_000_000
        r = self._response("text/plain", large)
        decoded, truncated = _decode_body(r)
        assert len(decoded) == 1_000_000
        assert truncated is True

    def test_cap_is_env_configurable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AIOS_HTTP_RESPONSE_MAX_CHARS", "50")
        r = self._response("text/plain", b"y" * 200)
        decoded, truncated = _decode_body(r)
        assert len(decoded) == 50
        assert truncated is True

    def test_bad_env_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # An unparseable / non-positive override must NOT disable the cap.
        for bad in ("not-a-number", "0", "-5"):
            monkeypatch.setenv("AIOS_HTTP_RESPONSE_MAX_CHARS", bad)
            r = self._response("text/plain", b"z" * 1_500_000)
            decoded, truncated = _decode_body(r)
            assert len(decoded) == 1_000_000
            assert truncated is True


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


def _patch_load_agent(agent: StepSurface, outbound_suppression: str = "off") -> Any:
    return patch(
        "aios.tools.http_request._load_session_agent",
        AsyncMock(return_value=(agent, "acc_test_stub", outbound_suppression)),
    )


def _patch_resolve_auth(headers: dict[str, str] | None = None) -> Any:
    return patch(
        "aios.tools.http_request.resolve_auth_for_target_url",
        AsyncMock(return_value=("vlt_x", headers or {})),
    )


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
        def __init__(self, **kw: Any) -> None:
            if capture is not None:
                capture["init_kwargs"] = kw
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
        # Post-#1680: an expected http_request failure raises ``ToolBail`` (one typed
        # failure channel) rather than returning a bare ``{"error": ...}`` dict.
        with _patch_load_agent(agent), pytest.raises(ToolBail) as excinfo:
            await http_request_handler(
                "sess_x",
                {"server_ref": "missing", "path": "/lights/1", "method": "GET"},
            )
        assert "unknown server_ref" in excinfo.value.message

    async def test_no_route_match_returns_error(self, _stub_runtime: Any) -> None:
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*")])])
        with _patch_load_agent(agent), pytest.raises(ToolBail) as excinfo:
            await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/sensors/1", "method": "GET"},
            )
        assert "does not match any enabled route" in excinfo.value.message

    async def test_client_is_mounted_with_pinned_transport(self, _stub_runtime: Any) -> None:
        """The SSRF guard lives in ``PinnedTransport`` (resolve-validate-pin at
        connect time — see ``tests/unit/test_pinned_transport.py`` for the
        behavioral rebinding assertions). The stub client swallows the transport,
        so what this test can honestly pin down is *construction*: the dispatch
        client must be mounted with ``PinnedTransport``, catching any future
        refactor that silently drops the guard."""
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*")])])
        captured: dict[str, Any] = {}
        stub = _make_stub_client(response=httpx.Response(200, content=b""), capture=captured)
        with (
            _patch_load_agent(agent),
            _patch_resolve_auth(),
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
        ):
            await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/lights/1", "method": "GET"},
            )
        assert isinstance(captured["init_kwargs"]["transport"], PinnedTransport)

    async def test_plaintext_http_with_credential_is_refused(self, _stub_runtime: Any) -> None:
        """SECURITY-02: a vault credential must not travel cleartext. A server
        whose base_url is plain http + an attached Authorization header is
        refused before any dispatch."""
        agent = _agent(
            http_servers=[
                _server(base_url="http://api.example.com/v1", routes=[_route("/lights/*")])
            ]
        )
        captured: dict[str, Any] = {}
        stub = _make_stub_client(response=httpx.Response(200, content=b""), capture=captured)
        with (
            _patch_load_agent(agent),
            _patch_resolve_auth({"Authorization": "Bearer x"}),
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
            pytest.raises(ToolBail) as excinfo,
        ):
            await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/lights/1", "method": "GET"},
            )
        assert "plaintext" in excinfo.value.message
        assert "url" not in captured  # refused before any upstream dispatch

    async def test_https_with_credential_is_not_refused_by_scheme_rule(
        self, _stub_runtime: Any
    ) -> None:
        """The same credentialed request over https is not refused by this rule
        — it dispatches like any other happy path."""
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*")])])
        captured: dict[str, Any] = {}
        stub = _make_stub_client(response=httpx.Response(200, content=b""), capture=captured)
        with (
            _patch_load_agent(agent),
            _patch_resolve_auth({"Authorization": "Bearer x"}),
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
        ):
            result = await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/lights/1", "method": "GET"},
            )
        assert result["status"] == 200

    async def test_plaintext_http_without_credential_is_allowed(self, _stub_runtime: Any) -> None:
        """A non-credentialed plaintext call is allowed — the rule gates on a
        credential actually being attached, not on the scheme alone."""
        agent = _agent(
            http_servers=[
                _server(base_url="http://api.example.com/v1", routes=[_route("/lights/*")])
            ]
        )
        captured: dict[str, Any] = {}
        stub = _make_stub_client(response=httpx.Response(200, content=b""), capture=captured)
        with (
            _patch_load_agent(agent),
            _patch_resolve_auth(),  # ("vlt_x", {}) — no auth header attached
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
        ):
            result = await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/lights/1", "method": "GET"},
            )
        assert result["status"] == 200

    async def test_method_not_allowed_returns_error(self, _stub_runtime: Any) -> None:
        # A route scoped to GET refuses a POST at the gate (no upstream dispatch).
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*", methods=["GET"])])])
        captured: dict[str, Any] = {}
        stub = _make_stub_client(response=httpx.Response(200, content=b""), capture=captured)
        with (
            _patch_load_agent(agent),
            _patch_resolve_auth(),
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
            pytest.raises(ToolBail) as excinfo,
        ):
            await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/lights/1", "method": "POST"},
            )
        assert "POST" in excinfo.value.message
        assert "does not match any enabled route" in excinfo.value.message
        assert "url" not in captured  # not dispatched upstream

    async def test_method_scoped_route_allows_matching_method(self, _stub_runtime: Any) -> None:
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*", methods=["GET"])])])
        response = httpx.Response(
            200, headers={"content-type": "application/json"}, content=b'{"on": true}'
        )
        stub = _make_stub_client(response=response)
        with (
            _patch_load_agent(agent),
            _patch_resolve_auth({"Authorization": "Bearer xyz"}),
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
        ):
            result = await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/lights/1", "method": "GET"},
            )
        assert result["status"] == 200

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
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
            pytest.raises(ToolBail),
        ):
            await http_request_handler(
                "sess_x",
                {
                    "server_ref": "hue",
                    "path": "/lights/1?action=delete",
                    "method": "GET",
                },
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
        with _patch_load_agent(agent), pytest.raises(ToolBail):
            await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/lights/1#frag", "method": "GET"},
            )

    async def test_query_string_allowed_when_route_opts_in(self, _stub_runtime: Any) -> None:
        """#1156: a route with ``allow_query=True`` permits a query string and
        dispatches it upstream verbatim. The GitHub ``/repos/**`` route opts in so
        the dev-pipeline can follow ``?per_page``/``?page`` pagination to read a full
        comment thread. The path portion is still glob-matched (query stripped first)."""
        agent = _agent(http_servers=[_server(routes=[_route("/repos/**", allow_query=True)])])
        captured: dict[str, Any] = {}
        stub = _make_stub_client(response=httpx.Response(200, content=b"[]"), capture=captured)
        with (
            _patch_load_agent(agent),
            _patch_resolve_auth(),
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
        ):
            result = await http_request_handler(
                "sess_x",
                {
                    "server_ref": "hue",
                    "path": "/repos/o/r/issues/5/comments?per_page=100&page=2",
                    "method": "GET",
                },
            )
        assert result["status"] == 200, result
        # The query string reached the upstream URL (httpx parses it off the path).
        assert "per_page=100" in captured["url"] and "page=2" in captured["url"], captured

    async def test_query_string_rejected_when_route_does_not_opt_in(
        self, _stub_runtime: Any
    ) -> None:
        """The #485 default holds for every route that does NOT set ``allow_query``:
        a ``?action=delete`` must not slip past a read-only ``/lights/*`` allowlist,
        and the upstream request must never be dispatched."""
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*")])])
        captured: dict[str, Any] = {}
        stub = _make_stub_client(response=httpx.Response(200, content=b""), capture=captured)
        with (
            _patch_load_agent(agent),
            _patch_resolve_auth(),
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
            pytest.raises(ToolBail),
        ):
            await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/lights/1?action=delete", "method": "GET"},
            )
        assert "url" not in captured, captured

    async def test_query_does_not_let_unmatched_path_through(self, _stub_runtime: Any) -> None:
        """Even with ``allow_query=True``, the query is stripped before the glob-match,
        so a path the route pattern does not cover is still refused (the query never
        widens the path-dimension grant)."""
        agent = _agent(http_servers=[_server(routes=[_route("/repos/**", allow_query=True)])])
        with _patch_load_agent(agent), pytest.raises(ToolBail) as excinfo:
            await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/orgs/o/members?per_page=100", "method": "GET"},
            )
        assert "does not match any enabled route" in excinfo.value.message

    async def test_fragment_rejected_even_when_query_allowed(self, _stub_runtime: Any) -> None:
        """``allow_query`` opts into a query string only — a ``#fragment`` is still
        rejected (httpx strips it with no route equivalent)."""
        agent = _agent(http_servers=[_server(routes=[_route("/repos/**", allow_query=True)])])
        with _patch_load_agent(agent), pytest.raises(ToolBail):
            await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/repos/o/r/issues/5#frag", "method": "GET"},
            )

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
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
        ):
            result = await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/lights/1", "method": "GET"},
            )
        assert result["status"] == 200
        assert result["body"] == '{"on": true}'
        # A complete body never carries the truncated flag.
        assert "truncated" not in result

    async def test_duplicate_response_headers_preserved(self, _stub_runtime: Any) -> None:
        """``headers`` is a list of ``[name, value]`` pairs so every occurrence
        survives in wire order — the contract the module docstring promises. A dict
        would collapse duplicate names, silently dropping all but the last; HTTP
        *requires* some be multi-valued (the canonical ``Set-Cookie``, but also
        ``Link`` etc.). We interleave distinct names with TWO duplicate names — one
        cookie, one non-cookie — so the assertion pins both cross-name order and the
        pair-list's generality (a fix that special-cased only ``Set-Cookie`` and
        ``dict()``'d the rest would fail here)."""
        sent = [
            ["x-first", "1"],
            ["set-cookie", "a=1"],
            ["link", "</p1>; rel=next"],
            ["set-cookie", "b=2"],
            ["link", "</p2>; rel=prev"],
            ["x-last", "9"],
        ]
        response = httpx.Response(200, headers=[(k, v) for k, v in sent], content=b"ok")
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*")])])
        stub = _make_stub_client(response=response)
        with (
            _patch_load_agent(agent),
            _patch_resolve_auth(),
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
        ):
            result = await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/lights/1", "method": "GET"},
            )
        # Filter to just the names we set (httpx auto-injects content-length); the
        # surviving sublist must equal our input verbatim — same order, both dups.
        names = {k for k, _ in sent}
        preserved = [[k.lower(), v] for k, v in result["headers"] if k.lower() in names]
        assert preserved == sent

    async def test_over_cap_body_sets_truncated_flag(
        self, _stub_runtime: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An over-cap 2xx body is cut AND the result carries ``truncated: True``
        # so the caller can fail loud rather than parse a half-body (aios#1294).
        monkeypatch.setenv("AIOS_HTTP_RESPONSE_MAX_CHARS", "100")
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*")])])
        response = httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=b"[" + b"0," * 500 + b"0]",
        )
        stub = _make_stub_client(response=response)
        with (
            _patch_load_agent(agent),
            _patch_resolve_auth({"Authorization": "Bearer xyz"}),
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
        ):
            result = await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/lights/1", "method": "GET"},
            )
        assert result["status"] == 200
        assert len(result["body"]) == 100
        assert result["truncated"] is True

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
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
            pytest.raises(ToolBail) as excinfo,
        ):
            await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/lights/1", "method": "GET"},
            )
        assert "timed out" in excinfo.value.message.lower()

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
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
            pytest.raises(ToolBail),
        ):
            await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/lights/..", "method": "GET"},
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
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
            pytest.raises(ToolBail),
        ):
            await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/lights/./state", "method": "GET"},
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
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
            pytest.raises(ToolBail),
        ):
            await http_request_handler(
                "sess_x",
                {
                    "server_ref": "hue",
                    "path": "/lights/../admin/delete",
                    "method": "GET",
                },
            )
        assert "url" not in captured, (
            f"upstream request was dispatched despite embedded '..' "
            f"traversal; captured={captured!r}"
        )


# ── _do_http_request (owner-agnostic core; the session/run shared entry) ───────


class TestDoHttpRequest:
    """The extracted core takes ``servers`` + an injected ``resolve_auth`` directly, so the
    session handler and the workflow-run dispatcher exercise one code path."""

    async def test_authors_header_from_injected_resolver(self) -> None:
        servers = [
            _server(name="api", base_url="https://api.example.com/v1", routes=[_route("/lights/*")])
        ]
        capture: dict[str, Any] = {}

        async def resolve_auth(base_url: str) -> tuple[str | None, dict[str, str]]:
            assert base_url == "https://api.example.com/v1"
            return ("vlt", {"Authorization": "Bearer RUNTOKEN"})

        with (
            patch(
                "aios.tools.http_request.httpx.AsyncClient",
                _make_stub_client(response=httpx.Response(200, json={"ok": True}), capture=capture),
            ),
        ):
            out = await _do_http_request(
                servers=servers,
                arguments={"server_ref": "api", "path": "/lights/1", "method": "GET"},
                resolve_auth=resolve_auth,
            )
        assert out["status"] == 200
        assert capture["kwargs"]["headers"]["Authorization"] == "Bearer RUNTOKEN"

    async def test_unknown_server_ref_raises_toolbail(self) -> None:
        # Post-#1680: the shared core signals an expected failure by raising
        # ``ToolBail``. The session path lets it propagate (single writer stamps
        # ``is_error``); the workflow-run path re-materializes it as an
        # ``{"error": ...}`` value at the ``invoke_run_tool`` seam (``_values``).
        async def resolve_auth(base_url: str) -> tuple[str | None, dict[str, str]]:
            return (None, {})

        with pytest.raises(ToolBail) as excinfo:
            await _do_http_request(
                servers=[],
                arguments={"server_ref": "missing", "path": "/x", "method": "GET"},
                resolve_auth=resolve_auth,
            )
        assert "unknown server_ref" in excinfo.value.message


# ── outbound suppression (#710) ─────────────────────────────────────────────


class TestOutboundSuppressionHttp:
    """When a session runs with ``outbound_suppression == 'on'`` the HTTP broker
    intercepts writes (synthesized success + audit event, no upstream dispatch)
    and lets reads through against real credentials."""

    async def test_write_suppressed_no_dispatch_and_records_event(self, _stub_runtime: Any) -> None:
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*")])])
        captured: dict[str, Any] = {}
        stub = _make_stub_client(response=httpx.Response(200, content=b""), capture=captured)
        record = AsyncMock()
        with (
            _patch_load_agent(agent, outbound_suppression="on"),
            _patch_resolve_auth(),
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
            patch(
                "aios.tools.http_request.outbound_suppression_service.record_http_suppression",
                record,
            ),
        ):
            result = await http_request_handler(
                "sess_x",
                {
                    "server_ref": "hue",
                    "path": "/lights/1",
                    "method": "POST",
                    "body": '{"on": true}',
                },
            )
        # Synthesized success — looks like a real response, no upstream dispatch.
        assert result == {"status": 200, "headers": [], "body": ""}
        assert "url" not in captured
        # Audit event recorded with the un-suppressed intent.
        record.assert_awaited_once()
        assert record.await_args is not None
        kwargs = record.await_args.kwargs
        assert kwargs["method"] == "POST"
        assert kwargs["path"] == "/lights/1"
        assert kwargs["body"] == '{"on": true}'
        assert kwargs["server_ref"] == "hue"

    async def test_read_passes_through_under_suppression(self, _stub_runtime: Any) -> None:
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*")])])
        response = httpx.Response(
            200, headers={"content-type": "application/json"}, content=b'{"on": true}'
        )
        captured: dict[str, Any] = {}
        stub = _make_stub_client(response=response, capture=captured)
        record = AsyncMock()
        with (
            _patch_load_agent(agent, outbound_suppression="on"),
            _patch_resolve_auth(),
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
            patch(
                "aios.tools.http_request.outbound_suppression_service.record_http_suppression",
                record,
            ),
        ):
            result = await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/lights/1", "method": "GET"},
            )
        # Real dispatch, real response; nothing recorded.
        assert captured["method"] == "GET"
        assert result["status"] == 200
        record.assert_not_awaited()

    async def test_per_route_suppress_override_suppresses_get(self, _stub_runtime: Any) -> None:
        # A GET that the operator marked suppress=True (a side-effecting read).
        agent = _agent(http_servers=[_server(routes=[_route("/trigger/*", suppress=True)])])
        captured: dict[str, Any] = {}
        stub = _make_stub_client(response=httpx.Response(200, content=b""), capture=captured)
        record = AsyncMock()
        with (
            _patch_load_agent(agent, outbound_suppression="on"),
            _patch_resolve_auth(),
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
            patch(
                "aios.tools.http_request.outbound_suppression_service.record_http_suppression",
                record,
            ),
        ):
            result = await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/trigger/1", "method": "GET"},
            )
        assert result == {"status": 200, "headers": [], "body": ""}
        assert "url" not in captured
        record.assert_awaited_once()

    async def test_per_route_suppress_false_lets_post_through(self, _stub_runtime: Any) -> None:
        # A read-only POST (a query endpoint) the operator marked suppress=False.
        agent = _agent(http_servers=[_server(routes=[_route("/graphql", suppress=False)])])
        captured: dict[str, Any] = {}
        stub = _make_stub_client(response=httpx.Response(200, content=b"{}"), capture=captured)
        record = AsyncMock()
        with (
            _patch_load_agent(agent, outbound_suppression="on"),
            _patch_resolve_auth(),
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
            patch(
                "aios.tools.http_request.outbound_suppression_service.record_http_suppression",
                record,
            ),
        ):
            result = await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/graphql", "method": "POST"},
            )
        assert captured["method"] == "POST"
        assert result["status"] == 200
        record.assert_not_awaited()

    async def test_suppressed_status_is_configurable(self, _stub_runtime: Any) -> None:
        agent = _agent(
            http_servers=[_server(routes=[_route("/things")], suppressed_response_status=201)]
        )
        record = AsyncMock()
        with (
            _patch_load_agent(agent, outbound_suppression="on"),
            _patch_resolve_auth(),
            patch(
                "aios.tools.http_request.outbound_suppression_service.record_http_suppression",
                record,
            ),
        ):
            result = await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/things", "method": "POST"},
            )
        assert result["status"] == 201
        assert record.await_args is not None
        assert record.await_args.kwargs["synthesized_status"] == 201

    async def test_suppression_off_does_not_gate(self, _stub_runtime: Any) -> None:
        # The default mode never consults suppression — a write dispatches.
        agent = _agent(http_servers=[_server(routes=[_route("/lights/*")])])
        captured: dict[str, Any] = {}
        stub = _make_stub_client(response=httpx.Response(200, content=b""), capture=captured)
        record = AsyncMock()
        with (
            _patch_load_agent(agent, outbound_suppression="off"),
            _patch_resolve_auth(),
            patch("aios.tools.http_request.httpx.AsyncClient", stub),
            patch(
                "aios.tools.http_request.outbound_suppression_service.record_http_suppression",
                record,
            ),
        ):
            await http_request_handler(
                "sess_x",
                {"server_ref": "hue", "path": "/lights/1", "method": "POST"},
            )
        assert captured["method"] == "POST"
        record.assert_not_awaited()
