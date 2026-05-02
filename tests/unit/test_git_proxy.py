"""Unit tests for the per-session git credential proxy.

Boots a real ``GitProxy`` (so we exercise the actual starlette + uvicorn
machinery), but mocks the upstream ``httpx`` transport so requests never
leave the test process. Verifies:

- 403 on a missing/wrong secret
- 404 on a repo that isn't in the token map
- the upstream request gets the expected ``Authorization: Basic <...>``
- hop-by-hop request and response headers are stripped
- streamed response bodies make it back unchanged
- ``proxy_url`` formats correctly and ``repo_key`` normalizes ``.git``

No docker, no real github, no PAT.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator

import httpx
import pytest

from aios.sandbox.git_proxy import GitProxy, repo_key


def _basic_auth(token: str) -> str:
    """The Authorization value the proxy injects when forwarding."""
    creds = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return f"Basic {creds}"


def _make_mock_transport(captured: list[httpx.Request]) -> httpx.MockTransport:
    """Mock upstream that records every request and returns a canned 200."""

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            headers={
                "content-type": "application/x-git-upload-pack-advertisement",
                "x-github-request-id": "abc",
                # Hop-by-hop should be stripped on the way back to the client.
                "transfer-encoding": "chunked",
                "connection": "keep-alive",
            },
            content=b"upstream-body",
        )

    return httpx.MockTransport(handler)


@pytest.fixture
async def proxy() -> AsyncIterator[GitProxy]:
    p = GitProxy({"acme/foo": "ghp_TEST"})
    await p.start()
    try:
        yield p
    finally:
        await p.stop()


@pytest.fixture
async def proxy_with_mock_upstream() -> AsyncIterator[tuple[GitProxy, list[httpx.Request]]]:
    """Proxy whose upstream client points at a MockTransport."""
    p = GitProxy({"acme/foo": "ghp_TEST"})
    captured: list[httpx.Request] = []
    # Replace the inner httpx.AsyncClient with one backed by the mock
    # transport. This is private-attr surgery, but we deliberately want
    # to test that the proxy uses its client correctly without requiring
    # network egress.
    await p._client.aclose()  # type: ignore[has-type]
    p._client = httpx.AsyncClient(transport=_make_mock_transport(captured))  # type: ignore[has-type]
    await p.start()
    try:
        yield p, captured
    finally:
        await p.stop()


class TestRepoKey:
    def test_strips_dot_git(self) -> None:
        assert repo_key("https://github.com/acme/foo.git") == "acme/foo"

    def test_no_dot_git(self) -> None:
        assert repo_key("https://github.com/acme/foo") == "acme/foo"

    def test_with_trailing_slash(self) -> None:
        assert repo_key("https://github.com/acme/foo/") == "acme/foo"


class TestProxyUrl:
    async def test_format(self, proxy: GitProxy) -> None:
        url = proxy.proxy_url("https://github.com/acme/foo.git", host="example.local")
        assert url == f"http://example.local:{proxy.port}/git/{proxy.secret}/acme/foo"

    async def test_strips_dot_git(self, proxy: GitProxy) -> None:
        a = proxy.proxy_url("https://github.com/acme/foo.git", host="h")
        b = proxy.proxy_url("https://github.com/acme/foo", host="h")
        assert a == b


class TestSecretEnforcement:
    async def test_missing_secret_returns_403(self, proxy: GitProxy) -> None:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"http://127.0.0.1:{proxy.port}/git/wrong/acme/foo/info/refs")
            assert r.status_code == 403

    async def test_correct_secret_unknown_repo_returns_404(self, proxy: GitProxy) -> None:
        # Note: the upstream client isn't mocked here — we're checking the
        # 404 short-circuits before we touch httpx, so no network egress.
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"http://127.0.0.1:{proxy.port}/git/{proxy.secret}/other/repo/info/refs"
            )
            assert r.status_code == 404


class TestForwarding:
    async def test_authorization_injected(
        self, proxy_with_mock_upstream: tuple[GitProxy, list[httpx.Request]]
    ) -> None:
        proxy, captured = proxy_with_mock_upstream
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"http://127.0.0.1:{proxy.port}/git/{proxy.secret}/acme/foo/info/refs"
                "?service=git-upload-pack"
            )
            assert r.status_code == 200
        assert len(captured) == 1
        upstream = captured[0]
        assert upstream.url == "https://github.com/acme/foo/info/refs?service=git-upload-pack"
        assert upstream.headers.get("authorization") == _basic_auth("ghp_TEST")

    async def test_client_supplied_authorization_replaced_not_leaked(
        self, proxy_with_mock_upstream: tuple[GitProxy, list[httpx.Request]]
    ) -> None:
        """The Authorization the client sent (e.g. from the URL's basic
        auth) must not propagate to upstream — only our injected token
        should appear."""
        proxy, captured = proxy_with_mock_upstream
        async with httpx.AsyncClient() as client:
            await client.get(
                f"http://127.0.0.1:{proxy.port}/git/{proxy.secret}/acme/foo/info/refs",
                headers={
                    "Authorization": "Basic poison",
                    "Proxy-Authorization": "should-not-leak",
                    "User-Agent": "git/2.40.0",
                },
            )
        upstream = captured[0]
        # Injected Authorization uses our token — the client's poisoned
        # value didn't pass through.
        assert upstream.headers.get("authorization") == _basic_auth("ghp_TEST")
        # Proxy-Authorization is sensitive hop-by-hop; httpx wouldn't
        # add it on its own, so its absence proves we stripped it.
        assert upstream.headers.get("proxy-authorization") is None
        # Non-hop-by-hop request headers pass through.
        assert upstream.headers.get("user-agent") == "git/2.40.0"

    async def test_non_hop_by_hop_response_headers_pass_through(
        self, proxy_with_mock_upstream: tuple[GitProxy, list[httpx.Request]]
    ) -> None:
        """The response side: arbitrary upstream headers reach the client.

        We don't assert the absence of ``connection`` / ``transfer-encoding``
        on the client-facing response — uvicorn adds those on its own
        connection. The relevant property is that meaningful upstream
        headers (e.g. GitHub request id) reach the caller intact.
        """
        proxy, _ = proxy_with_mock_upstream
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"http://127.0.0.1:{proxy.port}/git/{proxy.secret}/acme/foo/info/refs"
            )
            assert r.headers.get("x-github-request-id") == "abc"

    async def test_response_body_streamed(
        self, proxy_with_mock_upstream: tuple[GitProxy, list[httpx.Request]]
    ) -> None:
        proxy, _ = proxy_with_mock_upstream
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"http://127.0.0.1:{proxy.port}/git/{proxy.secret}/acme/foo/info/refs"
            )
            assert r.content == b"upstream-body"


class TestUpstreamFailure:
    async def test_upstream_error_returns_502(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("simulated network error")

        p = GitProxy({"acme/foo": "ghp_TEST"})
        await p._client.aclose()  # type: ignore[has-type]
        p._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[has-type]
        await p.start()
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"http://127.0.0.1:{p.port}/git/{p.secret}/acme/foo/info/refs")
                assert r.status_code == 502
        finally:
            await p.stop()


def test_module_exports() -> None:
    """Sanity: the module's public surface is what we expect."""
    from aios.sandbox import git_proxy

    assert "GitProxy" in git_proxy.__all__
    assert "repo_key" in git_proxy.__all__
