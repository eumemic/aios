"""Per-session HTTP proxy that holds GitHub PATs on the host side and
injects them into outbound git traffic from the sandbox.

The token never enters the container. The container's working tree has
``origin = http://host.docker.internal:<port>/git/<secret>/<owner>/<repo>``;
git speaks smart-HTTP to that URL; the proxy forwards each request to
``https://github.com/<owner>/<repo>/...`` with the ``Authorization``
header injected from the in-memory token map.

Per session lifecycle: started by the provisioner alongside the
container, torn down by the registry's ``release`` path. Token rotation
is an atomic swap of the in-memory map; the existing mount-snapshot
drift check (``updated_at`` is part of the snapshot key) also forces a
container recycle which restarts the proxy with a fresh map, so
correctness doesn't depend on ``update_repos``.

The per-session secret in the URL path is a sanity check, not a true
isolation primitive — the agent can read the secret from
``.git/config`` inside the container. Its job is to limit blast radius
if the proxy port is reachable from the wider network: a port scanner
without the secret gets a 403, and an attacker that learned the secret
can only proxy requests for the specific repos this session is already
attached to (and only while the proxy is alive).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import secrets
from collections.abc import AsyncIterator
from urllib.parse import urlsplit

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from aios.logging import get_logger

log = get_logger("aios.sandbox.git_proxy")

_HOP_BY_HOP_REQUEST_HEADERS: frozenset[str] = frozenset(
    {
        "host",
        "authorization",
        "connection",
        "content-length",
        "transfer-encoding",
        "expect",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "upgrade",
    }
)

_HOP_BY_HOP_RESPONSE_HEADERS: frozenset[str] = frozenset(
    {
        "connection",
        "content-encoding",  # httpx auto-decodes; we re-stream the decoded body
        "content-length",  # length changes once we drop other hop-by-hops
        "transfer-encoding",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "upgrade",
    }
)

# Generous bound on the upstream HTTP request — git smart HTTP can stream
# large pack files for big pushes/pulls.
_UPSTREAM_TIMEOUT_S = 300.0
_BIND_TIMEOUT_S = 5.0


def repo_key(repo_url: str) -> str:
    """Map a github repo URL to ``owner/repo`` (no ``.git`` suffix).

    Two URLs that differ only in the suffix resolve to the same token
    so the proxy doesn't duplicate entries for the same upstream.
    """
    parts = urlsplit(repo_url)
    path = parts.path.strip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]
    return path


class GitProxy:
    """One per session; holds tokens in memory, accepts git smart-HTTP
    requests on a random port, forwards to github.com with
    ``Authorization`` injected.

    Lifetime is bounded by the session container's lifetime: when the
    container is released or recycled, the proxy is stopped and the
    in-memory token map is dropped.
    """

    def __init__(self, repos: dict[str, str]) -> None:
        # Copy so callers can't mutate the live map under us.
        self._repos: dict[str, str] = dict(repos)
        self._secret = secrets.token_urlsafe(32)
        self._client = httpx.AsyncClient(timeout=_UPSTREAM_TIMEOUT_S, follow_redirects=False)
        self._server: uvicorn.Server | None = None
        self._serve_task: asyncio.Task[None] | None = None
        self._port: int | None = None

    @property
    def secret(self) -> str:
        return self._secret

    @property
    def port(self) -> int:
        assert self._port is not None, "GitProxy.start() has not completed"
        return self._port

    def proxy_url(self, repo_url: str, *, host: str) -> str:
        """The URL to write into a working tree's ``origin`` config so
        git operations from inside the container route through this
        proxy. Secret travels in the URL path; no userinfo, so git
        won't send a Basic-auth Authorization header that we'd have to
        strip on every request."""
        return f"http://{host}:{self.port}/git/{self._secret}/{repo_key(repo_url)}"

    def update_repos(self, repos: dict[str, str]) -> None:
        """Atomic replacement of the token map. Called on rotation /
        mount-set change so the proxy starts honoring the new tokens
        immediately. The container-recycle path also handles this case
        by restarting the proxy entirely; this method is a faster path
        when we want to skip the recycle."""
        self._repos = dict(repos)

    async def start(self) -> None:
        """Bind ``0.0.0.0:0`` (ephemeral port, all interfaces) and begin
        serving. Binding to 0.0.0.0 is required for the docker container
        to reach the proxy via ``host.docker.internal``; per-session
        secret + per-repo path keep the blast radius bounded if the port
        is exposed."""
        app = Starlette(
            routes=[
                Route(
                    "/git/{rest:path}",
                    self._handle,
                    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"],
                )
            ]
        )
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=0,
            log_level="error",
            access_log=False,
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._serve_task = asyncio.create_task(self._server.serve(), name="git-proxy-serve")

        # Wait for the bind to complete and grab the port.
        deadline = asyncio.get_event_loop().time() + _BIND_TIMEOUT_S
        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.01)
            if self._server.started and self._server.servers:
                sockets = self._server.servers[0].sockets
                if sockets:
                    self._port = sockets[0].getsockname()[1]
                    log.info("git_proxy.started", port=self._port)
                    return
        raise RuntimeError(f"git proxy failed to bind within {_BIND_TIMEOUT_S}s")

    async def stop(self) -> None:
        """Gracefully stop the server and close the httpx client."""
        if self._server is not None:
            self._server.should_exit = True
        if self._serve_task is not None:
            try:
                await asyncio.wait_for(self._serve_task, timeout=5.0)
            except TimeoutError:
                self._serve_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._serve_task
        await self._client.aclose()
        log.info("git_proxy.stopped", port=self._port)

    async def _handle(self, request: Request) -> Response:
        # URL path: /git/<secret>/<owner>/<repo>[/<rest>]
        rest = request.path_params["rest"]
        parts = rest.split("/", 3)
        if len(parts) < 3 or parts[0] != self._secret:
            return Response(status_code=403, content=b"forbidden")
        owner, repo = parts[1], parts[2]
        upstream_path = parts[3] if len(parts) > 3 else ""
        repo_no_git = repo.removesuffix(".git")
        token = self._repos.get(f"{owner}/{repo_no_git}")
        if token is None:
            return Response(status_code=404, content=b"unknown repo for this proxy")

        upstream_url = f"https://github.com/{owner}/{repo}"
        if upstream_path:
            upstream_url = f"{upstream_url}/{upstream_path}"
        if request.url.query:
            upstream_url = f"{upstream_url}?{request.url.query}"

        fwd_headers: dict[str, str] = {}
        for name, value in request.headers.items():
            if name.lower() in _HOP_BY_HOP_REQUEST_HEADERS:
                continue
            fwd_headers[name] = value
        # Use Basic auth with the standard "x-access-token:<pat>" form —
        # this mirrors what `https://x-access-token:$TOKEN@github.com/...`
        # produces and is accepted by both classic PATs and fine-grained
        # tokens. The plain "Authorization: token <pat>" header works for
        # classic PATs but is rejected by newer fine-grained tokens.
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        fwd_headers["Authorization"] = f"Basic {basic}"

        try:
            upstream_req = self._client.build_request(
                request.method,
                upstream_url,
                headers=fwd_headers,
                content=request.stream(),
            )
            upstream_resp = await self._client.send(upstream_req, stream=True)
        except httpx.RequestError as exc:
            log.warning(
                "git_proxy.upstream_error",
                upstream_url=upstream_url,
                method=request.method,
                error=str(exc),
            )
            return Response(status_code=502, content=b"upstream error")

        resp_headers: dict[str, str] = {}
        for name, value in upstream_resp.headers.items():
            if name.lower() in _HOP_BY_HOP_RESPONSE_HEADERS:
                continue
            resp_headers[name] = value

        async def _body() -> AsyncIterator[bytes]:
            # ``aiter_bytes`` works whether the underlying transport
            # streamed the body (real network) or pre-buffered it
            # (httpx.MockTransport in tests). Content-Encoding is
            # already stripped from ``resp_headers`` because httpx
            # decodes here, so the client sees raw bytes consistently.
            try:
                async for chunk in upstream_resp.aiter_bytes():
                    yield chunk
            finally:
                await upstream_resp.aclose()

        return StreamingResponse(
            _body(),
            status_code=upstream_resp.status_code,
            headers=resp_headers,
        )


__all__ = ["GitProxy", "repo_key"]
