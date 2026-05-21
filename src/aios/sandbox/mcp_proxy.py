"""Worker-side HTTP broker that exposes permitted MCP tools to
sandboxed agents.

Architecture mirror of :class:`aios.sandbox.git_proxy.GitProxy` but
scoped worker-wide instead of per-session: one shared
Starlette/uvicorn instance, demultiplexing by per-session secret
embedded in the URL path. Each session's secret is minted by the
sandbox provisioner and registered with :meth:`register_session`
before the container starts; the secret is injected into the sandbox
as ``MCP_BROKER_SECRET`` so the in-sandbox ``mcp`` CLI can present it
on every request.

Credentials never enter the sandbox: the broker holds vault-decrypted
auth headers in memory (via the existing :mod:`aios.mcp.client` pool)
and the sandbox only ever sees the broker's HTTP surface.

Authorization model (v1): only tools where the session's resolved
``mcp_toolset`` permission is ``always_allow`` are callable through
the CLI. Everything else returns 403 with an explanatory message;
the model-tool path remains available for those tools.

The per-session secret in the URL path is a sanity check, not a true
isolation primitive — the agent can read the secret from
``$MCP_BROKER_SECRET`` inside the container. Its job is to limit
blast radius if the broker port is reachable from the wider network
and to ensure the broker can identify which session is calling.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from aios.logging import get_logger
from aios.mcp.client import call_mcp_tool, discover_mcp_tools, resolve_auth_for_target_url
from aios.models.agents import McpServerSpec, ToolSpec

log = get_logger("aios.sandbox.mcp_proxy")

_BIND_TIMEOUT_S = 5.0


def _resolve_tool_permission(toolset: ToolSpec, tool_name: str) -> tuple[bool, str | None]:
    """Resolve ``(enabled, permission_policy_type)`` for a specific tool
    under an ``mcp_toolset`` entry.

    Precedence: explicit per-tool override in ``configs`` first; otherwise
    ``default_config``; otherwise ``(enabled=True, policy=None)``.
    """
    if toolset.configs:
        for cfg in toolset.configs:
            if cfg.name == tool_name:
                policy = cfg.permission_policy.type if cfg.permission_policy else None
                return cfg.enabled, policy
    default = toolset.default_config
    if default is not None:
        policy = default.permission_policy.type if default.permission_policy else None
        return default.enabled, policy
    return True, None


def _find_toolset(agent_tools: list[ToolSpec], server_name: str) -> ToolSpec | None:
    for spec in agent_tools:
        if spec.type == "mcp_toolset" and spec.enabled and spec.mcp_server_name == server_name:
            return spec
    return None


def _find_server(agent_servers: list[McpServerSpec], server_name: str) -> McpServerSpec | None:
    for s in agent_servers:
        if s.name == server_name:
            return s
    return None


def _err(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def _transport_err(server_name: str, exc: BaseException) -> JSONResponse:
    """502 response shape for an upstream MCP discovery failure.

    Mirrors :func:`aios.mcp.client.call_mcp_tool`'s error envelope
    (``{"error": ..., "code": "transport_error"}``) so the sandboxed
    model sees the same shape whether discovery or invocation hit the
    upstream blip — and so its CLI wrapper can branch on ``code``
    rather than substring-matching the human-readable message.
    """
    return JSONResponse(
        {
            "error": f"MCP server '{server_name}' error: {type(exc).__name__}: {exc}",
            "code": "transport_error",
        },
        status_code=502,
    )


class McpBroker:
    """One shared HTTP broker per worker.

    Lifecycle: instantiated and started during ``worker_main``,
    registered on ``aios.harness.runtime.mcp_broker``, stopped during
    shutdown. Sandbox provisioning calls :meth:`register_session` to
    bind a freshly minted per-session secret to a session id; release
    calls :meth:`unregister_session`.

    Transports: always TCP (ephemeral port on ``0.0.0.0``). When
    ``socket_path`` is provided, the same Starlette app is *also*
    served over a Unix-domain socket at that host path — this is the
    only mode that works in deployments where the worker's TCP port
    isn't published to the sandbox network (e.g. Coolify production).
    """

    def __init__(self, socket_path: Path | None = None) -> None:
        self._secrets: dict[str, str] = {}
        self._server: uvicorn.Server | None = None
        self._serve_task: asyncio.Task[None] | None = None
        self._port: int | None = None
        self._socket_path: Path | None = socket_path
        self._uds_server: uvicorn.Server | None = None
        self._uds_serve_task: asyncio.Task[None] | None = None

    @property
    def port(self) -> int:
        assert self._port is not None, "McpBroker.start() has not completed"
        return self._port

    @property
    def socket_path(self) -> Path | None:
        return self._socket_path

    def register_session(self, session_id: str, secret: str) -> None:
        self._secrets[secret] = session_id

    def unregister_session(self, session_id: str) -> None:
        stale = [s for s, sid in self._secrets.items() if sid == session_id]
        for s in stale:
            self._secrets.pop(s, None)

    def _build_app(self) -> Starlette:
        return Starlette(
            routes=[
                Route("/v1/{secret}/servers", self._list_servers, methods=["GET"]),
                Route(
                    "/v1/{secret}/servers/{server}/tools",
                    self._list_tools,
                    methods=["GET"],
                ),
                Route(
                    "/v1/{secret}/servers/{server}/tools/{tool}",
                    self._tool_help,
                    methods=["GET"],
                ),
                Route(
                    "/v1/{secret}/servers/{server}/tools/{tool}",
                    self._invoke,
                    methods=["POST"],
                ),
            ]
        )

    async def start(self) -> None:
        """Bind ``0.0.0.0:0`` and begin serving. ``0.0.0.0`` covers every
        interface the worker has — the sandbox may reach in via any of
        them depending on deployment topology.

        If ``socket_path`` was set on construction, also bind a Unix
        domain socket at that path with mode ``0o666`` so the sandbox's
        unprivileged user can connect.
        """
        app = self._build_app()
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=0,
            log_level="error",
            access_log=False,
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._serve_task = asyncio.create_task(self._server.serve(), name="mcp-broker-serve")

        # Set up UDS server (if requested) on the *same* Starlette app
        # so route handlers and the secret map are shared.
        if self._socket_path is not None:
            # Parent dir may not exist (e.g. ``/var/run/aios`` on a host
            # that hasn't been pre-provisioned); a stale file from a
            # previous crashed worker must be cleared before bind.
            self._socket_path.parent.mkdir(parents=True, exist_ok=True)
            if self._socket_path.exists() or self._socket_path.is_symlink():
                self._socket_path.unlink()
            uds_config = uvicorn.Config(
                app,
                uds=str(self._socket_path),
                log_level="error",
                access_log=False,
                lifespan="off",
            )
            self._uds_server = uvicorn.Server(uds_config)
            self._uds_serve_task = asyncio.create_task(
                self._uds_server.serve(), name="mcp-broker-serve-uds"
            )

        try:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + _BIND_TIMEOUT_S
            while loop.time() < deadline:
                await asyncio.sleep(0.01)
                tcp_ready = (
                    self._server.started
                    and bool(self._server.servers)
                    and bool(self._server.servers[0].sockets)
                )
                uds_ready = self._uds_server is None or (
                    self._uds_server.started and bool(self._uds_server.servers)
                )
                if tcp_ready and uds_ready:
                    assert self._server.servers[0].sockets  # for mypy
                    self._port = self._server.servers[0].sockets[0].getsockname()[1]
                    if self._socket_path is not None:
                        # Default uvicorn UDS mode (0o600) excludes the
                        # sandbox's unprivileged user; relax to 0o666.
                        os.chmod(self._socket_path, 0o666)
                    log.info(
                        "mcp_broker.started",
                        port=self._port,
                        socket_path=str(self._socket_path) if self._socket_path else None,
                    )
                    return
            raise RuntimeError(f"mcp broker failed to bind within {_BIND_TIMEOUT_S}s")
        except BaseException:
            # Bind never completed; the caller drops its reference, so
            # nothing else will call stop() — without this the serve
            # task and the half-bound socket leak for the worker's
            # lifetime.
            await self.stop()
            raise

    async def stop(self) -> None:
        try:
            # Graceful drain only matters once the server actually
            # bound — ``should_exit`` is checked inside the uvicorn
            # main loop, which never runs on a failed-bind path, so
            # without this gate the failed-bind cleanup pays a
            # pointless 5s wait on every retry.
            if self._server is not None and self._server.started:
                self._server.should_exit = True
                if self._serve_task is not None:
                    with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                        await asyncio.wait_for(self._serve_task, timeout=5.0)
            if self._uds_server is not None and self._uds_server.started:
                self._uds_server.should_exit = True
                if self._uds_serve_task is not None:
                    with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                        await asyncio.wait_for(self._uds_serve_task, timeout=5.0)
        finally:
            if self._serve_task is not None and not self._serve_task.done():
                self._serve_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._serve_task
            if self._uds_serve_task is not None and not self._uds_serve_task.done():
                self._uds_serve_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._uds_serve_task
            # Uvicorn doesn't always clean the socket file (especially
            # on cancelled-during-startup paths); a stale file across
            # worker restarts would break the next bind, so do it
            # here unconditionally.
            if self._socket_path is not None:
                with contextlib.suppress(FileNotFoundError):
                    self._socket_path.unlink()
        log.info("mcp_broker.stopped", port=self._port)

    async def _resolve_session(self, request: Request) -> str | Response:
        secret = request.path_params["secret"]
        session_id = self._secrets.get(secret)
        if session_id is None:
            return _err(403, "forbidden: unknown or expired secret")
        return session_id

    async def _load_agent(self, session_id: str) -> Any:
        """Fetch the agent (or agent_version) attached to ``session_id``.

        Mirrors the loading dance in
        :func:`aios.harness.loop.run_session_step` — session→agent
        with optional version pinning.
        """
        from aios.harness import runtime
        from aios.services import agents as agents_service
        from aios.services import sessions as sessions_service

        pool = runtime.require_pool()
        account_id = await sessions_service.load_session_account_id(pool, session_id)
        session = await sessions_service.get_session_basic(pool, session_id, account_id=account_id)
        return await agents_service.load_for_session(pool, session, account_id=account_id)

    async def _load_auth_for(
        self, session_id: str, server_url: str
    ) -> tuple[str | None, dict[str, str]]:
        """Resolve ``(vault_id, headers)`` for an MCP call out to ``server_url``.

        Used by all three tool routes (``_list_tools`` / ``_tool_help`` /
        ``_invoke``) to look up the session's vault binding for the server
        and decrypt the auth headers. Deferred-imports
        :mod:`aios.harness.runtime` and :mod:`aios.services.sessions` to
        avoid an import cycle at module load.
        """
        from aios.harness import runtime
        from aios.services import sessions as sessions_service

        pool = runtime.require_pool()
        crypto_box = runtime.require_crypto_box()
        account_id = await sessions_service.load_session_account_id(pool, session_id)
        return await resolve_auth_for_target_url(
            pool, crypto_box, session_id, server_url, account_id=account_id
        )

    async def _list_servers(self, request: Request) -> Response:
        resolved = await self._resolve_session(request)
        if isinstance(resolved, Response):
            return resolved
        agent = await self._load_agent(resolved)
        enabled_names: set[str] = {
            spec.mcp_server_name
            for spec in agent.tools
            if spec.type == "mcp_toolset" and spec.enabled and spec.mcp_server_name is not None
        }
        servers = [
            {"name": s.name, "host": urlsplit(s.url).netloc}
            for s in agent.mcp_servers
            if s.name in enabled_names
        ]
        return JSONResponse({"servers": servers})

    async def _resolve_for_tool(
        self, request: Request
    ) -> tuple[str, McpServerSpec, ToolSpec, str] | Response:
        """Common prelude for tool routes: resolve session, agent,
        toolset, server, and authorize the tool. Returns
        ``(session_id, server_spec, toolset, tool_name)`` on success or
        a ``Response`` to return immediately on failure.
        """
        resolved = await self._resolve_session(request)
        if isinstance(resolved, Response):
            return resolved
        session_id = resolved
        server_name = request.path_params["server"]
        tool_name = request.path_params["tool"]
        agent = await self._load_agent(session_id)
        toolset = _find_toolset(agent.tools, server_name)
        if toolset is None:
            return _err(404, f"server '{server_name}' is not available to this session")
        enabled, policy = _resolve_tool_permission(toolset, tool_name)
        if not enabled:
            return _err(403, f"tool '{tool_name}' is disabled on this agent")
        if policy != "always_allow":
            return _err(
                403,
                f"tool '{tool_name}' is not callable via CLI "
                f"(permission policy: {policy or 'unset'}). Use the model-tool path.",
            )
        server = _find_server(agent.mcp_servers, server_name)
        if server is None:
            return _err(404, f"server '{server_name}' has no URL configured")
        return session_id, server, toolset, tool_name

    async def _list_tools(self, request: Request) -> Response:
        resolved = await self._resolve_session(request)
        if isinstance(resolved, Response):
            return resolved
        session_id = resolved
        server_name = request.path_params["server"]
        agent = await self._load_agent(session_id)
        toolset = _find_toolset(agent.tools, server_name)
        if toolset is None:
            return _err(404, f"server '{server_name}' is not available to this session")
        server = _find_server(agent.mcp_servers, server_name)
        if server is None:
            return _err(404, f"server '{server_name}' has no URL configured")

        vault_id, headers = await self._load_auth_for(session_id, server.url)
        try:
            tool_dicts, _instructions = await discover_mcp_tools(
                server.url, vault_id, headers, server_name
            )
        except Exception as exc:
            log.warning(
                "mcp_broker.discovery_failed",
                server_name=server_name,
                url=server.url,
                exc_info=True,
            )
            return _transport_err(server_name, exc)

        out: list[dict[str, Any]] = []
        for td in tool_dicts:
            fn = td.get("function", {})
            qualified = fn.get("name", "")
            parts = qualified.split("__", 2)
            if len(parts) != 3 or parts[0] != "mcp" or parts[1] != server_name:
                continue
            tool_name = parts[2]
            enabled, policy = _resolve_tool_permission(toolset, tool_name)
            if not enabled or policy != "always_allow":
                continue
            out.append({"name": tool_name, "description": fn.get("description", "")})
        return JSONResponse({"tools": out})

    async def _tool_help(self, request: Request) -> Response:
        resolved = await self._resolve_for_tool(request)
        if isinstance(resolved, Response):
            return resolved
        session_id, server, _toolset, tool_name = resolved

        vault_id, headers = await self._load_auth_for(session_id, server.url)
        try:
            tool_dicts, _ = await discover_mcp_tools(server.url, vault_id, headers, server.name)
        except Exception as exc:
            log.warning(
                "mcp_broker.discovery_failed",
                server_name=server.name,
                url=server.url,
                exc_info=True,
            )
            return _transport_err(server.name, exc)
        qualified = f"mcp__{server.name}__{tool_name}"
        for td in tool_dicts:
            fn = td.get("function", {})
            if fn.get("name") == qualified:
                return JSONResponse(
                    {
                        "name": tool_name,
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {}),
                    }
                )
        return _err(404, f"tool '{tool_name}' not found on server '{server.name}'")

    async def _invoke(self, request: Request) -> Response:
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return _err(400, "request body must be JSON")
        arguments = body.get("arguments") if isinstance(body, dict) else None
        if not isinstance(arguments, dict):
            return _err(400, 'request body must contain {"arguments": <object>}')

        resolved = await self._resolve_for_tool(request)
        if isinstance(resolved, Response):
            return resolved
        session_id, server, _toolset, tool_name = resolved

        vault_id, headers = await self._load_auth_for(session_id, server.url)
        log.info(
            "mcp_broker.invoke",
            session_id=session_id,
            server=server.name,
            tool=tool_name,
        )
        envelope = await call_mcp_tool(server.url, vault_id, headers, tool_name, arguments)
        return JSONResponse(envelope)


__all__ = ["McpBroker"]
