"""Worker-side HTTP broker for the sandbox-side ``tool`` CLI.

Generalises the prior MCP-only broker into a single tool surface
covering every reachable tool: built-ins classified ``cli`` or ``both``
(per the registry default and the agent's ``ToolSpec.transport``
override) plus enabled MCP tools (per the operator's ``mcp_toolset``
config). The model's tool surface and the CLI tool surface share one
resolution chain:

* Effective transport via :func:`aios.tools.registry.effective_transport`.
* Effective permission via :func:`aios.models.agents.resolve_permission`
  (built-ins) and :func:`aios.services.agents.effective_mcp_permission`
  (MCP).
* Effective MCP enabled via :func:`aios.models.agents.resolve_mcp_enabled`.

The previous broker carried its own ``_resolve_tool_permission`` copy
that read per-tool ``configs[]``; that drift is gone — the canonical
resolvers are the single source of truth.

**v1 CLI execution gate.** A tool is reachable iff effective transport
∈ ``{"cli", "both"}`` *and* effective permission is ``always_allow``
*and* the ``ToolSpec`` is not ``type="custom"``. ``always_ask`` and
``custom`` tools are hidden from listing and refused on direct call:
doing them correctly means blocking the bash request until the operator
confirms / the client runtime POSTs a custom result — a synchronous
bridge over the async confirmation machinery. That bridge is deferred;
the seam where it slots in is the ``return _err(403, ...)`` branches
in :meth:`ToolBroker._resolve_builtin` and :meth:`ToolBroker._resolve_mcp`.

**Invocation.** Built-ins run through :func:`aios.tools.invoke.invoke_builtin`
— the same pure core the model path drives (the event-append and sweep
are model-path-only and stay in ``harness/tool_dispatch.py``). MCP
calls hit :func:`aios.mcp.client.call_mcp_tool` unchanged. Credentials
never enter the sandbox: the broker holds vault-decrypted auth headers
in memory and the sandbox only sees the broker's HTTP surface.

**Architecture.** One shared Starlette/uvicorn instance per worker,
demultiplexed by per-session secret in the URL path. Each session's
secret is minted by the sandbox provisioner and registered with
:meth:`register_session` before the container starts; the secret is
injected as ``TOOL_BROKER_SECRET`` so the in-sandbox ``tool`` CLI
presents it on every request. The secret is a sanity check, not an
isolation primitive — the agent can read it from the environment —
but it limits blast radius if the broker port is reachable from the
wider network.
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

from aios.errors import AiosError
from aios.logging import get_logger
from aios.mcp.client import call_mcp_tool, discover_mcp_tools
from aios.models.agents import (
    McpServerSpec,
    ToolSpec,
    resolve_mcp_enabled,
)
from aios.tools.invoke import ToolBail, invoke_builtin
from aios.tools.registry import ToolResult, effective_transport, registry

log = get_logger("aios.sandbox.tool_broker")

_BIND_TIMEOUT_S = 5.0


# ── shared helpers ────────────────────────────────────────────────────────────


def _err(status: int, message: str, code: str | None = None) -> JSONResponse:
    body: dict[str, Any] = {"error": message}
    if code is not None:
        body["code"] = code
    return JSONResponse(body, status_code=status)


def _transport_err(server_name: str, exc: BaseException) -> JSONResponse:
    """502 envelope for upstream MCP transport failures.

    Mirrors :func:`aios.mcp.client.call_mcp_tool`'s error shape so the
    CLI can branch on ``code`` rather than substring-matching the
    human-readable message.
    """
    return JSONResponse(
        {
            "error": f"MCP server '{server_name}' error: {type(exc).__name__}: {exc}",
            "code": "transport_error",
        },
        status_code=502,
    )


def _find_server(servers: list[McpServerSpec], server_name: str) -> McpServerSpec | None:
    for s in servers:
        if s.name == server_name:
            return s
    return None


def _find_mcp_toolset(agent_tools: list[ToolSpec], server_name: str) -> ToolSpec | None:
    for spec in agent_tools:
        if spec.type == "mcp_toolset" and spec.enabled and spec.mcp_server_name == server_name:
            return spec
    return None


def _find_builtin_spec(agent_tools: list[ToolSpec], name: str) -> ToolSpec | None:
    """Find the agent's ToolSpec for a built-in by name.

    Skips ``mcp_toolset`` (a different namespace) and ``custom`` (hidden
    from the CLI in v1 — needs the client-runtime-blocking bridge).
    """
    for spec in agent_tools:
        if spec.type in ("mcp_toolset", "custom"):
            continue
        if spec.type == name:
            return spec
    return None


def _builtin_cli_eligible(spec: ToolSpec) -> bool:
    """v1 CLI gate for a built-in ``ToolSpec``.

    Reachable iff enabled, effective transport ∈ ``{cli, both}``, and
    effective permission isn't ``always_ask`` (the confirmation bridge
    is deferred). Caller filters out ``type=custom`` / ``mcp_toolset``
    via :func:`_find_builtin_spec`.
    """
    if not spec.enabled:
        return False
    transport = spec.transport or registry.get(spec.type).transport
    if transport not in ("cli", "both"):
        return False
    return spec.permission != "always_ask"


def _envelope_from_result(result: ToolResult | dict[str, Any]) -> dict[str, Any]:
    """Shape a built-in handler's result into the CLI envelope.

    Matches :func:`aios.mcp.client.call_mcp_tool`'s shape — ``{"content": str}``
    on success, ``{"error": str, "code": str}`` on tool-level error —
    so the CLI consumer can dispatch on the response shape regardless
    of transport. Non-string content (dicts, multimodal lists) is
    JSON-encoded; ``--full`` on the CLI side prints the raw envelope.
    """
    if isinstance(result, ToolResult):
        if result.is_error:
            content = (
                result.content
                if isinstance(result.content, str)
                else json.dumps(result.content, ensure_ascii=False)
            )
            return {"error": content, "code": "tool_error"}
        if isinstance(result.content, str):
            return {"content": result.content}
        return {"content": json.dumps(result.content, ensure_ascii=False)}
    return {"content": json.dumps(result, ensure_ascii=False)}


# ── broker ────────────────────────────────────────────────────────────────────


class ToolBroker:
    """One shared HTTP broker per worker.

    Lifecycle: instantiated and started during ``worker_main``,
    registered on ``aios.harness.runtime.tool_broker``, stopped during
    shutdown. Sandbox provisioning calls :meth:`register_session` to
    bind a freshly minted per-session secret to a session id; release
    calls :meth:`unregister_session`.

    Transports: always TCP (ephemeral port on ``0.0.0.0``). When
    ``socket_path`` is provided, the same Starlette app is *also*
    served over a Unix-domain socket at that host path — the only mode
    that works in deployments where the worker's TCP port isn't
    published to the sandbox network (e.g. Coolify production).
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
        assert self._port is not None, "ToolBroker.start() has not completed"
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
                # Discovery: everything the CLI can see, in one round trip.
                Route("/v1/{secret}/tools", self._list_surface, methods=["GET"]),
                # Built-in: schema + invoke.
                Route("/v1/{secret}/builtins/{name}", self._builtin_schema, methods=["GET"]),
                Route("/v1/{secret}/builtins/{name}", self._builtin_invoke, methods=["POST"]),
                # MCP: list methods, method schema, method invoke.
                Route("/v1/{secret}/mcp/{server}", self._mcp_list_methods, methods=["GET"]),
                Route(
                    "/v1/{secret}/mcp/{server}/{tool}",
                    self._mcp_method_schema,
                    methods=["GET"],
                ),
                Route(
                    "/v1/{secret}/mcp/{server}/{tool}",
                    self._mcp_invoke,
                    methods=["POST"],
                ),
            ]
        )

    async def start(self) -> None:
        """Bind ``0.0.0.0:0`` and begin serving.

        ``0.0.0.0`` covers every interface the worker has — the sandbox
        may reach in via any of them depending on deployment topology.
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
        self._serve_task = asyncio.create_task(self._server.serve(), name="tool-broker-serve")

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
                self._uds_server.serve(), name="tool-broker-serve-uds"
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
                        "tool_broker.started",
                        port=self._port,
                        socket_path=str(self._socket_path) if self._socket_path else None,
                    )
                    return
            raise RuntimeError(f"tool broker failed to bind within {_BIND_TIMEOUT_S}s")
        except BaseException:
            # Bind never completed; the caller drops its reference, so
            # nothing else will call stop() — without this the serve
            # task and the half-bound socket leak for the worker's
            # lifetime.
            await self.stop()
            raise

    def serve_tasks(self) -> tuple[asyncio.Task[object], ...]:
        """Return uvicorn serve task handles created by :meth:`start`."""
        tasks: list[asyncio.Task[object]] = []
        if self._serve_task is not None:
            tasks.append(self._serve_task)
        if self._uds_serve_task is not None:
            tasks.append(self._uds_serve_task)
        return tuple(tasks)

    async def stop(self) -> None:
        try:
            # Graceful drain only matters once the server actually
            # bound — ``should_exit`` is checked inside the uvicorn
            # main loop, which never runs on a failed-bind path, so
            # without the ``started`` gate the failed-bind cleanup pays
            # a pointless 5s wait on every retry.
            #
            # Flip ``should_exit`` on *both* servers before awaiting
            # either drain: uvicorn pays a ~150-200 ms fixed shutdown
            # floor once the flag is set, so serializing the two waits
            # doubles it. Setting both flags first lets the two floors
            # overlap; ``gather`` then waits out the slower of the two.
            drains: list[asyncio.Task[None]] = []
            if self._server is not None and self._server.started:
                self._server.should_exit = True
                if self._serve_task is not None:
                    drains.append(self._serve_task)
            if self._uds_server is not None and self._uds_server.started:
                self._uds_server.should_exit = True
                if self._uds_serve_task is not None:
                    drains.append(self._uds_serve_task)
            if drains:
                with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(
                        asyncio.gather(*drains, return_exceptions=True), timeout=5.0
                    )
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
        log.info("tool_broker.stopped", port=self._port)

    # ── session / agent loading ───────────────────────────────────────────

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
        with optional version pinning. Deferred imports
        avoid an import cycle at module load.
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
        """Resolve ``(vault_id, headers)`` for an MCP call out to ``server_url``."""
        from aios.harness import runtime
        from aios.mcp.client import resolve_auth_for_target_url
        from aios.services import sessions as sessions_service

        pool = runtime.require_pool()
        crypto_box = runtime.require_crypto_box()
        account_id = await sessions_service.load_session_account_id(pool, session_id)
        return await resolve_auth_for_target_url(
            pool, crypto_box, session_id, server_url, account_id=account_id
        )

    # ── discovery ─────────────────────────────────────────────────────────

    async def _list_surface(self, request: Request) -> Response:
        """Return the CLI-reachable surface — built-ins + MCP servers.

        Built-ins are filtered by the v1 CLI gate (transport + permission
        + non-custom). MCP servers list every enabled ``mcp_toolset``
        whose ``McpServerSpec`` is configured; per-method filtering
        happens on dereference.
        """
        resolved = await self._resolve_session(request)
        if isinstance(resolved, Response):
            return resolved
        agent = await self._load_agent(resolved)

        builtins: list[dict[str, Any]] = []
        for spec in agent.tools:
            if spec.type in ("mcp_toolset", "custom"):
                continue
            if not _builtin_cli_eligible(spec):
                continue
            tool_def = registry.get(spec.type)
            builtins.append({"name": tool_def.name, "description": tool_def.description})

        enabled_server_names: set[str] = {
            spec.mcp_server_name
            for spec in agent.tools
            if spec.type == "mcp_toolset" and spec.enabled and spec.mcp_server_name is not None
        }
        servers = [
            {"name": s.name, "host": urlsplit(s.url).netloc}
            for s in agent.mcp_servers
            if s.name in enabled_server_names
        ]

        return JSONResponse({"builtins": builtins, "servers": servers})

    # ── built-in routes ───────────────────────────────────────────────────

    async def _resolve_builtin(self, request: Request) -> tuple[str, str, ToolSpec] | Response:
        """Resolve (session_id, name, spec) for a built-in route.

        Returns an error ``Response`` with a diagnostic message if the
        tool isn't on the agent, isn't CLI-reachable for transport
        reasons, or hits the deferred ``always_ask``/``custom``
        confirmation bridge.
        """
        resolved = await self._resolve_session(request)
        if isinstance(resolved, Response):
            return resolved
        session_id = resolved
        name = request.path_params["name"]
        agent = await self._load_agent(session_id)
        spec = _find_builtin_spec(agent.tools, name)
        if spec is None or not spec.enabled:
            return _err(404, f"tool {name!r} is not declared on this agent")
        if not _builtin_cli_eligible(spec):
            transport = spec.transport or registry.get(spec.type).transport
            if transport not in ("cli", "both"):
                return _err(
                    403,
                    f"tool {name!r} is not CLI-reachable on this agent "
                    f"(transport={transport!r}). The model can still call it.",
                )
            return _err(
                403,
                f"tool {name!r} requires confirmation (always_ask) and is "
                f"not yet invocable via the CLI — the synchronous "
                f"confirmation bridge is not built. Use the model-tool path.",
            )
        return session_id, name, spec

    async def _builtin_schema(self, request: Request) -> Response:
        resolved = await self._resolve_builtin(request)
        if isinstance(resolved, Response):
            return resolved
        _session_id, name, _spec = resolved
        tool_def = registry.get(name)
        return JSONResponse(
            {
                "name": tool_def.name,
                "description": tool_def.description,
                "input_schema": tool_def.parameters_schema,
            }
        )

    async def _builtin_invoke(self, request: Request) -> Response:
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return _err(400, "request body must be JSON")
        arguments = body.get("arguments") if isinstance(body, dict) else None
        if not isinstance(arguments, dict):
            return _err(400, 'request body must contain {"arguments": <object>}')

        resolved = await self._resolve_builtin(request)
        if isinstance(resolved, Response):
            return resolved
        session_id, name, _spec = resolved

        log.info("tool_broker.invoke_builtin", session_id=session_id, name=name)
        try:
            result = await invoke_builtin(session_id, name, arguments)
        except ToolBail as bail:
            return _err(400, str(bail), code="tool_bail")
        except AiosError as exc:
            # A typed aios error (a permission denial, not-found, conflict, rate-limit, or
            # a tool *ArgumentError) carries its real status + ``error_type`` code, not an
            # opaque 500 — matching the model dispatch path, which renders a client-class
            # error as a clean refusal (harness/tool_dispatch._classify_tool_error). The
            # broker doesn't (can't) evict; here it's purely the response envelope.
            return _err(exc.status_code, exc.to_message(), code=exc.error_type)
        except Exception as exc:
            # A truly untyped handler exception is an internal failure → opaque 500. (On
            # the model path its lifecycle would also have sandbox-evicted; the broker,
            # running inside the sandbox, cannot.)
            log.exception("tool_broker.invoke_failed", session_id=session_id, name=name)
            return _err(500, f"{type(exc).__name__}: {exc}", code="internal_error")

        return JSONResponse(_envelope_from_result(result))

    # ── MCP routes ────────────────────────────────────────────────────────

    async def _resolve_mcp(
        self, request: Request, *, require_tool: bool
    ) -> tuple[str, McpServerSpec, ToolSpec, str | None] | Response:
        """Resolve session, server, toolset, and (optionally) tool for an
        MCP route.

        With ``require_tool=False`` (the list-methods route), tool_name
        is ``None`` and only the server is resolved. With
        ``require_tool=True``, the per-tool CLI gate (enabled +
        transport + permission) is applied.
        """
        resolved = await self._resolve_session(request)
        if isinstance(resolved, Response):
            return resolved
        session_id = resolved
        server_name = request.path_params["server"]

        agent = await self._load_agent(session_id)
        toolset = _find_mcp_toolset(agent.tools, server_name)
        if toolset is None:
            return _err(404, f"MCP server {server_name!r} is not available to this session")
        server = _find_server(agent.mcp_servers, server_name)
        if server is None:
            return _err(404, f"MCP server {server_name!r} has no URL configured")

        if not require_tool:
            return session_id, server, toolset, None

        tool_name = request.path_params["tool"]
        qualified = f"mcp__{server_name}__{tool_name}"
        if not resolve_mcp_enabled(qualified, agent.tools):
            return _err(403, f"MCP tool {tool_name!r} is disabled on server {server_name!r}")
        transport = effective_transport(qualified, agent.tools)
        if transport not in ("cli", "both"):
            return _err(
                403,
                f"MCP tool {tool_name!r} is not CLI-reachable on server "
                f"{server_name!r} (transport={transport!r}). The model can "
                f"still call it.",
            )
        from aios.services import agents as agents_service

        perm = agents_service.effective_mcp_permission(qualified, agent.tools)
        if perm != "always_allow":
            return _err(
                403,
                f"MCP tool {tool_name!r} requires confirmation (permission "
                f"{perm!r}) and is not yet invocable via the CLI — the "
                f"synchronous confirmation bridge is not built. Use the "
                f"model-tool path.",
            )
        return session_id, server, toolset, tool_name

    async def _mcp_list_methods(self, request: Request) -> Response:
        resolved = await self._resolve_mcp(request, require_tool=False)
        if isinstance(resolved, Response):
            return resolved
        session_id, server, toolset, _ = resolved

        vault_id, headers = await self._load_auth_for(session_id, server.url)
        try:
            tool_dicts, _instructions = await discover_mcp_tools(
                server.url, vault_id, headers, server.name, spec_headers=server.headers
            )
        except Exception as exc:
            log.warning(
                "tool_broker.mcp_discovery_failed",
                server_name=server.name,
                url=server.url,
                exc_info=True,
            )
            return _transport_err(server.name, exc)

        from aios.services import agents as agents_service

        out: list[dict[str, Any]] = []
        # Pass a single-element list containing this toolset to the
        # resolvers — they match by ``mcp_server_name`` regardless of
        # the surrounding agent tools list.
        scope: list[ToolSpec] = [toolset]
        for td in tool_dicts:
            fn = td.get("function", {})
            qualified = fn.get("name", "")
            parts = qualified.split("__", 2)
            if len(parts) != 3 or parts[0] != "mcp" or parts[1] != server.name:
                continue
            tool_name = parts[2]
            if not resolve_mcp_enabled(qualified, scope):
                continue
            if effective_transport(qualified, scope) not in ("cli", "both"):
                continue
            if agents_service.effective_mcp_permission(qualified, scope) != "always_allow":
                continue
            out.append({"name": tool_name, "description": fn.get("description", "")})
        return JSONResponse({"tools": out})

    async def _mcp_method_schema(self, request: Request) -> Response:
        resolved = await self._resolve_mcp(request, require_tool=True)
        if isinstance(resolved, Response):
            return resolved
        session_id, server, _toolset, tool_name = resolved
        assert tool_name is not None  # require_tool=True

        vault_id, headers = await self._load_auth_for(session_id, server.url)
        try:
            tool_dicts, _ = await discover_mcp_tools(
                server.url, vault_id, headers, server.name, spec_headers=server.headers
            )
        except Exception as exc:
            log.warning(
                "tool_broker.mcp_discovery_failed",
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
        return _err(404, f"tool {tool_name!r} not found on MCP server {server.name!r}")

    async def _mcp_invoke(self, request: Request) -> Response:
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return _err(400, "request body must be JSON")
        arguments = body.get("arguments") if isinstance(body, dict) else None
        if not isinstance(arguments, dict):
            return _err(400, 'request body must contain {"arguments": <object>}')

        resolved = await self._resolve_mcp(request, require_tool=True)
        if isinstance(resolved, Response):
            return resolved
        session_id, server, _toolset, tool_name = resolved
        assert tool_name is not None

        vault_id, headers = await self._load_auth_for(session_id, server.url)
        log.info(
            "tool_broker.invoke_mcp",
            session_id=session_id,
            server=server.name,
            tool=tool_name,
        )
        envelope = await call_mcp_tool(
            server.url, vault_id, headers, tool_name, arguments, spec_headers=server.headers
        )
        return JSONResponse(envelope)


__all__ = ["ToolBroker"]
