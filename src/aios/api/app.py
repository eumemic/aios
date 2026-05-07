"""FastAPI app factory.

Builds the app, wires in the routers, the exception handlers, and the
lifespan that opens/closes the asyncpg pool and constructs the CryptoBox.
Also mounts the auto-reflected MCP server at ``/mcp`` so agents can
operate the management plane via the same Bearer auth as the HTTP API.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI

from aios.api.deps import require_bearer_auth
from aios.api.routers import (
    agents,
    connections,
    connectors,
    environments,
    health,
    memory_stores,
    session_templates,
    sessions,
    skills,
    vaults,
)
from aios.config import get_settings
from aios.crypto.vault import CryptoBox
from aios.db.pool import close_pool, create_pool
from aios.errors import install_exception_handlers
from aios.harness import runtime
from aios.harness.procrastinate_app import app as procrastinate_app
from aios.logging import configure_logging, get_logger


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger("aios.api")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        log.info("api.startup", db_url=_redact_dsn(settings.db_url))
        pool = await create_pool(settings.db_url, max_size=settings.db_pool_max_size)
        crypto_box = CryptoBox.from_base64(settings.vault_key.get_secret_value())
        await procrastinate_app.open_async()
        app.state.pool = pool
        app.state.crypto_box = crypto_box
        app.state.procrastinate = procrastinate_app
        app.state.db_url = settings.db_url
        # The ``/context`` endpoint (issue #60) reuses the worker's
        # ``compose_step_context`` → ``discover_session_mcp_tools`` path,
        # which reaches for ``runtime.require_crypto_box()`` to decrypt
        # per-vault MCP OAuth tokens.  Mirror the worker's globals on the
        # API side so the shared composer works from either process.
        prev_crypto = runtime.crypto_box
        runtime.crypto_box = crypto_box
        try:
            yield
        finally:
            log.info("api.shutdown")
            runtime.crypto_box = prev_crypto
            await procrastinate_app.close_async()
            await pool.close()
            await close_pool()

    app = FastAPI(
        title="aios",
        version="0.1.0",
        description="Open-source agent runtime: Postgres-backed sessions, "
        "Docker sandbox, any LiteLLM model.",
        lifespan=lifespan,
        # One schema per Pydantic model (not a Foo-Input / Foo-Output pair).
        # We use distinct ``Foo`` / ``FooCreate`` / ``FooUpdate`` types instead
        # of the read-only-field split, and the hyphenated split-schema names
        # break openapi-python-client when it generates the typed SDK.
        separate_input_output_schemas=False,
    )
    install_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(environments.router)
    app.include_router(agents.router)
    app.include_router(sessions.router)
    app.include_router(skills.router)
    app.include_router(vaults.router)
    app.include_router(memory_stores.router)
    app.include_router(connections.router)
    app.include_router(connectors.router)
    app.include_router(session_templates.router)
    _mount_mcp(app)
    return app


MCP_INSTRUCTIONS = """\
aios is a Postgres-backed agent runtime. The management plane is organized
around five resource types: agents (versioned config), sessions (live
execution contexts), environments (sandbox templates), memory stores
(shared file-like memory), and vaults (encrypted credentials). Resources
attach to sessions to grant agent read/write access.

Conventions:
- ``archive`` is reversible soft-removal preserving audit history;
  ``delete`` is hard-removal, no audit trail. Prefer archive unless you
  specifically need rows gone.
- Listed resources exclude archived by default unless ``include_archived``
  is supported on the operation.
- Mutations marked ``destructiveHint`` should be confirmed before invoking.
"""


def _mount_mcp(app: FastAPI) -> None:
    """Mount fastapi-mcp at ``/mcp`` after all routers are registered.

    The MCP tool surface is whatever FastAPI's OpenAPI spec says it is,
    minus operations whose ``x-codegen.targets`` excludes ``"mcp"``.
    Today that excludes the SSE / long-poll session-event routes (no
    JSON response shape) and the connectors RPC routes (raw
    ``dict[str, Any]`` envelopes that need Pydantic models authored
    before they MCP cleanly).
    """
    # fastapi-mcp doesn't ship a py.typed marker yet, so mypy can't see
    # the package's actual signatures.
    from fastapi_mcp import AuthConfig, FastApiMCP  # type: ignore[import-untyped]

    mcp = FastApiMCP(
        app,
        name="aios",
        description=(
            "aios management plane: agents, sessions, environments, vaults, "
            "memory stores, skills, connections, session templates."
        ),
        exclude_operations=_compute_mcp_excluded_operations(app),
        auth_config=AuthConfig(dependencies=[Depends(require_bearer_auth)]),
    )
    _apply_mcp_polish(app, mcp, instructions=MCP_INSTRUCTIONS)
    mcp.mount_http(mount_path="/mcp")


def _iter_operations(app: FastAPI) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield ``(operation_id, http_verb, x_codegen_dict)`` for every operation.

    Skips path-level keys that aren't HTTP methods (parameters, summary,
    description, ...) and operations without an ``operationId``. The
    ``x_codegen_dict`` is the raw ``x-codegen`` extension content (or
    ``{}`` if absent); each caller projects out the sub-key it needs.
    """
    for path_obj in app.openapi()["paths"].values():
        for verb, method_obj in path_obj.items():
            if not isinstance(method_obj, dict):
                continue
            op_id = method_obj.get("operationId")
            if not isinstance(op_id, str):
                continue
            yield op_id, verb.lower(), method_obj.get("x-codegen", {}) or {}


def _verb_default_annotations(verb: str) -> dict[str, bool]:
    """Map HTTP verb to default MCP tool annotation hints.

    GET → read-only. DELETE → destructive. PUT → destructive + idempotent.
    POST defaults to no annotations (additive by REST convention); routes
    that are POST-but-destructive (e.g. ``.../archive``) override via
    ``x-codegen.mcp`` per route.
    """
    if verb == "get":
        return {"readOnlyHint": True}
    if verb == "delete":
        return {"destructiveHint": True}
    if verb == "put":
        return {"destructiveHint": True, "idempotentHint": True}
    return {}


def _apply_mcp_polish(app: FastAPI, mcp: Any, *, instructions: str) -> None:
    """Post-construction patch: per-tool annotations + server instructions.

    fastapi-mcp 0.4 doesn't infer annotations from HTTP verbs and exposes
    no construction-time hook for them, so we walk the spec, build verb-
    based defaults, layer the route's ``x-codegen.mcp`` overrides on top,
    and assign the result to each ``mcp.tools[i].annotations``. Also sets
    ``mcp.server.instructions`` (the MCP handshake field, inlined into the
    calling LLM's system prompt by clients that respect it).
    """
    from mcp.types import ToolAnnotations

    op_meta: dict[str, tuple[str, dict[str, Any]]] = {
        op_id: (verb, codegen.get("mcp", {}) or {})
        for op_id, verb, codegen in _iter_operations(app)
    }

    for tool in mcp.tools:
        meta = op_meta.get(tool.name)
        if meta is None:
            continue
        verb, overrides = meta
        annotations = _verb_default_annotations(verb) | overrides
        if annotations:
            tool.annotations = ToolAnnotations(**annotations)

    mcp.server.instructions = instructions


def _compute_mcp_excluded_operations(app: FastAPI) -> list[str]:
    """Return operationIds whose ``x-codegen.targets`` excludes ``"mcp"``.

    Routes without an ``x-codegen.targets`` list default to included in
    every generator target (per the contract documented in #267); only
    explicit ``targets`` lists that omit ``"mcp"`` exclude.
    """
    return [
        op_id
        for op_id, _verb, codegen in _iter_operations(app)
        if (targets := codegen.get("targets")) is not None and "mcp" not in targets
    ]


def _redact_dsn(dsn: str) -> str:
    """Strip the password from a Postgres DSN for safe logging."""
    if "@" not in dsn or "://" not in dsn:
        return dsn
    scheme, _, rest = dsn.partition("://")
    auth, _, host_part = rest.partition("@")
    if ":" in auth:
        user, _, _ = auth.partition(":")
        return f"{scheme}://{user}:***@{host_part}"
    return dsn


# An app instance importable as ``aios.api.app:app`` for ``uvicorn``.
app: Any = create_app()
