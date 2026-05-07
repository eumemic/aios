"""MCP mount sanity checks.

Verifies that ``create_app()`` mounts fastapi-mcp at ``/mcp`` and that the
projection contract (``x-codegen.targets`` excluding ``"mcp"``) translates
into the right ``exclude_operations`` list. Full wire-protocol coverage
(initialize / tools/list / tool dispatch) lives in the e2e suite where a
real uvicorn process is booted — fastapi-mcp's StreamableHTTP session
manager has lazy-startup timing that's brittle under FastAPI's TestClient.

The ``aios.api.app`` import is deferred to the test bodies because
``aios.harness.procrastinate_app`` runs ``get_settings()`` at module-import
time and the conftest env-var fixture only fires after collection.
"""

from __future__ import annotations


def test_mcp_route_is_mounted() -> None:
    from aios.api.app import create_app

    app = create_app()
    assert any(getattr(r, "path", None) == "/mcp" for r in app.routes), (
        "FastApiMCP did not register a /mcp route during create_app()"
    )


def test_excluded_operations_match_x_codegen_annotations() -> None:
    """Routes annotated with ``x-codegen.targets: []`` must be excluded from MCP.

    Today that's the 2 SSE / long-poll session-event routes (no JSON
    response shape) and the 6 connectors RPC routes (raw ``dict[str, Any]``
    envelopes deferred per #267). Routes without the annotation default
    to included.
    """
    from aios.api.app import _compute_mcp_excluded_operations, create_app

    app = create_app()
    excluded = set(_compute_mcp_excluded_operations(app))

    spec = app.openapi()
    # Cross-check against the spec: every operationId we excluded should
    # in fact carry an ``x-codegen.targets`` list that omits ``"mcp"``,
    # and no unannotated route should appear in the excluded set.
    for path_obj in spec["paths"].values():
        for method_obj in path_obj.values():
            if not isinstance(method_obj, dict):
                continue
            op_id = method_obj.get("operationId")
            codegen = method_obj.get("x-codegen") or {}
            targets = codegen.get("targets") if isinstance(codegen, dict) else None
            if isinstance(targets, list) and "mcp" not in targets:
                assert op_id in excluded, f"expected {op_id} to be excluded"
            else:
                assert op_id not in excluded, (
                    f"unexpected exclusion of {op_id} (no x-codegen.targets opt-out)"
                )


class TestVerbDefaultAnnotations:
    """``_verb_default_annotations`` maps HTTP verb to MCP annotation hints.

    GETs are read-only; DELETE/PUT are destructive; POST defaults to no
    annotations (additive by REST convention, with per-route overrides
    via ``x-codegen.mcp`` for POST routes that are actually destructive).
    """

    def test_get_is_read_only(self) -> None:
        from aios.api.app import _verb_default_annotations

        assert _verb_default_annotations("get") == {"readOnlyHint": True}

    def test_delete_is_destructive(self) -> None:
        from aios.api.app import _verb_default_annotations

        assert _verb_default_annotations("delete") == {"destructiveHint": True}

    def test_put_is_destructive_and_idempotent(self) -> None:
        from aios.api.app import _verb_default_annotations

        assert _verb_default_annotations("put") == {
            "destructiveHint": True,
            "idempotentHint": True,
        }

    def test_post_has_no_default_annotations(self) -> None:
        from aios.api.app import _verb_default_annotations

        assert _verb_default_annotations("post") == {}


def test_apply_mcp_polish_populates_annotations_and_instructions() -> None:
    """End-to-end: the MCP tool list has verb-default annotations + overrides.

    Constructs a fresh FastApiMCP against the live app's OpenAPI spec and
    applies the polish — same code path as ``_mount_mcp`` but without the
    HTTP mount. Verifies representative tools across each
    annotation-bearing category.
    """
    from fastapi import Depends
    from fastapi_mcp import AuthConfig, FastApiMCP

    from aios.api.app import (
        _MCP_INSTRUCTIONS,
        _apply_mcp_polish,
        _compute_mcp_excluded_operations,
        create_app,
    )
    from aios.api.deps import require_bearer_auth

    app = create_app()
    mcp = FastApiMCP(
        app,
        name="test",
        description="test",
        exclude_operations=_compute_mcp_excluded_operations(app),
        auth_config=AuthConfig(dependencies=[Depends(require_bearer_auth)]),
    )
    _apply_mcp_polish(app, mcp, instructions=_MCP_INSTRUCTIONS)

    by_name = {tool.name: tool for tool in mcp.tools}

    # GET → readOnlyHint
    assert by_name["list_agents"].annotations is not None
    assert by_name["list_agents"].annotations.readOnlyHint is True
    assert by_name["get_health"].annotations.readOnlyHint is True

    # DELETE → destructiveHint
    assert by_name["delete_vault"].annotations.destructiveHint is True

    # PUT → destructiveHint + idempotentHint
    assert by_name["update_agent"].annotations.destructiveHint is True
    assert by_name["update_agent"].annotations.idempotentHint is True

    # POST .../archive → destructiveHint via x-codegen.mcp override
    # (POST verb default is empty, so the annotation only appears because
    # of the explicit override stamped on the route)
    assert by_name["archive_vault"].annotations is not None
    assert by_name["archive_vault"].annotations.destructiveHint is True
    assert by_name["archive_memory_store"].annotations.destructiveHint is True
    assert by_name["archive_vault_credential"].annotations.destructiveHint is True
    assert by_name["detach_connection"].annotations.destructiveHint is True
    assert by_name["unconfigure_connection"].annotations.destructiveHint is True

    # POST create → no annotations (additive; no override needed)
    assert by_name["create_agent"].annotations is None

    # Server-level instructions populated
    assert mcp.server.instructions == _MCP_INSTRUCTIONS
