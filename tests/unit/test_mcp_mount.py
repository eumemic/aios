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
