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

from typing import Any

import pytest


@pytest.fixture(scope="module")
def app() -> Any:
    """A ``FastAPI`` instance shared across every test in this module.

    ``create_app()`` is idempotent and costs ~470 ms — paying for it once
    instead of five times saves ~2s on the unit suite.  Import is inside
    the fixture body because ``aios.harness.procrastinate_app`` runs
    ``get_settings()`` at module-import time and the conftest env-var
    fixture only fires after collection.
    """
    from aios.api.app import create_app

    return create_app()


@pytest.fixture(scope="module")
def polished_mcp(app: Any) -> Any:
    """Live ``FastApiMCP`` against ``app``'s spec, with polish applied.

    Same code path as ``_mount_mcp`` minus the HTTP mount.  Polish
    mutates the MCP instance, not the app, so sharing the underlying
    app across this fixture and the route-shape assertions below is
    safe.
    """
    from fastapi import Depends
    from fastapi_mcp import AuthConfig, FastApiMCP

    from aios.api.app import (
        MCP_INSTRUCTIONS,
        _apply_mcp_polish,
        _compute_mcp_excluded_operations,
    )
    from aios.api.deps import require_bearer_auth

    mcp = FastApiMCP(
        app,
        name="test",
        description="test",
        exclude_operations=_compute_mcp_excluded_operations(app),
        auth_config=AuthConfig(dependencies=[Depends(require_bearer_auth)]),
    )
    _apply_mcp_polish(app, mcp, instructions=MCP_INSTRUCTIONS)
    return mcp


def test_mcp_route_is_mounted(app: Any) -> None:
    assert any(getattr(r, "path", None) == "/mcp" for r in app.routes), (
        "FastApiMCP did not register a /mcp route during create_app()"
    )


def test_excluded_operations_match_x_codegen_annotations(app: Any) -> None:
    """Routes annotated with ``x-codegen.targets: []`` must be excluded from MCP.

    Today that's the 2 SSE / long-poll session-event routes (no JSON
    response shape) and the 6 connectors RPC routes (raw ``dict[str, Any]``
    envelopes deferred per #267). Routes without the annotation default
    to included.
    """
    from aios.api.app import _compute_mcp_excluded_operations

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


def test_apply_mcp_polish_populates_annotations_and_instructions(polished_mcp: Any) -> None:
    """End-to-end: the MCP tool list has verb-default annotations + overrides.

    Verifies representative tools across each annotation-bearing category.
    """
    from aios.api.app import MCP_INSTRUCTIONS

    by_name = {tool.name: tool for tool in polished_mcp.tools}

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
    assert by_name["archive_session"].annotations.destructiveHint is True
    assert by_name["archive_memory_store"].annotations.destructiveHint is True
    assert by_name["archive_vault_credential"].annotations.destructiveHint is True
    assert by_name["detach_connection"].annotations.destructiveHint is True
    assert by_name["unconfigure_connection"].annotations.destructiveHint is True
    assert by_name["redact_memory_version"].annotations.destructiveHint is True

    # POST create → no annotations (additive; no override needed)
    assert by_name["create_agent"].annotations is None

    # Server-level instructions populated
    assert polished_mcp.server.instructions == MCP_INSTRUCTIONS


def test_apply_mcp_polish_cleans_schemas(polished_mcp: Any) -> None:
    """End-to-end: every MCP tool ships with cleaned-up descriptions and inputSchemas.

    Three transforms applied per tool:
    - The auto-appended ``### Responses:`` example block is gone.
    - Redundant ``title`` fields (whose value matches the property key) are dropped.
    - Two-branch ``anyOf: [T, {"type": "null"}]`` is flattened to ``type: [T, "null"]``.

    Verified against representative tools that exercise each transform —
    plus a deliberate non-target case to confirm complex multi-branch
    ``anyOf`` schemas are preserved.
    """
    by_name = {tool.name: tool for tool in polished_mcp.tools}

    # Description-strip: the ``### Responses:`` block + example payload are gone
    archive = by_name["archive_session"]
    assert archive.description is not None
    assert "### Responses:" not in archive.description
    assert "Example Response" not in archive.description
    # The route's actual prose ("Archive") survives
    assert "Archive" in archive.description

    # Title-drop: ``session_id`` property no longer carries a redundant ``title``
    session_id_prop = archive.inputSchema["properties"]["session_id"]
    assert "title" not in session_id_prop
    assert session_id_prop["type"] == "string"

    # Nullable-anyOf flatten: simple nullable string collapses to type-array
    list_conn = by_name["list_connections"]
    connector_prop = list_conn.inputSchema["properties"]["connector"]
    assert "anyOf" not in connector_prop
    assert connector_prop["type"] == ["string", "null"]

    # Nullable-anyOf with enum: ``mode`` keeps its enum, gains nullable type-array
    mode_prop = list_conn.inputSchema["properties"]["mode"]
    assert "anyOf" not in mode_prop
    assert mode_prop["type"] == ["string", "null"]
    assert set(mode_prop["enum"]) == {"detached", "single_session", "per_chat"}

    # Preservation: the ``tools[].type`` two-branch enum union (no null branch)
    # is left alone — flattening it would collapse the BuiltinToolType vs.
    # custom/mcp_toolset distinction the Pydantic union encodes.
    tools_type_prop = by_name["create_agent"].inputSchema["properties"]["tools"]["items"][
        "properties"
    ]["type"]
    assert "anyOf" in tools_type_prop
    assert len(tools_type_prop["anyOf"]) == 2
