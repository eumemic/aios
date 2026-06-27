"""The retirement registry — the single source of truth for declared retirements.

:data:`REGISTRY` is consulted (in dependent issues) by the read-tolerance
validator, the boot-gate, and the data-migration generator. Nothing else should
maintain a parallel list of retired tokens or affected tables; if a surface is
missing here it is missing everywhere, by construction.

This module seeds two descriptors in the ``tool_surface`` domain — the persisted
``ToolSpec`` ``type`` value space — and registers **all seven** persisted
tool-surface columns on each.
"""

from __future__ import annotations

from aios.retirements import Retirement, Surface

# ---------------------------------------------------------------------------
# The ``tool_surface`` domain.
#
# A persisted tool-surface column holds a JSONB array of ToolSpec objects; a
# retired tool ``type`` is detected by the same predicate everywhere: does any
# array element's ``type`` equal the token?  Declared once here as a template
# (``<jsonb_col>`` is substituted per-surface; ``:token`` is bound by the
# generator) so every surface of the domain shares one predicate.
# ---------------------------------------------------------------------------

TOOL_SURFACE_DOMAIN = "tool_surface"

#: Shared predicate for the tool-surface domain. ``<jsonb_col>`` is substituted
#: with each surface's column; ``:token`` is bound to the retired ``type``.
TOOL_SURFACE_PREDICATE_SQL = (
    "EXISTS(SELECT 1 FROM jsonb_array_elements(<jsonb_col>) e WHERE e->>'type' = :token)"
)


def _tool_surface(table: str, jsonb_col: str, *, nullable: bool = False) -> Surface:
    """Build a ``tool_surface`` :class:`Surface` with the shared predicate."""

    return Surface(
        table=table,
        jsonb_col=jsonb_col,
        predicate_sql=TOOL_SURFACE_PREDICATE_SQL.replace("<jsonb_col>", jsonb_col),
        nullable=nullable,
    )


#: ALL SEVEN persisted tool surfaces. The first six were enumerated by the
#: ad-hoc #1419 retirement (migration 0116); the seventh —
#: ``connectors.tools_schema`` — is the silent hole that retirement missed. It
#: stores a connector type's served ToolSpecs (``db/queries/connections.py:598``)
#: and is consumed via ``ToolSpec.model_validate`` at
#: ``harness/step_context.py:258``, so a retired tool ``type`` left there poisons
#: the same read path as the other six. Declaring it as data closes the hole for
#: every present and future tool-surface retirement.
TOOL_SURFACES: tuple[Surface, ...] = (
    _tool_surface("agents", "tools"),
    _tool_surface("agent_versions", "tools"),
    _tool_surface("workflows", "tools"),
    _tool_surface("workflow_versions", "tools"),
    # Launch-pinned surface: the tools a run was launched with.
    _tool_surface("wf_runs", "tools"),
    # Frozen-child surface: populated only on frozen child sessions → nullable.
    _tool_surface("sessions", "tools", nullable=True),
    # THE SEVENTH SURFACE — the connector tools_schema the ad-hoc retirement
    # missed.
    _tool_surface("connectors", "tools_schema"),
)


# ---------------------------------------------------------------------------
# Seeded descriptors.
# ---------------------------------------------------------------------------

#: #1419 invocation-kernel rename batch, teardown-tracked as #1432. The model
#: tools ``invoke``→``call_session``, ``invoke_agent``→``call_agent``, and the
#: ``invoke_workflow``/``create_run``/``await_run`` launch trio folded into the
#: unified ``call_workflow``; ``cancel_run``→``stop_task`` followed in #1428.
#: The read-tolerance shim lives as ``_LEGACY_BUILTIN_RENAMES`` in
#: ``models/agents.py`` (introduced with migration 0116); the data migrations
#: 0116 (#1419) and 0117 (#1428) rewrite persisted rows to canonical.
LEGACY_BUILTIN_RENAMES = Retirement(
    domain=TOOL_SURFACE_DOMAIN,
    action="rename",
    mappings=(
        ("invoke", "call_session"),
        ("invoke_agent", "call_agent"),
        ("invoke_workflow", "call_workflow"),
        ("create_run", "call_workflow"),
        ("await_run", "call_workflow"),
        ("cancel_run", "stop_task"),
    ),
    surfaces=TOOL_SURFACES,
    introduced_rev="0116",
    contract_rev="0117",
    sla_days=30,
)

#: Post-#1563 builtin-shim retirement (teardown-tracked as #1569). #1525
#: removed ``complete_goal``/``fail_goal`` from ``BuiltinToolType`` + the registry
#: but shipped neither shim nor migration, wedging the live kedalion-ultron agent
#: into an infinite reschedule; #1563 closed the gap with the
#: ``_RETIRED_BUILTINS`` + ``load_tool_specs`` read shim and the 0122 data
#: migration. These builtins have NO canonical successor (``return``/``error``
#: are general step verbs, not model-listed builtins) → ``action=drop``.
RETIRED_GOAL_OUTCOME_BUILTINS = Retirement(
    domain=TOOL_SURFACE_DOMAIN,
    action="drop",
    mappings=(
        ("complete_goal", None),
        ("fail_goal", None),
    ),
    surfaces=TOOL_SURFACES,
    introduced_rev="0122",
    contract_rev="0122",
    sla_days=30,
)


#: The registry: the single list of declared retirements. Append new descriptors
#: here; downstream tooling (validator / boot-gate / migration generator)
#: consults this and nothing else.
REGISTRY: tuple[Retirement, ...] = (
    LEGACY_BUILTIN_RENAMES,
    RETIRED_GOAL_OUTCOME_BUILTINS,
)
