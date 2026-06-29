"""The ``workflow:`` model-binding privilege — one runtime guard, all spawn paths (#1636).

Part of the **Workflows-as-Models** epic. The :mod:`aios.harness.model_binding`
boundary recognises a ``model = "workflow:<id>[@version]"`` binding and dispatches
the step's inference through a workflow run. This module owns the **authorization**
question that boundary never asked: *who may bind or select such a model?*

**Why a runtime guard, not an author-time one.** ``_enforce_surface_attenuation``
(``services.agents``) clamps the tool/mcp/http axis but NOT the model/api_base axis,
and a computed ``workflow:`` model string is invisible to the static authoring
validator. There are three distinct inference-spawning paths that can introduce a
``workflow:`` binding:

* ``create_agent`` / ``update_agent`` — the free-form ``model`` field (``services.agents``).
* the per-call ``agent(model=…)`` override (``workflows.step``).
* the **generic agentless child** (``agent()`` with no ``agent_id``), which carries
  the run's resolved model and previously had no model check at all.

Rather than bolt a separate check onto each, the privilege is enforced at the
**runtime dispatch seam** each path funnels through, keyed on the run/session's
**owning principal** — operator vs self-authoring — read at call time, not assumed
at author time. The privilege is **operator-only to start**: a self-authoring
(non-operator) principal may neither bind nor select a ``workflow:`` model via any
path; an operator may. This is a property of the guard, not a precondition — the
legacy "``create_agent`` is operator-only" assumption in ``config.py`` is stale.

The companion #823 api_base spawn-edge clamp (``workflows.step``) is orthogonal and
stays: it bounds *where* a child's inference routes; this guard bounds *whether* a
principal may route inference through a workflow at all.
"""

from __future__ import annotations

from aios.errors import ForbiddenError
from aios.harness.model_binding import is_workflow_model


def is_workflow_binding(model: str | None) -> bool:
    """True iff ``model`` is a ``workflow:`` binding (``None``/raw provider → False)."""
    return model is not None and is_workflow_model(model)


def enforce_workflow_binding_privilege(model: str | None, *, is_operator: bool) -> None:
    """Raise :class:`ForbiddenError` if a non-operator principal binds/selects a
    ``workflow:`` model.

    The single privilege check shared by every spawn path. ``model`` is the model
    string a path is about to bind or select (the ``create_agent``/``update_agent``
    ``model`` field, the per-call ``agent(model=…)`` override, or a generic child's
    resolved model). ``is_operator`` is whether the **owning principal** of the
    acting run/session is the operator (an operator/HTTP-launched edge) rather than
    a self-authoring agent.

    A no-op for a raw provider model (the overwhelmingly common case) or ``None``.
    For a ``workflow:`` binding it permits the operator and fails closed for a
    self-authoring principal — the binding privilege is operator-only to start.
    """
    if is_operator:
        return
    if is_workflow_binding(model):
        raise ForbiddenError(
            "binding or selecting a workflow: model is operator-only; a self-authoring "
            "principal may not route inference through a workflow",
            detail={"model": model},
        )
