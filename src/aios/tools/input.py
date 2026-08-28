"""Schema-first tool-argument parsing.

A single shared util for the *schema-first* builtin authoring path: pair
``parameters_schema=Model.model_json_schema()`` at registration with
``tool_input(Model, arguments)`` in the handler, so the schema the model
SEES and the schema the handler ENFORCES derive from one Pydantic arg
model.

Lifted verbatim from ``workflow_management._parse`` so a second module
reaching for the same idiom no longer has to reinvent it (and cannot
drift, as the trigger handlers' bare ``model_validate`` had).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from aios.tools.invoke import ToolBail


def tool_input[M: BaseModel](model: type[M], arguments: dict[str, Any]) -> M:
    """Parse + validate tool arguments through a Pydantic arg model — the
    schema-first authoring path. Pair with
    ``parameters_schema=Model.model_json_schema()`` at registration so the
    schema the model SEES and the schema the handler ENFORCES are one
    source. Set ``model_config = ConfigDict(extra="forbid")`` on the arg
    model so a model-injected trusted-id key (creator_session_id,
    account_id, ...) is rejected — both by jsonschema at dispatch
    (additionalProperties:false) and by Pydantic here. Semantic failures
    the JSON Schema can't encode (custom field_validators, cross-field
    rules) surface as ToolBail, so the model self-corrects through the
    session log instead of evicting the sandbox.

    The one sanctioned divergence is the model-facing schema diet
    (:mod:`aios.tools.schema_diet`, #2294): a registration may pass
    ``slim_tool_schema(Model.model_json_schema())``, which only ever
    *loosens* what the dispatch-time jsonschema check enforces (it drops
    boilerplate and collapses a config subtree to a bare array) and never
    touches ``additionalProperties: false``. This model stays the enforcing
    source of truth, so what the diet stops catching at dispatch is caught
    here instead — with a field-precise pydantic error rather than a
    schema one.
    """
    try:
        return model.model_validate(arguments)
    except PydanticValidationError as exc:
        raise ToolBail(f"invalid arguments: {exc}") from exc
