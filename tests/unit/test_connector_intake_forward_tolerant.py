"""Unit tests pinning the connector→api intake forward-tolerance fix (#1407).

The bug
-------
The connector-facing runtime intake models in
``src/aios/api/routers/connectors.py`` set ``model_config =
ConfigDict(extra="forbid")``. When a connector runs **ahead** of the api on a
coupled-schema change (a newer connector sends a field the older api doesn't
know yet), the api ``422``\\ s the POST (``{"type":"extra_forbidden", ...}``)
instead of degrading gracefully — the connector retries → crash-loops, the
tool-result never folds back → the session wedges (the 2026-06-19 Ultron
deploy-skew incident, memory ``project-ultron-422-deploy-skew-2026-06-19``).

The fix
-------
The connector-runtime intake models are made **forward-tolerant**
(``extra="ignore"``): an unknown extra field is silently dropped, the known
fields still validate and process. This is the forward half of the symmetry
#1398 established backward (a *newer* api defaulting ``no_reaction`` for an
*older* connector that omits it).

What this suite pins
--------------------
- Each connector-runtime intake model ACCEPTS a body carrying an unknown extra
  field (no ``ValidationError``) and parses the known fields normally.
- The unknown field is dropped (``extra="ignore"`` semantics) — it does not
  leak onto the model.
- The existing backward-compat behaviour (``no_reaction`` omitted → default
  ``False``) is unchanged.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from aios.api.routers.connectors import (
    RuntimeChatLifecycleRequest,
    RuntimeLifecycleRequest,
    RuntimeSessionLifecycleRequest,
    RuntimeToolResultRequest,
)

# (model, minimal-valid-body) for every connector-runtime intake a
# deployed-ahead connector POSTs to. These are exactly the models on the
# connector→api skew boundary; each must tolerate an unknown extra field.
_INTAKES: list[tuple[type[BaseModel], dict[str, object]]] = [
    (
        RuntimeToolResultRequest,
        {
            "connection_id": "conn_1",
            "session_id": "sess_1",
            "tool_call_id": "tc_1",
            "content": "ok",
        },
    ),
    (
        RuntimeLifecycleRequest,
        {"connection_id": "conn_1", "event": "signal.daemon.exited"},
    ),
    (
        RuntimeSessionLifecycleRequest,
        {"connection_id": "conn_1", "session_id": "sess_1", "event": "sms.delivery.failed"},
    ),
    (
        RuntimeChatLifecycleRequest,
        {"connection_id": "conn_1", "chat_id": "+15550001", "event": "sms.delivery.failed"},
    ),
]


@pytest.mark.parametrize("model,body", _INTAKES, ids=lambda v: getattr(v, "__name__", ""))
def test_intake_accepts_unknown_extra_field(
    model: type[BaseModel], body: dict[str, object]
) -> None:
    """A connector running AHEAD of the api adds a field the api doesn't know.
    The intake must accept it (2xx, not 422) and parse the known fields."""
    forward = {**body, "future_field": True}

    parsed = model.model_validate(forward)

    # Known fields are still parsed/validated and reachable to the handler.
    for key, value in body.items():
        assert getattr(parsed, key) == value
    # extra="ignore": the unknown field is dropped, not retained.
    assert not hasattr(parsed, "future_field")
    assert "future_field" not in parsed.model_dump()


class TestRuntimeToolResultRequest:
    """The intake that 422'd in the incident — pinned directly."""

    def test_tool_result_accepts_unknown_field(self) -> None:
        parsed = RuntimeToolResultRequest.model_validate(
            {
                "connection_id": "conn_1",
                "session_id": "sess_1",
                "tool_call_id": "tc_1",
                "content": "ok",
                "no_reaction": True,
                # exactly the incident shape: a newer connector adds a field
                # the older api has never heard of.
                "future_field": "anything",
            }
        )
        assert parsed.connection_id == "conn_1"
        assert parsed.no_reaction is True
        assert not hasattr(parsed, "future_field")

    def test_no_reaction_backward_compat_default_unchanged(self) -> None:
        """An older connector that OMITS ``no_reaction`` still defaults False —
        the backward half of the symmetry (#1398) is untouched."""
        parsed = RuntimeToolResultRequest.model_validate(
            {
                "connection_id": "conn_1",
                "session_id": "sess_1",
                "tool_call_id": "tc_1",
                "content": "ok",
            }
        )
        assert parsed.no_reaction is False
