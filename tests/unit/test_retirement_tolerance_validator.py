"""Registry-driven, telemetered read-tolerance for the ``mode="before"`` ToolSpec validator (#1574).

The ``mode="before"`` validator on :class:`ToolSpec` no longer reads a hand-maintained rename
constant; it consults the retirement REGISTRY (#1573) via
:func:`aios.retirements.registry.tolerated_rename_map`, which tolerates a token ONLY while its
descriptor's ``contract_rev IS NULL``. Every tolerance fire emits
``retirement_tolerance_hits{token}`` + stamps ``last_seen`` as **corroboration only** — never the
gate. These tests pin the #1574 acceptance:

* the validator consults the registry (a token absent from the still-tolerated map is NOT remapped);
* a token whose descriptor has ``contract_rev`` set is no longer tolerated (it raises);
* every fire records the metric + ``last_seen``, and the telemetry is gate-free.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import ValidationError

import aios.models.agents as agents_mod
from aios.models.agents import ToolSpec
from aios.retirements import Retirement, telemetry
from aios.retirements import registry as reg


@pytest.fixture(autouse=True)
def _clear_telemetry() -> Iterator[None]:
    telemetry.reset()
    yield
    telemetry.reset()


# ── The validator consults the registry (no hand-maintained constant) ─────────


def test_no_hand_maintained_rename_constant_remains() -> None:
    """The pre-#1574 ``_LEGACY_BUILTIN_RENAMES`` constant is gone; the map is registry-sourced."""
    assert not hasattr(agents_mod, "_LEGACY_BUILTIN_RENAMES")


def test_validator_uses_registry_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """A token present in the registry-driven map is remapped to its successor on read."""
    # A synthetic still-tolerated rename (contract_rev IS NULL) overrides the lookup.
    fake = {"legacy_synthetic": "bash"}
    monkeypatch.setattr(agents_mod, "tolerated_rename_map", lambda: fake)
    assert ToolSpec.model_validate({"type": "legacy_synthetic"}).type == "bash"


def test_token_absent_from_map_is_not_remapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A legacy token NOT in the still-tolerated map is left untouched (→ fails the Literal)."""
    monkeypatch.setattr(agents_mod, "tolerated_rename_map", lambda: {})
    with pytest.raises(ValidationError):
        ToolSpec.model_validate({"type": "invoke"})


# ── The contract_rev gate ─────────────────────────────────────────────────────


def _expand_span_rename() -> Retirement:
    return Retirement(
        domain=reg.TOOL_SURFACE_DOMAIN,
        action="rename",
        mappings=(("oldtok", "bash"),),
        surfaces=reg.TOOL_SURFACES,
        introduced_rev="9000",
        contract_rev=None,  # still expanding → tolerated
    )


def _contracted_rename() -> Retirement:
    return Retirement(
        domain=reg.TOOL_SURFACE_DOMAIN,
        action="rename",
        mappings=(("oldtok", "bash"),),
        surfaces=reg.TOOL_SURFACES,
        introduced_rev="9000",
        contract_rev="9001",  # contract migration declared → NO LONGER tolerated
    )


def test_contract_rev_null_token_is_tolerated() -> None:
    m = reg.tolerated_rename_map(registry=(_expand_span_rename(),))
    assert m == {"oldtok": "bash"}


def test_contract_rev_set_token_drops_out_of_map() -> None:
    m = reg.tolerated_rename_map(registry=(_contracted_rename(),))
    assert m == {}


def test_contract_rev_set_token_is_no_longer_tolerated(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a contract_rev stamped, the validator no longer remaps the token → it raises."""
    contracted = _contracted_rename()
    monkeypatch.setattr(
        agents_mod,
        "tolerated_rename_map",
        lambda: reg.tolerated_rename_map(registry=(contracted,)),
    )
    with pytest.raises(ValidationError):
        ToolSpec.model_validate({"type": "oldtok"})


def test_drop_descriptors_never_contribute_to_rename_map() -> None:
    """``action="drop"`` retirements have no successor → absent from the rename map."""
    m = reg.tolerated_rename_map(registry=(reg.RETIRED_GOAL_OUTCOME_BUILTINS,))
    assert m == {}


# ── Telemetry: corroboration only, NEVER the gate ─────────────────────────────


def test_every_fire_increments_hits_and_stamps_last_seen() -> None:
    assert telemetry.tolerance_hits("invoke") == 0
    assert telemetry.last_seen("invoke") is None

    ToolSpec.model_validate({"type": "invoke"})
    ToolSpec.model_validate({"type": "invoke"})

    assert telemetry.tolerance_hits("invoke") == 2
    assert telemetry.last_seen("invoke") is not None


def test_telemetry_is_keyed_per_token() -> None:
    ToolSpec.model_validate({"type": "invoke"})
    ToolSpec.model_validate({"type": "cancel_run"})
    assert telemetry.tolerance_hits("invoke") == 1
    assert telemetry.tolerance_hits("cancel_run") == 1
    assert telemetry.tolerance_hits("invoke_agent") == 0


def test_canonical_name_does_not_fire_telemetry() -> None:
    ToolSpec.model_validate({"type": "call_workflow"})
    assert telemetry.tolerance_hits("call_workflow") == 0


def test_telemetry_sink_failure_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blowing-up Prometheus sink must not propagate — telemetry is never load-bearing."""

    class _BoomCounter:
        def labels(self, **_kwargs: object) -> _BoomCounter:
            return self

        def inc(self) -> None:
            raise RuntimeError("metrics surface down")

    monkeypatch.setattr(telemetry, "_RETIREMENT_TOLERANCE_HITS", _BoomCounter())
    # The sink raises internally, but the call returns normally and the in-process tally is kept.
    telemetry.record_tolerance_hit("invoke")  # must not raise
    assert telemetry.tolerance_hits("invoke") == 1


def test_validator_maps_token_even_when_sink_is_broken(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken telemetry sink must never wedge a read — the gate decision still applies."""

    class _BoomCounter:
        def labels(self, **_kwargs: object) -> _BoomCounter:
            return self

        def inc(self) -> None:
            raise RuntimeError("metrics surface down")

    monkeypatch.setattr(telemetry, "_RETIREMENT_TOLERANCE_HITS", _BoomCounter())
    assert ToolSpec.model_validate({"type": "invoke"}).type == "call_session"


def test_telemetry_not_consulted_as_gate() -> None:
    """Pre-seeding hit counts must not change whether a token is tolerated."""
    telemetry.record_tolerance_hit("call_workflow")  # nonsense corroboration for a canonical name
    # call_workflow is canonical; it is never remapped regardless of any recorded hits.
    assert ToolSpec.model_validate({"type": "call_workflow"}).type == "call_workflow"
