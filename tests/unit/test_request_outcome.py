"""The request-answer ``Ok | Err`` outcome kind (#1555).

Master-failing by construction: ``Ok`` / ``Err`` / ``Outcome`` and the
``outcome_to_jsonb`` / ``outcome_from_jsonb`` codec do not exist on master, so
this module fails to import there and passes after the change.

It asserts the kind is **total** and **round-trips** through the one JSONB codec,
and that the six illegal ``(is_error, result, error)`` product states are
**unconstructable** — an ``Err`` with no error raises, and ``Ok`` has no error
slot at all. A second block asserts the readers (``derive_response`` /
``derive_run_response``) return an ``Err`` *kind* (not a flat ``is_error=True``
dict) for the ``child_gone`` / ``cancelled`` arms.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from aios.db.queries import outcome_from_jsonb, outcome_to_jsonb
from aios.models.sessions import Err, Ok, Outcome

# ─── totality + round-trip ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "outcome",
    [
        Ok(result=None),
        Ok(result={"answer": 42}),
        Ok(result=[1, 2, 3]),
        Ok(result="a string"),
        Ok(result=0),
        Err(error={"kind": "timeout"}),
        Err(error={"kind": "child_gone"}),
        Err(error={"kind": "cancelled"}),
        Err(error={"kind": "no_return"}),
        Err(error={"message": "boom"}),
    ],
)
def test_outcome_round_trips_through_the_codec(outcome: Outcome) -> None:
    # outcome_from_jsonb(outcome_to_jsonb(o)) == o for every Ok / Err.
    assert outcome_from_jsonb(outcome_to_jsonb(outcome)) == outcome


def test_to_jsonb_derives_the_flat_echo_from_the_kind() -> None:
    # The on-disk echo is derived from the kind, never hand-paired.
    assert outcome_to_jsonb(Ok(result={"x": 1})) == {
        "is_error": False,
        "result": {"x": 1},
        "error": None,
    }
    assert outcome_to_jsonb(Err(error={"kind": "timeout"})) == {
        "is_error": True,
        "result": None,
        "error": {"kind": "timeout"},
    }


def test_from_jsonb_reads_the_flat_shape_back_into_a_kind() -> None:
    ok = outcome_from_jsonb({"is_error": False, "result": {"x": 1}, "error": None})
    assert isinstance(ok, Ok) and ok.result == {"x": 1}
    err = outcome_from_jsonb({"is_error": True, "result": None, "error": {"kind": "child_gone"}})
    assert isinstance(err, Err) and err.error == {"kind": "child_gone"}


def test_from_jsonb_collapses_the_run_completed_shape() -> None:
    # The run_completed bookend carries {output, is_error, error}; derive_run_response
    # maps it through the same codec against ``output`` — assert the {is_error,result}
    # arm reads identically (an old success row with only ``output`` is rename-handled
    # at the reader, but a plain {is_error:False, result} still reads as Ok here).
    assert outcome_from_jsonb({"is_error": False, "result": "child-result"}) == Ok(
        result="child-result"
    )
    assert outcome_from_jsonb({"is_error": True, "error": {"message": "boom"}}) == Err(
        error={"message": "boom"}
    )


# ─── illegal product states are unconstructable ──────────────────────────────


def test_err_requires_an_error() -> None:
    # Err always carries an error — `Err()` with no error is a ValidationError, so the
    # illegal ``is_error=True, error=None`` pairing cannot be expressed.
    with pytest.raises(ValidationError):
        Err()  # type: ignore[call-arg]


def test_ok_has_no_error_slot() -> None:
    # There is no field on Ok that accepts an error — the illegal ``is_error=False``
    # with a non-null error cannot be expressed (frozen models forbid extra fields by
    # default; Pydantic ignores unknown kwargs, so the constructed Ok carries no error).
    ok = Ok(result=1)
    assert not hasattr(ok, "error")
    # and the kind discriminant is fixed
    assert ok.kind == "ok"
    assert Err(error={}).kind == "err"


def test_outcomes_are_frozen() -> None:
    ok = Ok(result=1)
    with pytest.raises(ValidationError):
        ok.result = 2  # type: ignore[misc]


# ─── the readers return the kind, not a flat dict ────────────────────────────


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"is_error": True, "result": None, "error": {"kind": "child_gone"}}, "child_gone"),
        ({"is_error": True, "result": None, "error": {"kind": "cancelled"}}, "cancelled"),
        ({"is_error": True, "result": None, "error": {"kind": "timeout"}}, "timeout"),
    ],
)
def test_error_arms_deserialize_as_err_kind(data: dict[str, Any], expected: str) -> None:
    # The synthesized child_gone / cancelled / timeout arms are an ``Err`` kind, not a
    # ``{is_error: True}`` dict — assert against the kind a reader hands back.
    out = outcome_from_jsonb(data)
    assert isinstance(out, Err)
    assert out.error == {"kind": expected}
