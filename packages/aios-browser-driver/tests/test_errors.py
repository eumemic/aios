"""The request-argument helpers — notably require_int's integral-float
acceptance, which keeps the driver from disagreeing with the worker's own
Draft 2020-12 schema gate (``2.0`` validates as an integer there)."""

from __future__ import annotations

import pytest
from aios_browser_driver.errors import ActionError, require_int, require_number, require_str


def test_require_str_rejects_empty_by_default() -> None:
    with pytest.raises(ActionError) as info:
        require_str({"k": ""}, "k")
    assert info.value.code == "invalid_request"


def test_require_str_allows_empty_when_asked() -> None:
    assert require_str({"k": ""}, "k", allow_empty=True) == ""


@pytest.mark.parametrize("value", [2, 2.0])
def test_require_int_accepts_int_and_integral_float(value: object) -> None:
    assert require_int({"k": value}, "k", lo=1, hi=3) == 2


@pytest.mark.parametrize("value", [2.5, 0, 4, "2", True, None])
def test_require_int_rejects_non_integral_out_of_range_and_wrong_type(value: object) -> None:
    with pytest.raises(ActionError) as info:
        require_int({"k": value}, "k", lo=1, hi=3)
    assert info.value.code == "invalid_request"


def test_require_number_rejects_bool() -> None:
    # bool is an int subclass — a stray True must not read as 1.
    with pytest.raises(ActionError):
        require_number({"k": True}, "k")
