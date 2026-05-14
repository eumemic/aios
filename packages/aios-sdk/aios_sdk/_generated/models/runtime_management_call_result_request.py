from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="RuntimeManagementCallResultRequest")


@_attrs_define
class RuntimeManagementCallResultRequest:
    """Idempotent on ``call_id`` — a replay POST against an already-resolved
    row no-ops (no double-NOTIFY).

        Attributes:
            call_id (str):
            result (Any | Unset):
            is_error (bool | Unset):  Default: False.
    """

    call_id: str
    result: Any | Unset = UNSET
    is_error: bool | Unset = False

    def to_dict(self) -> dict[str, Any]:
        call_id = self.call_id

        result = self.result

        is_error = self.is_error

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "call_id": call_id,
            }
        )
        if result is not UNSET:
            field_dict["result"] = result
        if is_error is not UNSET:
            field_dict["is_error"] = is_error

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        call_id = d.pop("call_id")

        result = d.pop("result", UNSET)

        is_error = d.pop("is_error", UNSET)

        runtime_management_call_result_request = cls(
            call_id=call_id,
            result=result,
            is_error=is_error,
        )

        return runtime_management_call_result_request
