from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="SignalUnregisterCancelResponse")


@_attrs_define
class SignalUnregisterCancelResponse:
    """
    Attributes:
        cancelled_call_ids (list[str]):
    """

    cancelled_call_ids: list[str]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        cancelled_call_ids = self.cancelled_call_ids

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "cancelled_call_ids": cancelled_call_ids,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        cancelled_call_ids = cast(list[str], d.pop("cancelled_call_ids"))

        signal_unregister_cancel_response = cls(
            cancelled_call_ids=cancelled_call_ids,
        )

        signal_unregister_cancel_response.additional_properties = d
        return signal_unregister_cancel_response

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
