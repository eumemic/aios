from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.takeover_close_request_outcome import TakeoverCloseRequestOutcome
from ..types import UNSET, Unset

T = TypeVar("T", bound="TakeoverCloseRequest")


@_attrs_define
class TakeoverCloseRequest:
    """
    Attributes:
        outcome (TakeoverCloseRequestOutcome | Unset):  Default: TakeoverCloseRequestOutcome.DONE.
    """

    outcome: TakeoverCloseRequestOutcome | Unset = TakeoverCloseRequestOutcome.DONE
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        outcome: str | Unset = UNSET
        if not isinstance(self.outcome, Unset):
            outcome = self.outcome.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if outcome is not UNSET:
            field_dict["outcome"] = outcome

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        _outcome = d.pop("outcome", UNSET)
        outcome: TakeoverCloseRequestOutcome | Unset
        if isinstance(_outcome, Unset):
            outcome = UNSET
        else:
            outcome = TakeoverCloseRequestOutcome(_outcome)

        takeover_close_request = cls(
            outcome=outcome,
        )

        takeover_close_request.additional_properties = d
        return takeover_close_request

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
