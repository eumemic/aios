from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="UsageCounters")


@_attrs_define
class UsageCounters:
    """Cumulative inference bought at one node or in one subtree.

    Attributes:
        cost_microusd (int | Unset):  Default: 0.
        input_tokens (int | Unset):  Default: 0.
        output_tokens (int | Unset):  Default: 0.
        cache_read_input_tokens (int | Unset):  Default: 0.
        cache_creation_input_tokens (int | Unset):  Default: 0.
        tokens_complete (bool | Unset):  Default: True.
    """

    cost_microusd: int | Unset = 0
    input_tokens: int | Unset = 0
    output_tokens: int | Unset = 0
    cache_read_input_tokens: int | Unset = 0
    cache_creation_input_tokens: int | Unset = 0
    tokens_complete: bool | Unset = True
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        cost_microusd = self.cost_microusd

        input_tokens = self.input_tokens

        output_tokens = self.output_tokens

        cache_read_input_tokens = self.cache_read_input_tokens

        cache_creation_input_tokens = self.cache_creation_input_tokens

        tokens_complete = self.tokens_complete

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if cost_microusd is not UNSET:
            field_dict["cost_microusd"] = cost_microusd
        if input_tokens is not UNSET:
            field_dict["input_tokens"] = input_tokens
        if output_tokens is not UNSET:
            field_dict["output_tokens"] = output_tokens
        if cache_read_input_tokens is not UNSET:
            field_dict["cache_read_input_tokens"] = cache_read_input_tokens
        if cache_creation_input_tokens is not UNSET:
            field_dict["cache_creation_input_tokens"] = cache_creation_input_tokens
        if tokens_complete is not UNSET:
            field_dict["tokens_complete"] = tokens_complete

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        cost_microusd = d.pop("cost_microusd", UNSET)

        input_tokens = d.pop("input_tokens", UNSET)

        output_tokens = d.pop("output_tokens", UNSET)

        cache_read_input_tokens = d.pop("cache_read_input_tokens", UNSET)

        cache_creation_input_tokens = d.pop("cache_creation_input_tokens", UNSET)

        tokens_complete = d.pop("tokens_complete", UNSET)

        usage_counters = cls(
            cost_microusd=cost_microusd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            tokens_complete=tokens_complete,
        )

        usage_counters.additional_properties = d
        return usage_counters

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
