from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="UsageRate")


@_attrs_define
class UsageRate:
    """Rolling-window inference rate, normalized to one hour.

    The usage ledger starts when the accounting migration lands.  Until one
    full requested window has elapsed, ``complete`` is false and
    ``observed_seconds`` states the actual denominator.  Rates remain useful
    immediately without pretending pre-ledger history was observed.

        Attributes:
            window_seconds (int):
            observed_seconds (int):
            complete (bool):
            cost_microusd_per_hour (float | Unset):  Default: 0.0.
            input_tokens_per_hour (float | Unset):  Default: 0.0.
            output_tokens_per_hour (float | Unset):  Default: 0.0.
            cache_read_input_tokens_per_hour (float | Unset):  Default: 0.0.
            cache_creation_input_tokens_per_hour (float | Unset):  Default: 0.0.
    """

    window_seconds: int
    observed_seconds: int
    complete: bool
    cost_microusd_per_hour: float | Unset = 0.0
    input_tokens_per_hour: float | Unset = 0.0
    output_tokens_per_hour: float | Unset = 0.0
    cache_read_input_tokens_per_hour: float | Unset = 0.0
    cache_creation_input_tokens_per_hour: float | Unset = 0.0
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        window_seconds = self.window_seconds

        observed_seconds = self.observed_seconds

        complete = self.complete

        cost_microusd_per_hour = self.cost_microusd_per_hour

        input_tokens_per_hour = self.input_tokens_per_hour

        output_tokens_per_hour = self.output_tokens_per_hour

        cache_read_input_tokens_per_hour = self.cache_read_input_tokens_per_hour

        cache_creation_input_tokens_per_hour = self.cache_creation_input_tokens_per_hour

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "window_seconds": window_seconds,
                "observed_seconds": observed_seconds,
                "complete": complete,
            }
        )
        if cost_microusd_per_hour is not UNSET:
            field_dict["cost_microusd_per_hour"] = cost_microusd_per_hour
        if input_tokens_per_hour is not UNSET:
            field_dict["input_tokens_per_hour"] = input_tokens_per_hour
        if output_tokens_per_hour is not UNSET:
            field_dict["output_tokens_per_hour"] = output_tokens_per_hour
        if cache_read_input_tokens_per_hour is not UNSET:
            field_dict["cache_read_input_tokens_per_hour"] = (
                cache_read_input_tokens_per_hour
            )
        if cache_creation_input_tokens_per_hour is not UNSET:
            field_dict["cache_creation_input_tokens_per_hour"] = (
                cache_creation_input_tokens_per_hour
            )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        window_seconds = d.pop("window_seconds")

        observed_seconds = d.pop("observed_seconds")

        complete = d.pop("complete")

        cost_microusd_per_hour = d.pop("cost_microusd_per_hour", UNSET)

        input_tokens_per_hour = d.pop("input_tokens_per_hour", UNSET)

        output_tokens_per_hour = d.pop("output_tokens_per_hour", UNSET)

        cache_read_input_tokens_per_hour = d.pop(
            "cache_read_input_tokens_per_hour", UNSET
        )

        cache_creation_input_tokens_per_hour = d.pop(
            "cache_creation_input_tokens_per_hour", UNSET
        )

        usage_rate = cls(
            window_seconds=window_seconds,
            observed_seconds=observed_seconds,
            complete=complete,
            cost_microusd_per_hour=cost_microusd_per_hour,
            input_tokens_per_hour=input_tokens_per_hour,
            output_tokens_per_hour=output_tokens_per_hour,
            cache_read_input_tokens_per_hour=cache_read_input_tokens_per_hour,
            cache_creation_input_tokens_per_hour=cache_creation_input_tokens_per_hour,
        )

        usage_rate.additional_properties = d
        return usage_rate

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
