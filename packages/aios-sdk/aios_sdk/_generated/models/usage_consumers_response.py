from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.usage_consumers_response_metric import UsageConsumersResponseMetric
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.usage_consumer import UsageConsumer


T = TypeVar("T", bound="UsageConsumersResponse")


@_attrs_define
class UsageConsumersResponse:
    """Ranked, additive root consumers for one account.

    Attributes:
        metric (UsageConsumersResponseMetric):
        window_seconds (int):
        coverage_started_at (datetime.datetime):
        total_rate_per_hour (float):
        items (list[UsageConsumer] | Unset):
    """

    metric: UsageConsumersResponseMetric
    window_seconds: int
    coverage_started_at: datetime.datetime
    total_rate_per_hour: float
    items: list[UsageConsumer] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        metric = self.metric.value

        window_seconds = self.window_seconds

        coverage_started_at = self.coverage_started_at.isoformat()

        total_rate_per_hour = self.total_rate_per_hour

        items: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.items, Unset):
            items = []
            for items_item_data in self.items:
                items_item = items_item_data.to_dict()
                items.append(items_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "metric": metric,
                "window_seconds": window_seconds,
                "coverage_started_at": coverage_started_at,
                "total_rate_per_hour": total_rate_per_hour,
            }
        )
        if items is not UNSET:
            field_dict["items"] = items

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.usage_consumer import UsageConsumer

        d = dict(src_dict)
        metric = UsageConsumersResponseMetric(d.pop("metric"))

        window_seconds = d.pop("window_seconds")

        coverage_started_at = isoparse(d.pop("coverage_started_at"))

        total_rate_per_hour = d.pop("total_rate_per_hour")

        _items = d.pop("items", UNSET)
        items: list[UsageConsumer] | Unset = UNSET
        if _items is not UNSET:
            items = []
            for items_item_data in _items:
                items_item = UsageConsumer.from_dict(items_item_data)

                items.append(items_item)

        usage_consumers_response = cls(
            metric=metric,
            window_seconds=window_seconds,
            coverage_started_at=coverage_started_at,
            total_rate_per_hour=total_rate_per_hour,
            items=items,
        )

        usage_consumers_response.additional_properties = d
        return usage_consumers_response

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
