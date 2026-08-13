from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="MonthlyStorageBucket")


@_attrs_define
class MonthlyStorageBucket:
    """
    Attributes:
        table (str):
        month (str):
        row_estimate (int):
        approx_bytes (int):
    """

    table: str
    month: str
    row_estimate: int
    approx_bytes: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        table = self.table

        month = self.month

        row_estimate = self.row_estimate

        approx_bytes = self.approx_bytes

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "table": table,
                "month": month,
                "row_estimate": row_estimate,
                "approx_bytes": approx_bytes,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        table = d.pop("table")

        month = d.pop("month")

        row_estimate = d.pop("row_estimate")

        approx_bytes = d.pop("approx_bytes")

        monthly_storage_bucket = cls(
            table=table,
            month=month,
            row_estimate=row_estimate,
            approx_bytes=approx_bytes,
        )

        monthly_storage_bucket.additional_properties = d
        return monthly_storage_bucket

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
