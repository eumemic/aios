from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="TableStorageStats")


@_attrs_define
class TableStorageStats:
    """
    Attributes:
        name (str):
        total_bytes (int):
        heap_bytes (int):
        index_bytes (int):
        toast_bytes (int):
        row_estimate (int):
        dead_tuple_estimate (int):
    """

    name: str
    total_bytes: int
    heap_bytes: int
    index_bytes: int
    toast_bytes: int
    row_estimate: int
    dead_tuple_estimate: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        total_bytes = self.total_bytes

        heap_bytes = self.heap_bytes

        index_bytes = self.index_bytes

        toast_bytes = self.toast_bytes

        row_estimate = self.row_estimate

        dead_tuple_estimate = self.dead_tuple_estimate

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "total_bytes": total_bytes,
                "heap_bytes": heap_bytes,
                "index_bytes": index_bytes,
                "toast_bytes": toast_bytes,
                "row_estimate": row_estimate,
                "dead_tuple_estimate": dead_tuple_estimate,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        name = d.pop("name")

        total_bytes = d.pop("total_bytes")

        heap_bytes = d.pop("heap_bytes")

        index_bytes = d.pop("index_bytes")

        toast_bytes = d.pop("toast_bytes")

        row_estimate = d.pop("row_estimate")

        dead_tuple_estimate = d.pop("dead_tuple_estimate")

        table_storage_stats = cls(
            name=name,
            total_bytes=total_bytes,
            heap_bytes=heap_bytes,
            index_bytes=index_bytes,
            toast_bytes=toast_bytes,
            row_estimate=row_estimate,
            dead_tuple_estimate=dead_tuple_estimate,
        )

        table_storage_stats.additional_properties = d
        return table_storage_stats

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
