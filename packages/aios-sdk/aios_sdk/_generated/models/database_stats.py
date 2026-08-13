from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.monthly_storage_bucket import MonthlyStorageBucket
    from ..models.table_storage_stats import TableStorageStats


T = TypeVar("T", bound="DatabaseStats")


@_attrs_define
class DatabaseStats:
    """
    Attributes:
        generated_at (datetime.datetime):
        database_bytes (int):
        tables (list[TableStorageStats] | Unset):
        buckets (list[MonthlyStorageBucket] | Unset):
    """

    generated_at: datetime.datetime
    database_bytes: int
    tables: list[TableStorageStats] | Unset = UNSET
    buckets: list[MonthlyStorageBucket] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        generated_at = self.generated_at.isoformat()

        database_bytes = self.database_bytes

        tables: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.tables, Unset):
            tables = []
            for tables_item_data in self.tables:
                tables_item = tables_item_data.to_dict()
                tables.append(tables_item)

        buckets: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.buckets, Unset):
            buckets = []
            for buckets_item_data in self.buckets:
                buckets_item = buckets_item_data.to_dict()
                buckets.append(buckets_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "generated_at": generated_at,
                "database_bytes": database_bytes,
            }
        )
        if tables is not UNSET:
            field_dict["tables"] = tables
        if buckets is not UNSET:
            field_dict["buckets"] = buckets

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.monthly_storage_bucket import MonthlyStorageBucket
        from ..models.table_storage_stats import TableStorageStats

        d = dict(src_dict)
        generated_at = isoparse(d.pop("generated_at"))

        database_bytes = d.pop("database_bytes")

        _tables = d.pop("tables", UNSET)
        tables: list[TableStorageStats] | Unset = UNSET
        if _tables is not UNSET:
            tables = []
            for tables_item_data in _tables:
                tables_item = TableStorageStats.from_dict(tables_item_data)

                tables.append(tables_item)

        _buckets = d.pop("buckets", UNSET)
        buckets: list[MonthlyStorageBucket] | Unset = UNSET
        if _buckets is not UNSET:
            buckets = []
            for buckets_item_data in _buckets:
                buckets_item = MonthlyStorageBucket.from_dict(buckets_item_data)

                buckets.append(buckets_item)

        database_stats = cls(
            generated_at=generated_at,
            database_bytes=database_bytes,
            tables=tables,
            buckets=buckets,
        )

        database_stats.additional_properties = d
        return database_stats

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
