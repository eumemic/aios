from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
    cast,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.memory_store_metadata import MemoryStoreMetadata


T = TypeVar("T", bound="MemoryStore")


@_attrs_define
class MemoryStore:
    """Read view of a memory store.

    Attributes:
        id (str):
        name (str):
        description (str):
        metadata (MemoryStoreMetadata):
        created_at (datetime.datetime):
        updated_at (datetime.datetime):
        type_ (Literal['memory_store'] | Unset):  Default: 'memory_store'.
        archived_at (datetime.datetime | None | Unset):
    """

    id: str
    name: str
    description: str
    metadata: MemoryStoreMetadata
    created_at: datetime.datetime
    updated_at: datetime.datetime
    type_: Literal["memory_store"] | Unset = "memory_store"
    archived_at: datetime.datetime | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        name = self.name

        description = self.description

        metadata = self.metadata.to_dict()

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        type_ = self.type_

        archived_at: None | str | Unset
        if isinstance(self.archived_at, Unset):
            archived_at = UNSET
        elif isinstance(self.archived_at, datetime.datetime):
            archived_at = self.archived_at.isoformat()
        else:
            archived_at = self.archived_at

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "name": name,
                "description": description,
                "metadata": metadata,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        if type_ is not UNSET:
            field_dict["type"] = type_
        if archived_at is not UNSET:
            field_dict["archived_at"] = archived_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.memory_store_metadata import MemoryStoreMetadata

        d = dict(src_dict)
        id = d.pop("id")

        name = d.pop("name")

        description = d.pop("description")

        metadata = MemoryStoreMetadata.from_dict(d.pop("metadata"))

        created_at = isoparse(d.pop("created_at"))

        updated_at = isoparse(d.pop("updated_at"))

        type_ = cast(Literal["memory_store"] | Unset, d.pop("type", UNSET))
        if type_ != "memory_store" and not isinstance(type_, Unset):
            raise ValueError(f"type must match const 'memory_store', got '{type_}'")

        def _parse_archived_at(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                archived_at_type_0 = isoparse(data)

                return archived_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        archived_at = _parse_archived_at(d.pop("archived_at", UNSET))

        memory_store = cls(
            id=id,
            name=name,
            description=description,
            metadata=metadata,
            created_at=created_at,
            updated_at=updated_at,
            type_=type_,
            archived_at=archived_at,
        )

        memory_store.additional_properties = d
        return memory_store

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
