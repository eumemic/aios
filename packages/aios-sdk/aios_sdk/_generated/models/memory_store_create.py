from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.memory_store_create_metadata import MemoryStoreCreateMetadata


T = TypeVar("T", bound="MemoryStoreCreate")


@_attrs_define
class MemoryStoreCreate:
    """Request body for ``POST /v1/memory-stores``.

    Attributes:
        name (str):
        description (str | Unset):  Default: ''.
        metadata (MemoryStoreCreateMetadata | Unset):
    """

    name: str
    description: str | Unset = ""
    metadata: MemoryStoreCreateMetadata | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        description = self.description

        metadata: dict[str, Any] | Unset = UNSET
        if not isinstance(self.metadata, Unset):
            metadata = self.metadata.to_dict()

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "name": name,
            }
        )
        if description is not UNSET:
            field_dict["description"] = description
        if metadata is not UNSET:
            field_dict["metadata"] = metadata

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.memory_store_create_metadata import MemoryStoreCreateMetadata

        d = dict(src_dict)
        name = d.pop("name")

        description = d.pop("description", UNSET)

        _metadata = d.pop("metadata", UNSET)
        metadata: MemoryStoreCreateMetadata | Unset
        if isinstance(_metadata, Unset):
            metadata = UNSET
        else:
            metadata = MemoryStoreCreateMetadata.from_dict(_metadata)

        memory_store_create = cls(
            name=name,
            description=description,
            metadata=metadata,
        )

        return memory_store_create
