from __future__ import annotations

from collections.abc import Mapping
from typing import (
    Any,
    Literal,
    TypeVar,
    cast,
)

from attrs import define as _attrs_define

from ..models.memory_store_resource_access import MemoryStoreResourceAccess
from ..types import UNSET, Unset

T = TypeVar("T", bound="MemoryStoreResource")


@_attrs_define
class MemoryStoreResource:
    """Item in ``Session.resources[]`` request shape.

    Only ``memory_store`` for now; the discriminator field keeps the door
    open for future ``file`` / ``repo`` resource types.

        Attributes:
            type_ (Literal['memory_store']):
            memory_store_id (str):
            access (MemoryStoreResourceAccess | Unset):  Default: MemoryStoreResourceAccess.READ_WRITE.
            instructions (str | Unset):  Default: ''.
    """

    type_: Literal["memory_store"]
    memory_store_id: str
    access: MemoryStoreResourceAccess | Unset = MemoryStoreResourceAccess.READ_WRITE
    instructions: str | Unset = ""

    def to_dict(self) -> dict[str, Any]:
        type_ = self.type_

        memory_store_id = self.memory_store_id

        access: str | Unset = UNSET
        if not isinstance(self.access, Unset):
            access = self.access.value

        instructions = self.instructions

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "type": type_,
                "memory_store_id": memory_store_id,
            }
        )
        if access is not UNSET:
            field_dict["access"] = access
        if instructions is not UNSET:
            field_dict["instructions"] = instructions

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        type_ = cast(Literal["memory_store"], d.pop("type"))
        if type_ != "memory_store":
            raise ValueError(f"type must match const 'memory_store', got '{type_}'")

        memory_store_id = d.pop("memory_store_id")

        _access = d.pop("access", UNSET)
        access: MemoryStoreResourceAccess | Unset
        if isinstance(_access, Unset):
            access = UNSET
        else:
            access = MemoryStoreResourceAccess(_access)

        instructions = d.pop("instructions", UNSET)

        memory_store_resource = cls(
            type_=type_,
            memory_store_id=memory_store_id,
            access=access,
            instructions=instructions,
        )

        return memory_store_resource
