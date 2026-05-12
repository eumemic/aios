from __future__ import annotations

from collections.abc import Mapping
from typing import (
    Any,
    Literal,
    TypeVar,
    cast,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.memory_store_resource_echo_access import MemoryStoreResourceEchoAccess
from ..types import UNSET, Unset

T = TypeVar("T", bound="MemoryStoreResourceEcho")


@_attrs_define
class MemoryStoreResourceEcho:
    """Read view of an attached memory store as echoed on ``Session.resources``.

    Carries the snapshotted ``name`` / ``description`` from the store at
    attach time, plus the derived ``mount_path``. These do not update if the
    underlying store is renamed or its description changes.

        Attributes:
            memory_store_id (str):
            access (MemoryStoreResourceEchoAccess):
            instructions (str):
            name (str):
            description (str):
            mount_path (str):
            type_ (Literal['memory_store'] | Unset):  Default: 'memory_store'.
    """

    memory_store_id: str
    access: MemoryStoreResourceEchoAccess
    instructions: str
    name: str
    description: str
    mount_path: str
    type_: Literal["memory_store"] | Unset = "memory_store"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        memory_store_id = self.memory_store_id

        access = self.access.value

        instructions = self.instructions

        name = self.name

        description = self.description

        mount_path = self.mount_path

        type_ = self.type_

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "memory_store_id": memory_store_id,
                "access": access,
                "instructions": instructions,
                "name": name,
                "description": description,
                "mount_path": mount_path,
            }
        )
        if type_ is not UNSET:
            field_dict["type"] = type_

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        memory_store_id = d.pop("memory_store_id")

        access = MemoryStoreResourceEchoAccess(d.pop("access"))

        instructions = d.pop("instructions")

        name = d.pop("name")

        description = d.pop("description")

        mount_path = d.pop("mount_path")

        type_ = cast(Literal["memory_store"] | Unset, d.pop("type", UNSET))
        if type_ != "memory_store" and not isinstance(type_, Unset):
            raise ValueError(f"type must match const 'memory_store', got '{type_}'")

        memory_store_resource_echo = cls(
            memory_store_id=memory_store_id,
            access=access,
            instructions=instructions,
            name=name,
            description=description,
            mount_path=mount_path,
            type_=type_,
        )

        memory_store_resource_echo.additional_properties = d
        return memory_store_resource_echo

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
