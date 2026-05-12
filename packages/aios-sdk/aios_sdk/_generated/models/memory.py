from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import (
    Any,
    Literal,
    TypeVar,
    cast,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="Memory")


@_attrs_define
class Memory:
    """Read view of a memory. ``content`` only on retrieve.

    Attributes:
        id (str):
        memory_store_id (str):
        memory_version_id (str):
        path (str):
        content_sha256 (str):
        content_size_bytes (int):
        created_at (datetime.datetime):
        updated_at (datetime.datetime):
        type_ (Literal['memory'] | Unset):  Default: 'memory'.
        content (None | str | Unset):
    """

    id: str
    memory_store_id: str
    memory_version_id: str
    path: str
    content_sha256: str
    content_size_bytes: int
    created_at: datetime.datetime
    updated_at: datetime.datetime
    type_: Literal["memory"] | Unset = "memory"
    content: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        memory_store_id = self.memory_store_id

        memory_version_id = self.memory_version_id

        path = self.path

        content_sha256 = self.content_sha256

        content_size_bytes = self.content_size_bytes

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        type_ = self.type_

        content: None | str | Unset
        if isinstance(self.content, Unset):
            content = UNSET
        else:
            content = self.content

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "memory_store_id": memory_store_id,
                "memory_version_id": memory_version_id,
                "path": path,
                "content_sha256": content_sha256,
                "content_size_bytes": content_size_bytes,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        if type_ is not UNSET:
            field_dict["type"] = type_
        if content is not UNSET:
            field_dict["content"] = content

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        id = d.pop("id")

        memory_store_id = d.pop("memory_store_id")

        memory_version_id = d.pop("memory_version_id")

        path = d.pop("path")

        content_sha256 = d.pop("content_sha256")

        content_size_bytes = d.pop("content_size_bytes")

        created_at = isoparse(d.pop("created_at"))

        updated_at = isoparse(d.pop("updated_at"))

        type_ = cast(Literal["memory"] | Unset, d.pop("type", UNSET))
        if type_ != "memory" and not isinstance(type_, Unset):
            raise ValueError(f"type must match const 'memory', got '{type_}'")

        def _parse_content(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        content = _parse_content(d.pop("content", UNSET))

        memory = cls(
            id=id,
            memory_store_id=memory_store_id,
            memory_version_id=memory_version_id,
            path=path,
            content_sha256=content_sha256,
            content_size_bytes=content_size_bytes,
            created_at=created_at,
            updated_at=updated_at,
            type_=type_,
            content=content,
        )

        memory.additional_properties = d
        return memory

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
