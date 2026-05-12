from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

T = TypeVar("T", bound="MemoryCreate")


@_attrs_define
class MemoryCreate:
    """Request body for ``POST /v1/memory-stores/{store_id}/memories``.

    Attributes:
        path (str): Absolute, slash-separated path. Segments may not contain / or NUL, and `.` and `..` are not allowed
            as segments.
        content (str):
    """

    path: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        path = self.path

        content = self.content

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "path": path,
                "content": content,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        path = d.pop("path")

        content = d.pop("content")

        memory_create = cls(
            path=path,
            content=content,
        )

        return memory_create
