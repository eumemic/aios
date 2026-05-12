from __future__ import annotations

from collections.abc import Mapping
from typing import (
    Any,
    Literal,
    TypeVar,
    cast,
)

from attrs import define as _attrs_define

T = TypeVar("T", bound="MemoryUpdatePrecondition")


@_attrs_define
class MemoryUpdatePrecondition:
    """``content_sha256`` precondition envelope.

    Attributes:
        type_ (Literal['content_sha256']):
        content_sha256 (str):
    """

    type_: Literal["content_sha256"]
    content_sha256: str

    def to_dict(self) -> dict[str, Any]:
        type_ = self.type_

        content_sha256 = self.content_sha256

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "type": type_,
                "content_sha256": content_sha256,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        type_ = cast(Literal["content_sha256"], d.pop("type"))
        if type_ != "content_sha256":
            raise ValueError(f"type must match const 'content_sha256', got '{type_}'")

        content_sha256 = d.pop("content_sha256")

        memory_update_precondition = cls(
            type_=type_,
            content_sha256=content_sha256,
        )

        return memory_update_precondition
