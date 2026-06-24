from __future__ import annotations

from collections.abc import Mapping
from typing import (
    Any,
    Literal,
    TypeVar,
    cast,
)

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="AllowList")


@_attrs_define
class AllowList:
    """Admit only the enumerated ``chat_ids``.

    ``chat_ids`` is required and ``min_length=1`` — an empty list is a
    validation error (422), never a silent deny-all. (Use ``DenyAll`` to
    deny everyone explicitly.)

        Attributes:
            chat_ids (list[str]):
            kind (Literal['allow_list'] | Unset):  Default: 'allow_list'.
    """

    chat_ids: list[str]
    kind: Literal["allow_list"] | Unset = "allow_list"

    def to_dict(self) -> dict[str, Any]:
        chat_ids = self.chat_ids

        kind = self.kind

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "chat_ids": chat_ids,
            }
        )
        if kind is not UNSET:
            field_dict["kind"] = kind

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        chat_ids = cast(list[str], d.pop("chat_ids"))

        kind = cast(Literal["allow_list"] | Unset, d.pop("kind", UNSET))
        if kind != "allow_list" and not isinstance(kind, Unset):
            raise ValueError(f"kind must match const 'allow_list', got '{kind}'")

        allow_list = cls(
            chat_ids=chat_ids,
            kind=kind,
        )

        return allow_list
