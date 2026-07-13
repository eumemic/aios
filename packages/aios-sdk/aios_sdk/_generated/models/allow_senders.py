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

T = TypeVar("T", bound="AllowSenders")


@_attrs_define
class AllowSenders:
    """Admit only canonical connector-supplied sender identifiers.

    Attributes:
        sender_ids (list[str]):
        kind (Literal['allow_senders'] | Unset):  Default: 'allow_senders'.
    """

    sender_ids: list[str]
    kind: Literal["allow_senders"] | Unset = "allow_senders"

    def to_dict(self) -> dict[str, Any]:
        sender_ids = self.sender_ids

        kind = self.kind

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "sender_ids": sender_ids,
            }
        )
        if kind is not UNSET:
            field_dict["kind"] = kind

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        sender_ids = cast(list[str], d.pop("sender_ids"))

        kind = cast(Literal["allow_senders"] | Unset, d.pop("kind", UNSET))
        if kind != "allow_senders" and not isinstance(kind, Unset):
            raise ValueError(f"kind must match const 'allow_senders', got '{kind}'")

        allow_senders = cls(
            sender_ids=sender_ids,
            kind=kind,
        )

        return allow_senders
