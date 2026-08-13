from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

T = TypeVar("T", bound="InboundGrantAction")


@_attrs_define
class InboundGrantAction:
    """Identify the canonical chat whose grant is being changed.

    Attributes:
        chat_id (str):
    """

    chat_id: str

    def to_dict(self) -> dict[str, Any]:
        chat_id = self.chat_id

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "chat_id": chat_id,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        chat_id = d.pop("chat_id")

        inbound_grant_action = cls(
            chat_id=chat_id,
        )

        return inbound_grant_action
