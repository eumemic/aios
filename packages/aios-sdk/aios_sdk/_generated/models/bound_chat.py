from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

T = TypeVar("T", bound="BoundChat")


@_attrs_define
class BoundChat:
    """Read view of one ``connection_chat_sessions`` row.

    Returned by ``GET /v1/connections/{id}/bound-chats``.  Operator-bound
    rows and supervisor-spawned rows are returned together — the table
    doesn't tag the writer.

        Attributes:
            chat_id (str):
            session_id (str):
            created_at (datetime.datetime):
    """

    chat_id: str
    session_id: str
    created_at: datetime.datetime
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        chat_id = self.chat_id

        session_id = self.session_id

        created_at = self.created_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "chat_id": chat_id,
                "session_id": session_id,
                "created_at": created_at,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        chat_id = d.pop("chat_id")

        session_id = d.pop("session_id")

        created_at = isoparse(d.pop("created_at"))

        bound_chat = cls(
            chat_id=chat_id,
            session_id=session_id,
            created_at=created_at,
        )

        bound_chat.additional_properties = d
        return bound_chat

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
