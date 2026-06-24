from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

T = TypeVar("T", bound="RecentChat")


@_attrs_define
class RecentChat:
    """Distinct chat_id observed on a connection's account, with the
    most-recent inbound timestamp.

    Returned by ``GET /v1/connections/{id}/recent-chats`` so operators
    can find the chat_id for a specific peer without digging through
    event logs before calling ``bind-chat`` or opening an ``AllowList``.

    ``chat_id`` is the **authoritative, gate-relevant identifier** — it is
    exactly what the inbound-admission gate (``AllowList.chat_ids``) matches
    on, so it is the value an operator copies into an allowlist. This view
    deliberately does not present a connector-supplied ``display_name`` /
    ``sender_name`` alongside it: that name is self-reported by the
    connector and forgeable, never a verified identity.

        Attributes:
            chat_id (str):
            last_seen_at (datetime.datetime):
    """

    chat_id: str
    last_seen_at: datetime.datetime
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        chat_id = self.chat_id

        last_seen_at = self.last_seen_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "chat_id": chat_id,
                "last_seen_at": last_seen_at,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        chat_id = d.pop("chat_id")

        last_seen_at = isoparse(d.pop("last_seen_at"))

        recent_chat = cls(
            chat_id=chat_id,
            last_seen_at=last_seen_at,
        )

        recent_chat.additional_properties = d
        return recent_chat

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
