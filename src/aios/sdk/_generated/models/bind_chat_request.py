from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

T = TypeVar("T", bound="BindChatRequest")


@_attrs_define
class BindChatRequest:
    """Request body for ``POST /v1/connections/{id}/bind-chat``.

    Pre-populates a ``connection_chat_sessions`` row so inbound on
    ``chat_id`` routes to ``session_id`` regardless of the connection's
    mode-default fallback (#215).  Operators use this to point
    different chats on a single account at different operator-curated
    existing sessions.

        Attributes:
            chat_id (str):
            session_id (str):
    """

    chat_id: str
    session_id: str

    def to_dict(self) -> dict[str, Any]:
        chat_id = self.chat_id

        session_id = self.session_id

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "chat_id": chat_id,
                "session_id": session_id,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        chat_id = d.pop("chat_id")

        session_id = d.pop("session_id")

        bind_chat_request = cls(
            chat_id=chat_id,
            session_id=session_id,
        )

        return bind_chat_request
