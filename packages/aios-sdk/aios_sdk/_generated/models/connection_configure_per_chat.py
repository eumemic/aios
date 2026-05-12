from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

T = TypeVar("T", bound="ConnectionConfigurePerChat")


@_attrs_define
class ConnectionConfigurePerChat:
    """Request body for ``POST /v1/connections/{id}/configure-per-chat``.

    Attributes:
        session_template_id (str):
    """

    session_template_id: str

    def to_dict(self) -> dict[str, Any]:
        session_template_id = self.session_template_id

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "session_template_id": session_template_id,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        session_template_id = d.pop("session_template_id")

        connection_configure_per_chat = cls(
            session_template_id=session_template_id,
        )

        return connection_configure_per_chat
