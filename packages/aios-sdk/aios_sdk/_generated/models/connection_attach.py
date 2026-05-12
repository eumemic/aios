from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

T = TypeVar("T", bound="ConnectionAttach")


@_attrs_define
class ConnectionAttach:
    """Request body for ``POST /v1/connections/{id}/attach``.

    Attributes:
        session_id (str):
    """

    session_id: str

    def to_dict(self) -> dict[str, Any]:
        session_id = self.session_id

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "session_id": session_id,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        session_id = d.pop("session_id")

        connection_attach = cls(
            session_id=session_id,
        )

        return connection_attach
