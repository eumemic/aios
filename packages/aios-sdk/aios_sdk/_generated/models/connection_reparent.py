from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

T = TypeVar("T", bound="ConnectionReparent")


@_attrs_define
class ConnectionReparent:
    """Request body for ``POST /v1/connections/{id}/reparent``.

    Moves the connection's ``account_id`` to ``destination_account_id``
    atomically, preserving ``connection.id`` so dependent connector
    state (signal-cli's ``account.dat``, whatsmeow's ``sqlstore.db``,
    telegram webhook config) carries over without recreation. v1
    auth: root operator only.

    Length bounds match the ULID-shaped ``account_id`` format used
    elsewhere on the wire (1..64 chars covers ``acc_<ULID>``).

        Attributes:
            destination_account_id (str):
    """

    destination_account_id: str

    def to_dict(self) -> dict[str, Any]:
        destination_account_id = self.destination_account_id

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "destination_account_id": destination_account_id,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        destination_account_id = d.pop("destination_account_id")

        connection_reparent = cls(
            destination_account_id=destination_account_id,
        )

        return connection_reparent
