from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.connection_create_metadata import ConnectionCreateMetadata


T = TypeVar("T", bound="ConnectionCreate")


@_attrs_define
class ConnectionCreate:
    """Request body for ``POST /v1/connections``.

    Created in detached mode — neither ``session_id`` nor
    ``session_template_id`` is set.  Use ``POST .../attach`` or
    ``POST .../configure-per-chat`` afterward to bind a routing mode.

    ``connector`` and ``account`` may not contain ``/`` — they're used
    in the focal-channel address scheme ``{connector}/{account}/{chat_id}``
    and a ``/`` would create ambiguous segment boundaries.

        Attributes:
            connector (str):
            account (str):
            metadata (ConnectionCreateMetadata | Unset):
    """

    connector: str
    account: str
    metadata: ConnectionCreateMetadata | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        connector = self.connector

        account = self.account

        metadata: dict[str, Any] | Unset = UNSET
        if not isinstance(self.metadata, Unset):
            metadata = self.metadata.to_dict()

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "connector": connector,
                "account": account,
            }
        )
        if metadata is not UNSET:
            field_dict["metadata"] = metadata

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.connection_create_metadata import ConnectionCreateMetadata

        d = dict(src_dict)
        connector = d.pop("connector")

        account = d.pop("account")

        _metadata = d.pop("metadata", UNSET)
        metadata: ConnectionCreateMetadata | Unset
        if isinstance(_metadata, Unset):
            metadata = UNSET
        else:
            metadata = ConnectionCreateMetadata.from_dict(_metadata)

        connection_create = cls(
            connector=connector,
            account=account,
            metadata=metadata,
        )

        return connection_create
