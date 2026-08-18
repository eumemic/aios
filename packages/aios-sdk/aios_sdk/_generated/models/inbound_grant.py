from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.inbound_grant_status import InboundGrantStatus
from ..types import UNSET, Unset

T = TypeVar("T", bound="InboundGrant")


@_attrs_define
class InboundGrant:
    """Audited inbound approval state.

    Attributes:
        id (str):
        account_id (str):
        connection_id (str):
        chat_id (str):
        status (InboundGrantStatus):
        created_at (datetime.datetime):
        updated_at (datetime.datetime):
        approved_by (None | str | Unset):
        approved_at (datetime.datetime | None | Unset):
        approved_via_channel (None | str | Unset):
    """

    id: str
    account_id: str
    connection_id: str
    chat_id: str
    status: InboundGrantStatus
    created_at: datetime.datetime
    updated_at: datetime.datetime
    approved_by: None | str | Unset = UNSET
    approved_at: datetime.datetime | None | Unset = UNSET
    approved_via_channel: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        account_id = self.account_id

        connection_id = self.connection_id

        chat_id = self.chat_id

        status = self.status.value

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        approved_by: None | str | Unset
        if isinstance(self.approved_by, Unset):
            approved_by = UNSET
        else:
            approved_by = self.approved_by

        approved_at: None | str | Unset
        if isinstance(self.approved_at, Unset):
            approved_at = UNSET
        elif isinstance(self.approved_at, datetime.datetime):
            approved_at = self.approved_at.isoformat()
        else:
            approved_at = self.approved_at

        approved_via_channel: None | str | Unset
        if isinstance(self.approved_via_channel, Unset):
            approved_via_channel = UNSET
        else:
            approved_via_channel = self.approved_via_channel

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "account_id": account_id,
                "connection_id": connection_id,
                "chat_id": chat_id,
                "status": status,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        if approved_by is not UNSET:
            field_dict["approved_by"] = approved_by
        if approved_at is not UNSET:
            field_dict["approved_at"] = approved_at
        if approved_via_channel is not UNSET:
            field_dict["approved_via_channel"] = approved_via_channel

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        id = d.pop("id")

        account_id = d.pop("account_id")

        connection_id = d.pop("connection_id")

        chat_id = d.pop("chat_id")

        status = InboundGrantStatus(d.pop("status"))

        created_at = isoparse(d.pop("created_at"))

        updated_at = isoparse(d.pop("updated_at"))

        def _parse_approved_by(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        approved_by = _parse_approved_by(d.pop("approved_by", UNSET))

        def _parse_approved_at(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                approved_at_type_0 = isoparse(data)

                return approved_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        approved_at = _parse_approved_at(d.pop("approved_at", UNSET))

        def _parse_approved_via_channel(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        approved_via_channel = _parse_approved_via_channel(
            d.pop("approved_via_channel", UNSET)
        )

        inbound_grant = cls(
            id=id,
            account_id=account_id,
            connection_id=connection_id,
            chat_id=chat_id,
            status=status,
            created_at=created_at,
            updated_at=updated_at,
            approved_by=approved_by,
            approved_at=approved_at,
            approved_via_channel=approved_via_channel,
        )

        inbound_grant.additional_properties = d
        return inbound_grant

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
