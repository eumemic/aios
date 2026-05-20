from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="RuntimeToken")


@_attrs_define
class RuntimeToken:
    """Read view of a runtime token.  Never carries plaintext.

    ``connection_ids`` is the optional allowlist scope (#350).  ``None``
    means the token is unscoped — it sees every connection of its
    ``connector`` type.  A non-``None`` list (including ``[]``) limits
    visibility / operations to the listed IDs only.

        Attributes:
            id (str):
            connector (str):
            created_at (datetime.datetime):
            label (None | str | Unset):
            connection_ids (list[str] | None | Unset):
            last_used_at (datetime.datetime | None | Unset):
            revoked_at (datetime.datetime | None | Unset):
    """

    id: str
    connector: str
    created_at: datetime.datetime
    label: None | str | Unset = UNSET
    connection_ids: list[str] | None | Unset = UNSET
    last_used_at: datetime.datetime | None | Unset = UNSET
    revoked_at: datetime.datetime | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        connector = self.connector

        created_at = self.created_at.isoformat()

        label: None | str | Unset
        if isinstance(self.label, Unset):
            label = UNSET
        else:
            label = self.label

        connection_ids: list[str] | None | Unset
        if isinstance(self.connection_ids, Unset):
            connection_ids = UNSET
        elif isinstance(self.connection_ids, list):
            connection_ids = self.connection_ids

        else:
            connection_ids = self.connection_ids

        last_used_at: None | str | Unset
        if isinstance(self.last_used_at, Unset):
            last_used_at = UNSET
        elif isinstance(self.last_used_at, datetime.datetime):
            last_used_at = self.last_used_at.isoformat()
        else:
            last_used_at = self.last_used_at

        revoked_at: None | str | Unset
        if isinstance(self.revoked_at, Unset):
            revoked_at = UNSET
        elif isinstance(self.revoked_at, datetime.datetime):
            revoked_at = self.revoked_at.isoformat()
        else:
            revoked_at = self.revoked_at

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "connector": connector,
                "created_at": created_at,
            }
        )
        if label is not UNSET:
            field_dict["label"] = label
        if connection_ids is not UNSET:
            field_dict["connection_ids"] = connection_ids
        if last_used_at is not UNSET:
            field_dict["last_used_at"] = last_used_at
        if revoked_at is not UNSET:
            field_dict["revoked_at"] = revoked_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        id = d.pop("id")

        connector = d.pop("connector")

        created_at = isoparse(d.pop("created_at"))

        def _parse_label(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        label = _parse_label(d.pop("label", UNSET))

        def _parse_connection_ids(data: object) -> list[str] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                connection_ids_type_0 = cast(list[str], data)

                return connection_ids_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None | Unset, data)

        connection_ids = _parse_connection_ids(d.pop("connection_ids", UNSET))

        def _parse_last_used_at(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                last_used_at_type_0 = isoparse(data)

                return last_used_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        last_used_at = _parse_last_used_at(d.pop("last_used_at", UNSET))

        def _parse_revoked_at(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                revoked_at_type_0 = isoparse(data)

                return revoked_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        revoked_at = _parse_revoked_at(d.pop("revoked_at", UNSET))

        runtime_token = cls(
            id=id,
            connector=connector,
            created_at=created_at,
            label=label,
            connection_ids=connection_ids,
            last_used_at=last_used_at,
            revoked_at=revoked_at,
        )

        runtime_token.additional_properties = d
        return runtime_token

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
