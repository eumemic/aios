from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

T = TypeVar("T", bound="RuntimeTokenIssued")


@_attrs_define
class RuntimeTokenIssued:
    """Response body for ``POST /v1/runtime-tokens``.

    Includes the plaintext token — this is the ONLY time it's surfaced.
    Subsequent ``GET`` returns the read view without plaintext.

        Attributes:
            id (str):
            connector (str):
            label (None | str):
            plaintext (str): The bearer token value.  Save this — it cannot be recovered.
            created_at (datetime.datetime):
    """

    id: str
    connector: str
    label: None | str
    plaintext: str
    created_at: datetime.datetime
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        connector = self.connector

        label: None | str
        label = self.label

        plaintext = self.plaintext

        created_at = self.created_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "connector": connector,
                "label": label,
                "plaintext": plaintext,
                "created_at": created_at,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        id = d.pop("id")

        connector = d.pop("connector")

        def _parse_label(data: object) -> None | str:
            if data is None:
                return data
            return cast(None | str, data)

        label = _parse_label(d.pop("label"))

        plaintext = d.pop("plaintext")

        created_at = isoparse(d.pop("created_at"))

        runtime_token_issued = cls(
            id=id,
            connector=connector,
            label=label,
            plaintext=plaintext,
            created_at=created_at,
        )

        runtime_token_issued.additional_properties = d
        return runtime_token_issued

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
