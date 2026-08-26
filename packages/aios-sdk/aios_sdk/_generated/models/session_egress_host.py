from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="SessionEgressHost")


@_attrs_define
class SessionEgressHost:
    """Metadata identifying one host in a sandbox's live intercept set.

    Attributes:
        host (str):
        intercepted (bool):
        source_credential_id (str):
        secret_name (str):
    """

    host: str
    intercepted: bool
    source_credential_id: str
    secret_name: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        host = self.host

        intercepted = self.intercepted

        source_credential_id = self.source_credential_id

        secret_name = self.secret_name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "host": host,
                "intercepted": intercepted,
                "source_credential_id": source_credential_id,
                "secret_name": secret_name,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        host = d.pop("host")

        intercepted = d.pop("intercepted")

        source_credential_id = d.pop("source_credential_id")

        secret_name = d.pop("secret_name")

        session_egress_host = cls(
            host=host,
            intercepted=intercepted,
            source_credential_id=source_credential_id,
            secret_name=secret_name,
        )

        session_egress_host.additional_properties = d
        return session_egress_host

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
