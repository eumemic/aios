from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

if TYPE_CHECKING:
    from ..models.session_egress_host import SessionEgressHost


T = TypeVar("T", bound="SessionEgressResponse")


@_attrs_define
class SessionEgressResponse:
    """Worker-observed egress state from the most recent live provisioning.

    Attributes:
        hosts (list[SessionEgressHost]):
        provisioned_at (datetime.datetime):
        sandbox_generation (int):
    """

    hosts: list[SessionEgressHost]
    provisioned_at: datetime.datetime
    sandbox_generation: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        hosts = []
        for hosts_item_data in self.hosts:
            hosts_item = hosts_item_data.to_dict()
            hosts.append(hosts_item)

        provisioned_at = self.provisioned_at.isoformat()

        sandbox_generation = self.sandbox_generation

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "hosts": hosts,
                "provisioned_at": provisioned_at,
                "sandbox_generation": sandbox_generation,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.session_egress_host import SessionEgressHost

        d = dict(src_dict)
        hosts = []
        _hosts = d.pop("hosts")
        for hosts_item_data in _hosts:
            hosts_item = SessionEgressHost.from_dict(hosts_item_data)

            hosts.append(hosts_item)

        provisioned_at = isoparse(d.pop("provisioned_at"))

        sandbox_generation = d.pop("sandbox_generation")

        session_egress_response = cls(
            hosts=hosts,
            provisioned_at=provisioned_at,
            sandbox_generation=sandbox_generation,
        )

        session_egress_response.additional_properties = d
        return session_egress_response

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
