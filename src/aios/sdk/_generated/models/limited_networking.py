from __future__ import annotations

from collections.abc import Mapping
from typing import (
    Any,
    Literal,
    TypeVar,
    cast,
)

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="LimitedNetworking")


@_attrs_define
class LimitedNetworking:
    """Deny-all with domain allowlist.

    Outbound HTTP/HTTPS is restricted to ``allowed_hosts`` plus any hosts
    implied by the boolean flags.  DNS (port 53) remains open so tools
    like ``curl`` can resolve names.

        Attributes:
            type_ (Literal['limited']):
            allowed_hosts (list[str] | Unset):
            allow_package_managers (bool | Unset):  Default: False.
            allow_mcp_servers (bool | Unset):  Default: False.
    """

    type_: Literal["limited"]
    allowed_hosts: list[str] | Unset = UNSET
    allow_package_managers: bool | Unset = False
    allow_mcp_servers: bool | Unset = False

    def to_dict(self) -> dict[str, Any]:
        type_ = self.type_

        allowed_hosts: list[str] | Unset = UNSET
        if not isinstance(self.allowed_hosts, Unset):
            allowed_hosts = self.allowed_hosts

        allow_package_managers = self.allow_package_managers

        allow_mcp_servers = self.allow_mcp_servers

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "type": type_,
            }
        )
        if allowed_hosts is not UNSET:
            field_dict["allowed_hosts"] = allowed_hosts
        if allow_package_managers is not UNSET:
            field_dict["allow_package_managers"] = allow_package_managers
        if allow_mcp_servers is not UNSET:
            field_dict["allow_mcp_servers"] = allow_mcp_servers

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        type_ = cast(Literal["limited"], d.pop("type"))
        if type_ != "limited":
            raise ValueError(f"type must match const 'limited', got '{type_}'")

        allowed_hosts = cast(list[str], d.pop("allowed_hosts", UNSET))

        allow_package_managers = d.pop("allow_package_managers", UNSET)

        allow_mcp_servers = d.pop("allow_mcp_servers", UNSET)

        limited_networking = cls(
            type_=type_,
            allowed_hosts=allowed_hosts,
            allow_package_managers=allow_package_managers,
            allow_mcp_servers=allow_mcp_servers,
        )

        return limited_networking
