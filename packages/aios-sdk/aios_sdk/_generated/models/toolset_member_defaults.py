from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..models.toolset_member_defaults_permission_type_0 import (
    ToolsetMemberDefaultsPermissionType0,
)
from ..models.toolset_member_defaults_transport_type_0 import (
    ToolsetMemberDefaultsTransportType0,
)
from ..types import UNSET, Unset

T = TypeVar("T", bound="ToolsetMemberDefaults")


@_attrs_define
class ToolsetMemberDefaults:
    """``default_config`` for a curated toolset: applied to every member.

    ``enabled=False`` gives the upstream "start all off, then opt in" shape;
    opt members back in with ``configs[]`` entries.

        Attributes:
            enabled (bool | Unset):  Default: True.
            permission (None | ToolsetMemberDefaultsPermissionType0 | Unset):
            transport (None | ToolsetMemberDefaultsTransportType0 | Unset):
    """

    enabled: bool | Unset = True
    permission: None | ToolsetMemberDefaultsPermissionType0 | Unset = UNSET
    transport: None | ToolsetMemberDefaultsTransportType0 | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        enabled = self.enabled

        permission: None | str | Unset
        if isinstance(self.permission, Unset):
            permission = UNSET
        elif isinstance(self.permission, ToolsetMemberDefaultsPermissionType0):
            permission = self.permission.value
        else:
            permission = self.permission

        transport: None | str | Unset
        if isinstance(self.transport, Unset):
            transport = UNSET
        elif isinstance(self.transport, ToolsetMemberDefaultsTransportType0):
            transport = self.transport.value
        else:
            transport = self.transport

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if enabled is not UNSET:
            field_dict["enabled"] = enabled
        if permission is not UNSET:
            field_dict["permission"] = permission
        if transport is not UNSET:
            field_dict["transport"] = transport

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        enabled = d.pop("enabled", UNSET)

        def _parse_permission(
            data: object,
        ) -> None | ToolsetMemberDefaultsPermissionType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                permission_type_0 = ToolsetMemberDefaultsPermissionType0(data)

                return permission_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | ToolsetMemberDefaultsPermissionType0 | Unset, data)

        permission = _parse_permission(d.pop("permission", UNSET))

        def _parse_transport(
            data: object,
        ) -> None | ToolsetMemberDefaultsTransportType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                transport_type_0 = ToolsetMemberDefaultsTransportType0(data)

                return transport_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | ToolsetMemberDefaultsTransportType0 | Unset, data)

        transport = _parse_transport(d.pop("transport", UNSET))

        toolset_member_defaults = cls(
            enabled=enabled,
            permission=permission,
            transport=transport,
        )

        return toolset_member_defaults
