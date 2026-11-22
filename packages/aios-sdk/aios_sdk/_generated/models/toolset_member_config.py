from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..models.toolset_member_config_name import ToolsetMemberConfigName
from ..models.toolset_member_config_permission_type_0 import (
    ToolsetMemberConfigPermissionType0,
)
from ..models.toolset_member_config_transport_type_0 import (
    ToolsetMemberConfigTransportType0,
)
from ..types import UNSET, Unset

T = TypeVar("T", bound="ToolsetMemberConfig")


@_attrs_define
class ToolsetMemberConfig:
    """Per-member override within a curated toolset (``configs[]``).

    Unset fields inherit from the toolset's ``default_config``.

        Attributes:
            name (ToolsetMemberConfigName):
            enabled (bool | None | Unset):
            permission (None | ToolsetMemberConfigPermissionType0 | Unset):
            transport (None | ToolsetMemberConfigTransportType0 | Unset):
    """

    name: ToolsetMemberConfigName
    enabled: bool | None | Unset = UNSET
    permission: None | ToolsetMemberConfigPermissionType0 | Unset = UNSET
    transport: None | ToolsetMemberConfigTransportType0 | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        name = self.name.value

        enabled: bool | None | Unset
        if isinstance(self.enabled, Unset):
            enabled = UNSET
        else:
            enabled = self.enabled

        permission: None | str | Unset
        if isinstance(self.permission, Unset):
            permission = UNSET
        elif isinstance(self.permission, ToolsetMemberConfigPermissionType0):
            permission = self.permission.value
        else:
            permission = self.permission

        transport: None | str | Unset
        if isinstance(self.transport, Unset):
            transport = UNSET
        elif isinstance(self.transport, ToolsetMemberConfigTransportType0):
            transport = self.transport.value
        else:
            transport = self.transport

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "name": name,
            }
        )
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
        name = ToolsetMemberConfigName(d.pop("name"))

        def _parse_enabled(data: object) -> bool | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(bool | None | Unset, data)

        enabled = _parse_enabled(d.pop("enabled", UNSET))

        def _parse_permission(
            data: object,
        ) -> None | ToolsetMemberConfigPermissionType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                permission_type_0 = ToolsetMemberConfigPermissionType0(data)

                return permission_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | ToolsetMemberConfigPermissionType0 | Unset, data)

        permission = _parse_permission(d.pop("permission", UNSET))

        def _parse_transport(
            data: object,
        ) -> None | ToolsetMemberConfigTransportType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                transport_type_0 = ToolsetMemberConfigTransportType0(data)

                return transport_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | ToolsetMemberConfigTransportType0 | Unset, data)

        transport = _parse_transport(d.pop("transport", UNSET))

        toolset_member_config = cls(
            name=name,
            enabled=enabled,
            permission=permission,
            transport=transport,
        )

        return toolset_member_config
