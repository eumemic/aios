from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..models.mcp_toolset_config_transport_type_0 import McpToolsetConfigTransportType0
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.mcp_permission_policy import McpPermissionPolicy


T = TypeVar("T", bound="McpToolsetConfig")


@_attrs_define
class McpToolsetConfig:
    """Default config for all tools discovered from an MCP server.

    Attributes:
        enabled (bool | Unset):  Default: True.
        permission_policy (McpPermissionPolicy | None | Unset):
        transport (McpToolsetConfigTransportType0 | None | Unset):
    """

    enabled: bool | Unset = True
    permission_policy: McpPermissionPolicy | None | Unset = UNSET
    transport: McpToolsetConfigTransportType0 | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.mcp_permission_policy import McpPermissionPolicy

        enabled = self.enabled

        permission_policy: dict[str, Any] | None | Unset
        if isinstance(self.permission_policy, Unset):
            permission_policy = UNSET
        elif isinstance(self.permission_policy, McpPermissionPolicy):
            permission_policy = self.permission_policy.to_dict()
        else:
            permission_policy = self.permission_policy

        transport: None | str | Unset
        if isinstance(self.transport, Unset):
            transport = UNSET
        elif isinstance(self.transport, McpToolsetConfigTransportType0):
            transport = self.transport.value
        else:
            transport = self.transport

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if enabled is not UNSET:
            field_dict["enabled"] = enabled
        if permission_policy is not UNSET:
            field_dict["permission_policy"] = permission_policy
        if transport is not UNSET:
            field_dict["transport"] = transport

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.mcp_permission_policy import McpPermissionPolicy

        d = dict(src_dict)
        enabled = d.pop("enabled", UNSET)

        def _parse_permission_policy(
            data: object,
        ) -> McpPermissionPolicy | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                permission_policy_type_0 = McpPermissionPolicy.from_dict(data)

                return permission_policy_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(McpPermissionPolicy | None | Unset, data)

        permission_policy = _parse_permission_policy(d.pop("permission_policy", UNSET))

        def _parse_transport(
            data: object,
        ) -> McpToolsetConfigTransportType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                transport_type_0 = McpToolsetConfigTransportType0(data)

                return transport_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(McpToolsetConfigTransportType0 | None | Unset, data)

        transport = _parse_transport(d.pop("transport", UNSET))

        mcp_toolset_config = cls(
            enabled=enabled,
            permission_policy=permission_policy,
            transport=transport,
        )

        return mcp_toolset_config
