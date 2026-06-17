from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..models.mcp_tool_config_transport_type_0 import McpToolConfigTransportType0
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.mcp_permission_policy import McpPermissionPolicy


T = TypeVar("T", bound="McpToolConfig")


@_attrs_define
class McpToolConfig:
    """Per-tool override within an ``mcp_toolset`` entry.

    ``read_allow`` opts a single discovered tool into the outbound-suppression
    read allowlist (#710): MCP has no HTTP-method convention, so when a session
    runs with ``outbound_suppression == "on"`` every MCP call is *default-deny*
    (suppressed with a synthesized success) UNLESS the operator marked the
    specific tool ``read_allow=True`` at config time. A read-allowed tool runs
    for real even under suppression. Default ``False`` — the safe choice for a
    protocol that can't self-describe side effects.

        Attributes:
            name (str):
            enabled (bool | Unset):  Default: True.
            permission_policy (McpPermissionPolicy | None | Unset):
            transport (McpToolConfigTransportType0 | None | Unset):
            read_allow (bool | Unset):  Default: False.
    """

    name: str
    enabled: bool | Unset = True
    permission_policy: McpPermissionPolicy | None | Unset = UNSET
    transport: McpToolConfigTransportType0 | None | Unset = UNSET
    read_allow: bool | Unset = False

    def to_dict(self) -> dict[str, Any]:
        from ..models.mcp_permission_policy import McpPermissionPolicy

        name = self.name

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
        elif isinstance(self.transport, McpToolConfigTransportType0):
            transport = self.transport.value
        else:
            transport = self.transport

        read_allow = self.read_allow

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "name": name,
            }
        )
        if enabled is not UNSET:
            field_dict["enabled"] = enabled
        if permission_policy is not UNSET:
            field_dict["permission_policy"] = permission_policy
        if transport is not UNSET:
            field_dict["transport"] = transport
        if read_allow is not UNSET:
            field_dict["read_allow"] = read_allow

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.mcp_permission_policy import McpPermissionPolicy

        d = dict(src_dict)
        name = d.pop("name")

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
        ) -> McpToolConfigTransportType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                transport_type_0 = McpToolConfigTransportType0(data)

                return transport_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(McpToolConfigTransportType0 | None | Unset, data)

        transport = _parse_transport(d.pop("transport", UNSET))

        read_allow = d.pop("read_allow", UNSET)

        mcp_tool_config = cls(
            name=name,
            enabled=enabled,
            permission_policy=permission_policy,
            transport=transport,
            read_allow=read_allow,
        )

        return mcp_tool_config
