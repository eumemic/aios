from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..models.tool_spec_permission_type_0 import ToolSpecPermissionType0
from ..models.tool_spec_type_type_0 import ToolSpecTypeType0
from ..models.tool_spec_type_type_1 import ToolSpecTypeType1
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.mcp_tool_config import McpToolConfig
    from ..models.mcp_toolset_config import McpToolsetConfig
    from ..models.tool_spec_input_schema_type_0 import ToolSpecInputSchemaType0


T = TypeVar("T", bound="ToolSpec")


@_attrs_define
class ToolSpec:
    """One entry in an agent's ``tools`` list.

    For built-in tools, ``type`` is the tool name (``"bash"``, ``"read"``,
    etc.). For custom (client-executed) tools, ``type`` is ``"custom"`` and
    ``name``, ``description``, and ``input_schema`` are required. For MCP
    toolsets, ``type`` is ``"mcp_toolset"`` and ``mcp_server_name`` is
    required.

    ``enabled`` controls whether the tool is included in the schema sent to
    the model. Disabled tools are invisible to the model.

    ``permission`` controls execution policy for built-in tools:
    ``None`` or ``"always_allow"`` executes immediately (current default);
    ``"always_ask"`` idles the session with ``requires_action`` until the
    client confirms or denies.

        Attributes:
            type_ (ToolSpecTypeType0 | ToolSpecTypeType1):
            name (None | str | Unset):
            description (None | str | Unset):
            input_schema (None | ToolSpecInputSchemaType0 | Unset):
            enabled (bool | Unset):  Default: True.
            permission (None | ToolSpecPermissionType0 | Unset):
            mcp_server_name (None | str | Unset):
            default_config (McpToolsetConfig | None | Unset):
            configs (list[McpToolConfig] | None | Unset):
    """

    type_: ToolSpecTypeType0 | ToolSpecTypeType1
    name: None | str | Unset = UNSET
    description: None | str | Unset = UNSET
    input_schema: None | ToolSpecInputSchemaType0 | Unset = UNSET
    enabled: bool | Unset = True
    permission: None | ToolSpecPermissionType0 | Unset = UNSET
    mcp_server_name: None | str | Unset = UNSET
    default_config: McpToolsetConfig | None | Unset = UNSET
    configs: list[McpToolConfig] | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.mcp_toolset_config import McpToolsetConfig
        from ..models.tool_spec_input_schema_type_0 import ToolSpecInputSchemaType0

        type_: str
        if isinstance(self.type_, ToolSpecTypeType0):
            type_ = self.type_.value
        else:
            type_ = self.type_.value

        name: None | str | Unset
        if isinstance(self.name, Unset):
            name = UNSET
        else:
            name = self.name

        description: None | str | Unset
        if isinstance(self.description, Unset):
            description = UNSET
        else:
            description = self.description

        input_schema: dict[str, Any] | None | Unset
        if isinstance(self.input_schema, Unset):
            input_schema = UNSET
        elif isinstance(self.input_schema, ToolSpecInputSchemaType0):
            input_schema = self.input_schema.to_dict()
        else:
            input_schema = self.input_schema

        enabled = self.enabled

        permission: None | str | Unset
        if isinstance(self.permission, Unset):
            permission = UNSET
        elif isinstance(self.permission, ToolSpecPermissionType0):
            permission = self.permission.value
        else:
            permission = self.permission

        mcp_server_name: None | str | Unset
        if isinstance(self.mcp_server_name, Unset):
            mcp_server_name = UNSET
        else:
            mcp_server_name = self.mcp_server_name

        default_config: dict[str, Any] | None | Unset
        if isinstance(self.default_config, Unset):
            default_config = UNSET
        elif isinstance(self.default_config, McpToolsetConfig):
            default_config = self.default_config.to_dict()
        else:
            default_config = self.default_config

        configs: list[dict[str, Any]] | None | Unset
        if isinstance(self.configs, Unset):
            configs = UNSET
        elif isinstance(self.configs, list):
            configs = []
            for configs_type_0_item_data in self.configs:
                configs_type_0_item = configs_type_0_item_data.to_dict()
                configs.append(configs_type_0_item)

        else:
            configs = self.configs

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "type": type_,
            }
        )
        if name is not UNSET:
            field_dict["name"] = name
        if description is not UNSET:
            field_dict["description"] = description
        if input_schema is not UNSET:
            field_dict["input_schema"] = input_schema
        if enabled is not UNSET:
            field_dict["enabled"] = enabled
        if permission is not UNSET:
            field_dict["permission"] = permission
        if mcp_server_name is not UNSET:
            field_dict["mcp_server_name"] = mcp_server_name
        if default_config is not UNSET:
            field_dict["default_config"] = default_config
        if configs is not UNSET:
            field_dict["configs"] = configs

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.mcp_tool_config import McpToolConfig
        from ..models.mcp_toolset_config import McpToolsetConfig
        from ..models.tool_spec_input_schema_type_0 import ToolSpecInputSchemaType0

        d = dict(src_dict)

        def _parse_type_(data: object) -> ToolSpecTypeType0 | ToolSpecTypeType1:
            try:
                if not isinstance(data, str):
                    raise TypeError()
                type_type_0 = ToolSpecTypeType0(data)

                return type_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            if not isinstance(data, str):
                raise TypeError()
            type_type_1 = ToolSpecTypeType1(data)

            return type_type_1

        type_ = _parse_type_(d.pop("type"))

        def _parse_name(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        name = _parse_name(d.pop("name", UNSET))

        def _parse_description(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        description = _parse_description(d.pop("description", UNSET))

        def _parse_input_schema(
            data: object,
        ) -> None | ToolSpecInputSchemaType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                input_schema_type_0 = ToolSpecInputSchemaType0.from_dict(data)

                return input_schema_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | ToolSpecInputSchemaType0 | Unset, data)

        input_schema = _parse_input_schema(d.pop("input_schema", UNSET))

        enabled = d.pop("enabled", UNSET)

        def _parse_permission(data: object) -> None | ToolSpecPermissionType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                permission_type_0 = ToolSpecPermissionType0(data)

                return permission_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | ToolSpecPermissionType0 | Unset, data)

        permission = _parse_permission(d.pop("permission", UNSET))

        def _parse_mcp_server_name(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        mcp_server_name = _parse_mcp_server_name(d.pop("mcp_server_name", UNSET))

        def _parse_default_config(data: object) -> McpToolsetConfig | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                default_config_type_0 = McpToolsetConfig.from_dict(data)

                return default_config_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(McpToolsetConfig | None | Unset, data)

        default_config = _parse_default_config(d.pop("default_config", UNSET))

        def _parse_configs(data: object) -> list[McpToolConfig] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                configs_type_0 = []
                _configs_type_0 = data
                for configs_type_0_item_data in _configs_type_0:
                    configs_type_0_item = McpToolConfig.from_dict(
                        configs_type_0_item_data
                    )

                    configs_type_0.append(configs_type_0_item)

                return configs_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[McpToolConfig] | None | Unset, data)

        configs = _parse_configs(d.pop("configs", UNSET))

        tool_spec = cls(
            type_=type_,
            name=name,
            description=description,
            input_schema=input_schema,
            enabled=enabled,
            permission=permission,
            mcp_server_name=mcp_server_name,
            default_config=default_config,
            configs=configs,
        )

        return tool_spec
