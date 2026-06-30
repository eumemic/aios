from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.http_server_spec import HttpServerSpec
    from ..models.mcp_server_spec import McpServerSpec
    from ..models.tool_spec import ToolSpec
    from ..models.workflow_version_input_schema_type_0 import (
        WorkflowVersionInputSchemaType0,
    )
    from ..models.workflow_version_output_schema_type_0 import (
        WorkflowVersionOutputSchemaType0,
    )


T = TypeVar("T", bound="WorkflowVersion")


@_attrs_define
class WorkflowVersion:
    """Read view of a specific workflow version from the immutable history.

    The workflow analog of :class:`aios.models.agents.AgentVersion`: a complete,
    immutable snapshot of a workflow's definition at one ``version``. ``name`` IS
    versioned — a rename mints a new version — so this carries it alongside the
    script + declared surface. Snapshots exactly ``update_workflow``'s no-op
    comparison set (``name, script, input_schema, output_schema, description,
    tools, mcp_servers, http_servers``).

        Attributes:
            workflow_id (str):
            version (int):
            name (str):
            script (str):
            created_at (datetime.datetime):
            input_schema (None | Unset | WorkflowVersionInputSchemaType0):
            output_schema (None | Unset | WorkflowVersionOutputSchemaType0):
            output_model (None | str | Unset):
            description (None | str | Unset):
            tools (list[ToolSpec] | Unset):
            mcp_servers (list[McpServerSpec] | Unset):
            http_servers (list[HttpServerSpec] | Unset):
    """

    workflow_id: str
    version: int
    name: str
    script: str
    created_at: datetime.datetime
    input_schema: None | Unset | WorkflowVersionInputSchemaType0 = UNSET
    output_schema: None | Unset | WorkflowVersionOutputSchemaType0 = UNSET
    output_model: None | str | Unset = UNSET
    description: None | str | Unset = UNSET
    tools: list[ToolSpec] | Unset = UNSET
    mcp_servers: list[McpServerSpec] | Unset = UNSET
    http_servers: list[HttpServerSpec] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.workflow_version_input_schema_type_0 import (
            WorkflowVersionInputSchemaType0,
        )
        from ..models.workflow_version_output_schema_type_0 import (
            WorkflowVersionOutputSchemaType0,
        )

        workflow_id = self.workflow_id

        version = self.version

        name = self.name

        script = self.script

        created_at = self.created_at.isoformat()

        input_schema: dict[str, Any] | None | Unset
        if isinstance(self.input_schema, Unset):
            input_schema = UNSET
        elif isinstance(self.input_schema, WorkflowVersionInputSchemaType0):
            input_schema = self.input_schema.to_dict()
        else:
            input_schema = self.input_schema

        output_schema: dict[str, Any] | None | Unset
        if isinstance(self.output_schema, Unset):
            output_schema = UNSET
        elif isinstance(self.output_schema, WorkflowVersionOutputSchemaType0):
            output_schema = self.output_schema.to_dict()
        else:
            output_schema = self.output_schema

        output_model: None | str | Unset
        if isinstance(self.output_model, Unset):
            output_model = UNSET
        else:
            output_model = self.output_model

        description: None | str | Unset
        if isinstance(self.description, Unset):
            description = UNSET
        else:
            description = self.description

        tools: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.tools, Unset):
            tools = []
            for tools_item_data in self.tools:
                tools_item = tools_item_data.to_dict()
                tools.append(tools_item)

        mcp_servers: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.mcp_servers, Unset):
            mcp_servers = []
            for mcp_servers_item_data in self.mcp_servers:
                mcp_servers_item = mcp_servers_item_data.to_dict()
                mcp_servers.append(mcp_servers_item)

        http_servers: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.http_servers, Unset):
            http_servers = []
            for http_servers_item_data in self.http_servers:
                http_servers_item = http_servers_item_data.to_dict()
                http_servers.append(http_servers_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "workflow_id": workflow_id,
                "version": version,
                "name": name,
                "script": script,
                "created_at": created_at,
            }
        )
        if input_schema is not UNSET:
            field_dict["input_schema"] = input_schema
        if output_schema is not UNSET:
            field_dict["output_schema"] = output_schema
        if output_model is not UNSET:
            field_dict["output_model"] = output_model
        if description is not UNSET:
            field_dict["description"] = description
        if tools is not UNSET:
            field_dict["tools"] = tools
        if mcp_servers is not UNSET:
            field_dict["mcp_servers"] = mcp_servers
        if http_servers is not UNSET:
            field_dict["http_servers"] = http_servers

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.http_server_spec import HttpServerSpec
        from ..models.mcp_server_spec import McpServerSpec
        from ..models.tool_spec import ToolSpec
        from ..models.workflow_version_input_schema_type_0 import (
            WorkflowVersionInputSchemaType0,
        )
        from ..models.workflow_version_output_schema_type_0 import (
            WorkflowVersionOutputSchemaType0,
        )

        d = dict(src_dict)
        workflow_id = d.pop("workflow_id")

        version = d.pop("version")

        name = d.pop("name")

        script = d.pop("script")

        created_at = isoparse(d.pop("created_at"))

        def _parse_input_schema(
            data: object,
        ) -> None | Unset | WorkflowVersionInputSchemaType0:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                input_schema_type_0 = WorkflowVersionInputSchemaType0.from_dict(data)

                return input_schema_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | WorkflowVersionInputSchemaType0, data)

        input_schema = _parse_input_schema(d.pop("input_schema", UNSET))

        def _parse_output_schema(
            data: object,
        ) -> None | Unset | WorkflowVersionOutputSchemaType0:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                output_schema_type_0 = WorkflowVersionOutputSchemaType0.from_dict(data)

                return output_schema_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | WorkflowVersionOutputSchemaType0, data)

        output_schema = _parse_output_schema(d.pop("output_schema", UNSET))

        def _parse_output_model(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        output_model = _parse_output_model(d.pop("output_model", UNSET))

        def _parse_description(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        description = _parse_description(d.pop("description", UNSET))

        _tools = d.pop("tools", UNSET)
        tools: list[ToolSpec] | Unset = UNSET
        if _tools is not UNSET:
            tools = []
            for tools_item_data in _tools:
                tools_item = ToolSpec.from_dict(tools_item_data)

                tools.append(tools_item)

        _mcp_servers = d.pop("mcp_servers", UNSET)
        mcp_servers: list[McpServerSpec] | Unset = UNSET
        if _mcp_servers is not UNSET:
            mcp_servers = []
            for mcp_servers_item_data in _mcp_servers:
                mcp_servers_item = McpServerSpec.from_dict(mcp_servers_item_data)

                mcp_servers.append(mcp_servers_item)

        _http_servers = d.pop("http_servers", UNSET)
        http_servers: list[HttpServerSpec] | Unset = UNSET
        if _http_servers is not UNSET:
            http_servers = []
            for http_servers_item_data in _http_servers:
                http_servers_item = HttpServerSpec.from_dict(http_servers_item_data)

                http_servers.append(http_servers_item)

        workflow_version = cls(
            workflow_id=workflow_id,
            version=version,
            name=name,
            script=script,
            created_at=created_at,
            input_schema=input_schema,
            output_schema=output_schema,
            output_model=output_model,
            description=description,
            tools=tools,
            mcp_servers=mcp_servers,
            http_servers=http_servers,
        )

        workflow_version.additional_properties = d
        return workflow_version

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
