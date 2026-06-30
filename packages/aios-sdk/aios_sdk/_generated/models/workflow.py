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
    from ..models.workflow_input_schema_type_0 import WorkflowInputSchemaType0
    from ..models.workflow_output_schema_type_0 import WorkflowOutputSchemaType0


T = TypeVar("T", bound="Workflow")


@_attrs_define
class Workflow:
    """A versioned workflow definition (updated in place; ``version`` bumps per change).

    Attributes:
        id (str):
        account_id (str):
        name (str):
        version (int):
        script (str):
        created_at (datetime.datetime):
        updated_at (datetime.datetime):
        input_schema (None | Unset | WorkflowInputSchemaType0):
        output_schema (None | Unset | WorkflowOutputSchemaType0):
        output_model (None | str | Unset):
        description (None | str | Unset):
        tools (list[ToolSpec] | Unset):
        mcp_servers (list[McpServerSpec] | Unset):
        http_servers (list[HttpServerSpec] | Unset):
        archived_at (datetime.datetime | None | Unset):
    """

    id: str
    account_id: str
    name: str
    version: int
    script: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    input_schema: None | Unset | WorkflowInputSchemaType0 = UNSET
    output_schema: None | Unset | WorkflowOutputSchemaType0 = UNSET
    output_model: None | str | Unset = UNSET
    description: None | str | Unset = UNSET
    tools: list[ToolSpec] | Unset = UNSET
    mcp_servers: list[McpServerSpec] | Unset = UNSET
    http_servers: list[HttpServerSpec] | Unset = UNSET
    archived_at: datetime.datetime | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.workflow_input_schema_type_0 import WorkflowInputSchemaType0
        from ..models.workflow_output_schema_type_0 import WorkflowOutputSchemaType0

        id = self.id

        account_id = self.account_id

        name = self.name

        version = self.version

        script = self.script

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        input_schema: dict[str, Any] | None | Unset
        if isinstance(self.input_schema, Unset):
            input_schema = UNSET
        elif isinstance(self.input_schema, WorkflowInputSchemaType0):
            input_schema = self.input_schema.to_dict()
        else:
            input_schema = self.input_schema

        output_schema: dict[str, Any] | None | Unset
        if isinstance(self.output_schema, Unset):
            output_schema = UNSET
        elif isinstance(self.output_schema, WorkflowOutputSchemaType0):
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

        archived_at: None | str | Unset
        if isinstance(self.archived_at, Unset):
            archived_at = UNSET
        elif isinstance(self.archived_at, datetime.datetime):
            archived_at = self.archived_at.isoformat()
        else:
            archived_at = self.archived_at

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "account_id": account_id,
                "name": name,
                "version": version,
                "script": script,
                "created_at": created_at,
                "updated_at": updated_at,
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
        if archived_at is not UNSET:
            field_dict["archived_at"] = archived_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.http_server_spec import HttpServerSpec
        from ..models.mcp_server_spec import McpServerSpec
        from ..models.tool_spec import ToolSpec
        from ..models.workflow_input_schema_type_0 import WorkflowInputSchemaType0
        from ..models.workflow_output_schema_type_0 import WorkflowOutputSchemaType0

        d = dict(src_dict)
        id = d.pop("id")

        account_id = d.pop("account_id")

        name = d.pop("name")

        version = d.pop("version")

        script = d.pop("script")

        created_at = isoparse(d.pop("created_at"))

        updated_at = isoparse(d.pop("updated_at"))

        def _parse_input_schema(
            data: object,
        ) -> None | Unset | WorkflowInputSchemaType0:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                input_schema_type_0 = WorkflowInputSchemaType0.from_dict(data)

                return input_schema_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | WorkflowInputSchemaType0, data)

        input_schema = _parse_input_schema(d.pop("input_schema", UNSET))

        def _parse_output_schema(
            data: object,
        ) -> None | Unset | WorkflowOutputSchemaType0:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                output_schema_type_0 = WorkflowOutputSchemaType0.from_dict(data)

                return output_schema_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | WorkflowOutputSchemaType0, data)

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

        def _parse_archived_at(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                archived_at_type_0 = isoparse(data)

                return archived_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        archived_at = _parse_archived_at(d.pop("archived_at", UNSET))

        workflow = cls(
            id=id,
            account_id=account_id,
            name=name,
            version=version,
            script=script,
            created_at=created_at,
            updated_at=updated_at,
            input_schema=input_schema,
            output_schema=output_schema,
            output_model=output_model,
            description=description,
            tools=tools,
            mcp_servers=mcp_servers,
            http_servers=http_servers,
            archived_at=archived_at,
        )

        workflow.additional_properties = d
        return workflow

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
