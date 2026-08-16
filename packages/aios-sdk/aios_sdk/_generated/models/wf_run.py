from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.wf_run_status import WfRunStatus
from ..models.wf_run_workspace import WfRunWorkspace
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.http_server_spec import HttpServerSpec
    from ..models.mcp_server_spec import McpServerSpec
    from ..models.tool_spec import ToolSpec
    from ..models.wf_run_caller_type_0 import WfRunCallerType0
    from ..models.wf_run_request_output_schema_type_0 import (
        WfRunRequestOutputSchemaType0,
    )
    from ..models.wf_run_usage import WfRunUsage


T = TypeVar("T", bound="WfRun")


@_attrs_define
class WfRun:
    """A durable workflow execution instance.

    ``script`` is the run's own immutable snapshot of the workflow source at
    creation time (``script_sha`` is its hash); every wake execs exactly this.
    ``tools``/``mcp_servers``/``http_servers`` are the matching snapshot of the
    declared tool surface — pinned at launch like ``script``, so a later
    ``update_workflow`` never shifts an in-flight run's authority.
    ``status`` is persisted (unlike sessions): the run loop writes
    ``suspended``/``completed``/``errored``.

    NB (#1140): the run's lifecycle field is ``status`` — there is no ``state``
    field on a run. A watcher polling ``.state`` reads ``None`` forever even
    though ``output`` is already populated; poll ``status`` (terminal values:
    ``completed``/``errored``/``cancelled``).

        Attributes:
            id (str):
            account_id (str):
            environment_id (str):
            script (str):
            script_sha (str):
            host_semantics_epoch (int):
            status (WfRunStatus): The run's lifecycle status — the ONLY lifecycle field on a run (there is no `state` field;
                a watcher keying on `.state` waits forever). Terminal values: `completed`/`errored`/`cancelled`.
            last_event_seq (int):
            created_at (datetime.datetime):
            updated_at (datetime.datetime):
            workflow_id (None | str | Unset):
            workspace (WfRunWorkspace | Unset):  Default: WfRunWorkspace.FRESH.
            workspace_path (None | str | Unset):
            parent_run_id (None | str | Unset):
            launcher_session_id (None | str | Unset):
            depth (int | Unset):  Default: 0.
            request_id (None | str | Unset):
            caller (None | Unset | WfRunCallerType0):
            request_output_schema (None | Unset | WfRunRequestOutputSchemaType0):
            source_version (int | None | Unset):
            tools (list[ToolSpec] | Unset):
            mcp_servers (list[McpServerSpec] | Unset):
            http_servers (list[HttpServerSpec] | Unset):
            input_ (Any | Unset):
            output (Any | Unset):
            budget_usd (float | None | Unset):
            default_child_model (None | str | Unset):
            call_llm_cost_microusd (int | Unset):  Default: 0.
            archived_at (datetime.datetime | None | Unset):
            usage (None | Unset | WfRunUsage):
    """

    id: str
    account_id: str
    environment_id: str
    script: str
    script_sha: str
    host_semantics_epoch: int
    status: WfRunStatus
    last_event_seq: int
    created_at: datetime.datetime
    updated_at: datetime.datetime
    workflow_id: None | str | Unset = UNSET
    workspace: WfRunWorkspace | Unset = WfRunWorkspace.FRESH
    workspace_path: None | str | Unset = UNSET
    parent_run_id: None | str | Unset = UNSET
    launcher_session_id: None | str | Unset = UNSET
    depth: int | Unset = 0
    request_id: None | str | Unset = UNSET
    caller: None | Unset | WfRunCallerType0 = UNSET
    request_output_schema: None | Unset | WfRunRequestOutputSchemaType0 = UNSET
    source_version: int | None | Unset = UNSET
    tools: list[ToolSpec] | Unset = UNSET
    mcp_servers: list[McpServerSpec] | Unset = UNSET
    http_servers: list[HttpServerSpec] | Unset = UNSET
    input_: Any | Unset = UNSET
    output: Any | Unset = UNSET
    budget_usd: float | None | Unset = UNSET
    default_child_model: None | str | Unset = UNSET
    call_llm_cost_microusd: int | Unset = 0
    archived_at: datetime.datetime | None | Unset = UNSET
    usage: None | Unset | WfRunUsage = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.wf_run_caller_type_0 import WfRunCallerType0
        from ..models.wf_run_request_output_schema_type_0 import (
            WfRunRequestOutputSchemaType0,
        )
        from ..models.wf_run_usage import WfRunUsage

        id = self.id

        account_id = self.account_id

        environment_id = self.environment_id

        script = self.script

        script_sha = self.script_sha

        host_semantics_epoch = self.host_semantics_epoch

        status = self.status.value

        last_event_seq = self.last_event_seq

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        workflow_id: None | str | Unset
        if isinstance(self.workflow_id, Unset):
            workflow_id = UNSET
        else:
            workflow_id = self.workflow_id

        workspace: str | Unset = UNSET
        if not isinstance(self.workspace, Unset):
            workspace = self.workspace.value

        workspace_path: None | str | Unset
        if isinstance(self.workspace_path, Unset):
            workspace_path = UNSET
        else:
            workspace_path = self.workspace_path

        parent_run_id: None | str | Unset
        if isinstance(self.parent_run_id, Unset):
            parent_run_id = UNSET
        else:
            parent_run_id = self.parent_run_id

        launcher_session_id: None | str | Unset
        if isinstance(self.launcher_session_id, Unset):
            launcher_session_id = UNSET
        else:
            launcher_session_id = self.launcher_session_id

        depth = self.depth

        request_id: None | str | Unset
        if isinstance(self.request_id, Unset):
            request_id = UNSET
        else:
            request_id = self.request_id

        caller: dict[str, Any] | None | Unset
        if isinstance(self.caller, Unset):
            caller = UNSET
        elif isinstance(self.caller, WfRunCallerType0):
            caller = self.caller.to_dict()
        else:
            caller = self.caller

        request_output_schema: dict[str, Any] | None | Unset
        if isinstance(self.request_output_schema, Unset):
            request_output_schema = UNSET
        elif isinstance(self.request_output_schema, WfRunRequestOutputSchemaType0):
            request_output_schema = self.request_output_schema.to_dict()
        else:
            request_output_schema = self.request_output_schema

        source_version: int | None | Unset
        if isinstance(self.source_version, Unset):
            source_version = UNSET
        else:
            source_version = self.source_version

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

        input_ = self.input_

        output = self.output

        budget_usd: float | None | Unset
        if isinstance(self.budget_usd, Unset):
            budget_usd = UNSET
        else:
            budget_usd = self.budget_usd

        default_child_model: None | str | Unset
        if isinstance(self.default_child_model, Unset):
            default_child_model = UNSET
        else:
            default_child_model = self.default_child_model

        call_llm_cost_microusd = self.call_llm_cost_microusd

        archived_at: None | str | Unset
        if isinstance(self.archived_at, Unset):
            archived_at = UNSET
        elif isinstance(self.archived_at, datetime.datetime):
            archived_at = self.archived_at.isoformat()
        else:
            archived_at = self.archived_at

        usage: dict[str, Any] | None | Unset
        if isinstance(self.usage, Unset):
            usage = UNSET
        elif isinstance(self.usage, WfRunUsage):
            usage = self.usage.to_dict()
        else:
            usage = self.usage

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "account_id": account_id,
                "environment_id": environment_id,
                "script": script,
                "script_sha": script_sha,
                "host_semantics_epoch": host_semantics_epoch,
                "status": status,
                "last_event_seq": last_event_seq,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        if workflow_id is not UNSET:
            field_dict["workflow_id"] = workflow_id
        if workspace is not UNSET:
            field_dict["workspace"] = workspace
        if workspace_path is not UNSET:
            field_dict["workspace_path"] = workspace_path
        if parent_run_id is not UNSET:
            field_dict["parent_run_id"] = parent_run_id
        if launcher_session_id is not UNSET:
            field_dict["launcher_session_id"] = launcher_session_id
        if depth is not UNSET:
            field_dict["depth"] = depth
        if request_id is not UNSET:
            field_dict["request_id"] = request_id
        if caller is not UNSET:
            field_dict["caller"] = caller
        if request_output_schema is not UNSET:
            field_dict["request_output_schema"] = request_output_schema
        if source_version is not UNSET:
            field_dict["source_version"] = source_version
        if tools is not UNSET:
            field_dict["tools"] = tools
        if mcp_servers is not UNSET:
            field_dict["mcp_servers"] = mcp_servers
        if http_servers is not UNSET:
            field_dict["http_servers"] = http_servers
        if input_ is not UNSET:
            field_dict["input"] = input_
        if output is not UNSET:
            field_dict["output"] = output
        if budget_usd is not UNSET:
            field_dict["budget_usd"] = budget_usd
        if default_child_model is not UNSET:
            field_dict["default_child_model"] = default_child_model
        if call_llm_cost_microusd is not UNSET:
            field_dict["call_llm_cost_microusd"] = call_llm_cost_microusd
        if archived_at is not UNSET:
            field_dict["archived_at"] = archived_at
        if usage is not UNSET:
            field_dict["usage"] = usage

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.http_server_spec import HttpServerSpec
        from ..models.mcp_server_spec import McpServerSpec
        from ..models.tool_spec import ToolSpec
        from ..models.wf_run_caller_type_0 import WfRunCallerType0
        from ..models.wf_run_request_output_schema_type_0 import (
            WfRunRequestOutputSchemaType0,
        )
        from ..models.wf_run_usage import WfRunUsage

        d = dict(src_dict)
        id = d.pop("id")

        account_id = d.pop("account_id")

        environment_id = d.pop("environment_id")

        script = d.pop("script")

        script_sha = d.pop("script_sha")

        host_semantics_epoch = d.pop("host_semantics_epoch")

        status = WfRunStatus(d.pop("status"))

        last_event_seq = d.pop("last_event_seq")

        created_at = isoparse(d.pop("created_at"))

        updated_at = isoparse(d.pop("updated_at"))

        def _parse_workflow_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        workflow_id = _parse_workflow_id(d.pop("workflow_id", UNSET))

        _workspace = d.pop("workspace", UNSET)
        workspace: WfRunWorkspace | Unset
        if isinstance(_workspace, Unset):
            workspace = UNSET
        else:
            workspace = WfRunWorkspace(_workspace)

        def _parse_workspace_path(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        workspace_path = _parse_workspace_path(d.pop("workspace_path", UNSET))

        def _parse_parent_run_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        parent_run_id = _parse_parent_run_id(d.pop("parent_run_id", UNSET))

        def _parse_launcher_session_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        launcher_session_id = _parse_launcher_session_id(
            d.pop("launcher_session_id", UNSET)
        )

        depth = d.pop("depth", UNSET)

        def _parse_request_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        request_id = _parse_request_id(d.pop("request_id", UNSET))

        def _parse_caller(data: object) -> None | Unset | WfRunCallerType0:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                caller_type_0 = WfRunCallerType0.from_dict(data)

                return caller_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | WfRunCallerType0, data)

        caller = _parse_caller(d.pop("caller", UNSET))

        def _parse_request_output_schema(
            data: object,
        ) -> None | Unset | WfRunRequestOutputSchemaType0:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                request_output_schema_type_0 = WfRunRequestOutputSchemaType0.from_dict(
                    data
                )

                return request_output_schema_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | WfRunRequestOutputSchemaType0, data)

        request_output_schema = _parse_request_output_schema(
            d.pop("request_output_schema", UNSET)
        )

        def _parse_source_version(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        source_version = _parse_source_version(d.pop("source_version", UNSET))

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

        input_ = d.pop("input", UNSET)

        output = d.pop("output", UNSET)

        def _parse_budget_usd(data: object) -> float | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(float | None | Unset, data)

        budget_usd = _parse_budget_usd(d.pop("budget_usd", UNSET))

        def _parse_default_child_model(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        default_child_model = _parse_default_child_model(
            d.pop("default_child_model", UNSET)
        )

        call_llm_cost_microusd = d.pop("call_llm_cost_microusd", UNSET)

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

        def _parse_usage(data: object) -> None | Unset | WfRunUsage:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                usage_type_0 = WfRunUsage.from_dict(data)

                return usage_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | WfRunUsage, data)

        usage = _parse_usage(d.pop("usage", UNSET))

        wf_run = cls(
            id=id,
            account_id=account_id,
            environment_id=environment_id,
            script=script,
            script_sha=script_sha,
            host_semantics_epoch=host_semantics_epoch,
            status=status,
            last_event_seq=last_event_seq,
            created_at=created_at,
            updated_at=updated_at,
            workflow_id=workflow_id,
            workspace=workspace,
            workspace_path=workspace_path,
            parent_run_id=parent_run_id,
            launcher_session_id=launcher_session_id,
            depth=depth,
            request_id=request_id,
            caller=caller,
            request_output_schema=request_output_schema,
            source_version=source_version,
            tools=tools,
            mcp_servers=mcp_servers,
            http_servers=http_servers,
            input_=input_,
            output=output,
            budget_usd=budget_usd,
            default_child_model=default_child_model,
            call_llm_cost_microusd=call_llm_cost_microusd,
            archived_at=archived_at,
            usage=usage,
        )

        wf_run.additional_properties = d
        return wf_run

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
