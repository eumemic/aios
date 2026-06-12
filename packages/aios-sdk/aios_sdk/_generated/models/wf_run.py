from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.wf_run_status import WfRunStatus
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.http_server_spec import HttpServerSpec
    from ..models.mcp_server_spec import McpServerSpec
    from ..models.tool_spec import ToolSpec


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

        Attributes:
            id (str):
            workflow_id (str):
            account_id (str):
            environment_id (str):
            script (str):
            script_sha (str):
            host_semantics_epoch (int):
            status (WfRunStatus):
            last_event_seq (int):
            created_at (datetime.datetime):
            updated_at (datetime.datetime):
            parent_run_id (None | str | Unset):
            launcher_session_id (None | str | Unset):
            tools (list[ToolSpec] | Unset):
            mcp_servers (list[McpServerSpec] | Unset):
            http_servers (list[HttpServerSpec] | Unset):
            input_ (Any | Unset):
            output (Any | Unset):
            budget_usd (float | None | Unset):
            archived_at (datetime.datetime | None | Unset):
    """

    id: str
    workflow_id: str
    account_id: str
    environment_id: str
    script: str
    script_sha: str
    host_semantics_epoch: int
    status: WfRunStatus
    last_event_seq: int
    created_at: datetime.datetime
    updated_at: datetime.datetime
    parent_run_id: None | str | Unset = UNSET
    launcher_session_id: None | str | Unset = UNSET
    tools: list[ToolSpec] | Unset = UNSET
    mcp_servers: list[McpServerSpec] | Unset = UNSET
    http_servers: list[HttpServerSpec] | Unset = UNSET
    input_: Any | Unset = UNSET
    output: Any | Unset = UNSET
    budget_usd: float | None | Unset = UNSET
    archived_at: datetime.datetime | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        workflow_id = self.workflow_id

        account_id = self.account_id

        environment_id = self.environment_id

        script = self.script

        script_sha = self.script_sha

        host_semantics_epoch = self.host_semantics_epoch

        status = self.status.value

        last_event_seq = self.last_event_seq

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

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
                "workflow_id": workflow_id,
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
        if parent_run_id is not UNSET:
            field_dict["parent_run_id"] = parent_run_id
        if launcher_session_id is not UNSET:
            field_dict["launcher_session_id"] = launcher_session_id
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
        if archived_at is not UNSET:
            field_dict["archived_at"] = archived_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.http_server_spec import HttpServerSpec
        from ..models.mcp_server_spec import McpServerSpec
        from ..models.tool_spec import ToolSpec

        d = dict(src_dict)
        id = d.pop("id")

        workflow_id = d.pop("workflow_id")

        account_id = d.pop("account_id")

        environment_id = d.pop("environment_id")

        script = d.pop("script")

        script_sha = d.pop("script_sha")

        host_semantics_epoch = d.pop("host_semantics_epoch")

        status = WfRunStatus(d.pop("status"))

        last_event_seq = d.pop("last_event_seq")

        created_at = isoparse(d.pop("created_at"))

        updated_at = isoparse(d.pop("updated_at"))

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

        wf_run = cls(
            id=id,
            workflow_id=workflow_id,
            account_id=account_id,
            environment_id=environment_id,
            script=script,
            script_sha=script_sha,
            host_semantics_epoch=host_semantics_epoch,
            status=status,
            last_event_seq=last_event_seq,
            created_at=created_at,
            updated_at=updated_at,
            parent_run_id=parent_run_id,
            launcher_session_id=launcher_session_id,
            tools=tools,
            mcp_servers=mcp_servers,
            http_servers=http_servers,
            input_=input_,
            output=output,
            budget_usd=budget_usd,
            archived_at=archived_at,
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
