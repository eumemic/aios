from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.agent_litellm_extra import AgentLitellmExtra
    from ..models.agent_metadata import AgentMetadata
    from ..models.agent_skill_ref import AgentSkillRef
    from ..models.mcp_server_spec import McpServerSpec
    from ..models.tool_spec import ToolSpec


T = TypeVar("T", bound="Agent")


@_attrs_define
class Agent:
    """Read view of an agent (always the latest version).

    Attributes:
        id (str):
        version (int):
        name (str):
        model (str):
        system (str):
        tools (list[ToolSpec]):
        mcp_servers (list[McpServerSpec]):
        description (None | str):
        metadata (AgentMetadata):
        window_min (int):
        window_max (int):
        created_at (datetime.datetime):
        updated_at (datetime.datetime):
        skills (list[AgentSkillRef] | Unset):
        litellm_extra (AgentLitellmExtra | Unset):
        archived_at (datetime.datetime | None | Unset):
    """

    id: str
    version: int
    name: str
    model: str
    system: str
    tools: list[ToolSpec]
    mcp_servers: list[McpServerSpec]
    description: None | str
    metadata: AgentMetadata
    window_min: int
    window_max: int
    created_at: datetime.datetime
    updated_at: datetime.datetime
    skills: list[AgentSkillRef] | Unset = UNSET
    litellm_extra: AgentLitellmExtra | Unset = UNSET
    archived_at: datetime.datetime | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        version = self.version

        name = self.name

        model = self.model

        system = self.system

        tools = []
        for tools_item_data in self.tools:
            tools_item = tools_item_data.to_dict()
            tools.append(tools_item)

        mcp_servers = []
        for mcp_servers_item_data in self.mcp_servers:
            mcp_servers_item = mcp_servers_item_data.to_dict()
            mcp_servers.append(mcp_servers_item)

        description: None | str
        description = self.description

        metadata = self.metadata.to_dict()

        window_min = self.window_min

        window_max = self.window_max

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        skills: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.skills, Unset):
            skills = []
            for skills_item_data in self.skills:
                skills_item = skills_item_data.to_dict()
                skills.append(skills_item)

        litellm_extra: dict[str, Any] | Unset = UNSET
        if not isinstance(self.litellm_extra, Unset):
            litellm_extra = self.litellm_extra.to_dict()

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
                "version": version,
                "name": name,
                "model": model,
                "system": system,
                "tools": tools,
                "mcp_servers": mcp_servers,
                "description": description,
                "metadata": metadata,
                "window_min": window_min,
                "window_max": window_max,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        if skills is not UNSET:
            field_dict["skills"] = skills
        if litellm_extra is not UNSET:
            field_dict["litellm_extra"] = litellm_extra
        if archived_at is not UNSET:
            field_dict["archived_at"] = archived_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.agent_litellm_extra import AgentLitellmExtra
        from ..models.agent_metadata import AgentMetadata
        from ..models.agent_skill_ref import AgentSkillRef
        from ..models.mcp_server_spec import McpServerSpec
        from ..models.tool_spec import ToolSpec

        d = dict(src_dict)
        id = d.pop("id")

        version = d.pop("version")

        name = d.pop("name")

        model = d.pop("model")

        system = d.pop("system")

        tools = []
        _tools = d.pop("tools")
        for tools_item_data in _tools:
            tools_item = ToolSpec.from_dict(tools_item_data)

            tools.append(tools_item)

        mcp_servers = []
        _mcp_servers = d.pop("mcp_servers")
        for mcp_servers_item_data in _mcp_servers:
            mcp_servers_item = McpServerSpec.from_dict(mcp_servers_item_data)

            mcp_servers.append(mcp_servers_item)

        def _parse_description(data: object) -> None | str:
            if data is None:
                return data
            return cast(None | str, data)

        description = _parse_description(d.pop("description"))

        metadata = AgentMetadata.from_dict(d.pop("metadata"))

        window_min = d.pop("window_min")

        window_max = d.pop("window_max")

        created_at = isoparse(d.pop("created_at"))

        updated_at = isoparse(d.pop("updated_at"))

        _skills = d.pop("skills", UNSET)
        skills: list[AgentSkillRef] | Unset = UNSET
        if _skills is not UNSET:
            skills = []
            for skills_item_data in _skills:
                skills_item = AgentSkillRef.from_dict(skills_item_data)

                skills.append(skills_item)

        _litellm_extra = d.pop("litellm_extra", UNSET)
        litellm_extra: AgentLitellmExtra | Unset
        if isinstance(_litellm_extra, Unset):
            litellm_extra = UNSET
        else:
            litellm_extra = AgentLitellmExtra.from_dict(_litellm_extra)

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

        agent = cls(
            id=id,
            version=version,
            name=name,
            model=model,
            system=system,
            tools=tools,
            mcp_servers=mcp_servers,
            description=description,
            metadata=metadata,
            window_min=window_min,
            window_max=window_max,
            created_at=created_at,
            updated_at=updated_at,
            skills=skills,
            litellm_extra=litellm_extra,
            archived_at=archived_at,
        )

        agent.additional_properties = d
        return agent

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
