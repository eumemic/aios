from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.agent_version_preempt_policy import AgentVersionPreemptPolicy
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.agent_skill_ref import AgentSkillRef
    from ..models.agent_version_litellm_extra import AgentVersionLitellmExtra
    from ..models.http_server_spec import HttpServerSpec
    from ..models.mcp_server_spec import McpServerSpec
    from ..models.tool_spec import ToolSpec


T = TypeVar("T", bound="AgentVersion")


@_attrs_define
class AgentVersion:
    """Read view of a specific agent version from the version history.

    Attributes:
        agent_id (str):
        version (int):
        model (str):
        system (str):
        tools (list[ToolSpec]):
        mcp_servers (list[McpServerSpec]):
        window_min (int):
        window_max (int):
        created_at (datetime.datetime):
        skills (list[AgentSkillRef] | Unset):
        http_servers (list[HttpServerSpec] | Unset):
        litellm_extra (AgentVersionLitellmExtra | Unset):
        preempt_policy (AgentVersionPreemptPolicy | Unset):  Default: AgentVersionPreemptPolicy.WAIT.
    """

    agent_id: str
    version: int
    model: str
    system: str
    tools: list[ToolSpec]
    mcp_servers: list[McpServerSpec]
    window_min: int
    window_max: int
    created_at: datetime.datetime
    skills: list[AgentSkillRef] | Unset = UNSET
    http_servers: list[HttpServerSpec] | Unset = UNSET
    litellm_extra: AgentVersionLitellmExtra | Unset = UNSET
    preempt_policy: AgentVersionPreemptPolicy | Unset = AgentVersionPreemptPolicy.WAIT
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        agent_id = self.agent_id

        version = self.version

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

        window_min = self.window_min

        window_max = self.window_max

        created_at = self.created_at.isoformat()

        skills: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.skills, Unset):
            skills = []
            for skills_item_data in self.skills:
                skills_item = skills_item_data.to_dict()
                skills.append(skills_item)

        http_servers: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.http_servers, Unset):
            http_servers = []
            for http_servers_item_data in self.http_servers:
                http_servers_item = http_servers_item_data.to_dict()
                http_servers.append(http_servers_item)

        litellm_extra: dict[str, Any] | Unset = UNSET
        if not isinstance(self.litellm_extra, Unset):
            litellm_extra = self.litellm_extra.to_dict()

        preempt_policy: str | Unset = UNSET
        if not isinstance(self.preempt_policy, Unset):
            preempt_policy = self.preempt_policy.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "agent_id": agent_id,
                "version": version,
                "model": model,
                "system": system,
                "tools": tools,
                "mcp_servers": mcp_servers,
                "window_min": window_min,
                "window_max": window_max,
                "created_at": created_at,
            }
        )
        if skills is not UNSET:
            field_dict["skills"] = skills
        if http_servers is not UNSET:
            field_dict["http_servers"] = http_servers
        if litellm_extra is not UNSET:
            field_dict["litellm_extra"] = litellm_extra
        if preempt_policy is not UNSET:
            field_dict["preempt_policy"] = preempt_policy

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.agent_skill_ref import AgentSkillRef
        from ..models.agent_version_litellm_extra import AgentVersionLitellmExtra
        from ..models.http_server_spec import HttpServerSpec
        from ..models.mcp_server_spec import McpServerSpec
        from ..models.tool_spec import ToolSpec

        d = dict(src_dict)
        agent_id = d.pop("agent_id")

        version = d.pop("version")

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

        window_min = d.pop("window_min")

        window_max = d.pop("window_max")

        created_at = isoparse(d.pop("created_at"))

        _skills = d.pop("skills", UNSET)
        skills: list[AgentSkillRef] | Unset = UNSET
        if _skills is not UNSET:
            skills = []
            for skills_item_data in _skills:
                skills_item = AgentSkillRef.from_dict(skills_item_data)

                skills.append(skills_item)

        _http_servers = d.pop("http_servers", UNSET)
        http_servers: list[HttpServerSpec] | Unset = UNSET
        if _http_servers is not UNSET:
            http_servers = []
            for http_servers_item_data in _http_servers:
                http_servers_item = HttpServerSpec.from_dict(http_servers_item_data)

                http_servers.append(http_servers_item)

        _litellm_extra = d.pop("litellm_extra", UNSET)
        litellm_extra: AgentVersionLitellmExtra | Unset
        if isinstance(_litellm_extra, Unset):
            litellm_extra = UNSET
        else:
            litellm_extra = AgentVersionLitellmExtra.from_dict(_litellm_extra)

        _preempt_policy = d.pop("preempt_policy", UNSET)
        preempt_policy: AgentVersionPreemptPolicy | Unset
        if isinstance(_preempt_policy, Unset):
            preempt_policy = UNSET
        else:
            preempt_policy = AgentVersionPreemptPolicy(_preempt_policy)

        agent_version = cls(
            agent_id=agent_id,
            version=version,
            model=model,
            system=system,
            tools=tools,
            mcp_servers=mcp_servers,
            window_min=window_min,
            window_max=window_max,
            created_at=created_at,
            skills=skills,
            http_servers=http_servers,
            litellm_extra=litellm_extra,
            preempt_policy=preempt_policy,
        )

        agent_version.additional_properties = d
        return agent_version

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
