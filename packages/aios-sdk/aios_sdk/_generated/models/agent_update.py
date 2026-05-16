from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.agent_skill_ref import AgentSkillRef
    from ..models.agent_update_litellm_extra_type_0 import AgentUpdateLitellmExtraType0
    from ..models.agent_update_metadata_type_0 import AgentUpdateMetadataType0
    from ..models.http_server_spec import HttpServerSpec
    from ..models.mcp_server_spec import McpServerSpec
    from ..models.tool_spec import ToolSpec


T = TypeVar("T", bound="AgentUpdate")


@_attrs_define
class AgentUpdate:
    """Request body for ``PUT /v1/agents/{id}``.

    All config fields are optional; omitted fields are preserved. The
    ``version`` field is required for optimistic concurrency — it must match
    the current version. If the update produces a change, a new version is
    created; otherwise the existing version is returned unchanged.

        Attributes:
            version (int): Current version for optimistic concurrency.
            name (None | str | Unset):
            model (None | str | Unset):
            system (None | str | Unset):
            tools (list[ToolSpec] | None | Unset):
            skills (list[AgentSkillRef] | None | Unset):
            mcp_servers (list[McpServerSpec] | None | Unset):
            http_servers (list[HttpServerSpec] | None | Unset):
            description (None | str | Unset):
            metadata (AgentUpdateMetadataType0 | None | Unset):
            litellm_extra (AgentUpdateLitellmExtraType0 | None | Unset):
            window_min (int | None | Unset):
            window_max (int | None | Unset):
    """

    version: int
    name: None | str | Unset = UNSET
    model: None | str | Unset = UNSET
    system: None | str | Unset = UNSET
    tools: list[ToolSpec] | None | Unset = UNSET
    skills: list[AgentSkillRef] | None | Unset = UNSET
    mcp_servers: list[McpServerSpec] | None | Unset = UNSET
    http_servers: list[HttpServerSpec] | None | Unset = UNSET
    description: None | str | Unset = UNSET
    metadata: AgentUpdateMetadataType0 | None | Unset = UNSET
    litellm_extra: AgentUpdateLitellmExtraType0 | None | Unset = UNSET
    window_min: int | None | Unset = UNSET
    window_max: int | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.agent_update_litellm_extra_type_0 import (
            AgentUpdateLitellmExtraType0,
        )
        from ..models.agent_update_metadata_type_0 import AgentUpdateMetadataType0

        version = self.version

        name: None | str | Unset
        if isinstance(self.name, Unset):
            name = UNSET
        else:
            name = self.name

        model: None | str | Unset
        if isinstance(self.model, Unset):
            model = UNSET
        else:
            model = self.model

        system: None | str | Unset
        if isinstance(self.system, Unset):
            system = UNSET
        else:
            system = self.system

        tools: list[dict[str, Any]] | None | Unset
        if isinstance(self.tools, Unset):
            tools = UNSET
        elif isinstance(self.tools, list):
            tools = []
            for tools_type_0_item_data in self.tools:
                tools_type_0_item = tools_type_0_item_data.to_dict()
                tools.append(tools_type_0_item)

        else:
            tools = self.tools

        skills: list[dict[str, Any]] | None | Unset
        if isinstance(self.skills, Unset):
            skills = UNSET
        elif isinstance(self.skills, list):
            skills = []
            for skills_type_0_item_data in self.skills:
                skills_type_0_item = skills_type_0_item_data.to_dict()
                skills.append(skills_type_0_item)

        else:
            skills = self.skills

        mcp_servers: list[dict[str, Any]] | None | Unset
        if isinstance(self.mcp_servers, Unset):
            mcp_servers = UNSET
        elif isinstance(self.mcp_servers, list):
            mcp_servers = []
            for mcp_servers_type_0_item_data in self.mcp_servers:
                mcp_servers_type_0_item = mcp_servers_type_0_item_data.to_dict()
                mcp_servers.append(mcp_servers_type_0_item)

        else:
            mcp_servers = self.mcp_servers

        http_servers: list[dict[str, Any]] | None | Unset
        if isinstance(self.http_servers, Unset):
            http_servers = UNSET
        elif isinstance(self.http_servers, list):
            http_servers = []
            for http_servers_type_0_item_data in self.http_servers:
                http_servers_type_0_item = http_servers_type_0_item_data.to_dict()
                http_servers.append(http_servers_type_0_item)

        else:
            http_servers = self.http_servers

        description: None | str | Unset
        if isinstance(self.description, Unset):
            description = UNSET
        else:
            description = self.description

        metadata: dict[str, Any] | None | Unset
        if isinstance(self.metadata, Unset):
            metadata = UNSET
        elif isinstance(self.metadata, AgentUpdateMetadataType0):
            metadata = self.metadata.to_dict()
        else:
            metadata = self.metadata

        litellm_extra: dict[str, Any] | None | Unset
        if isinstance(self.litellm_extra, Unset):
            litellm_extra = UNSET
        elif isinstance(self.litellm_extra, AgentUpdateLitellmExtraType0):
            litellm_extra = self.litellm_extra.to_dict()
        else:
            litellm_extra = self.litellm_extra

        window_min: int | None | Unset
        if isinstance(self.window_min, Unset):
            window_min = UNSET
        else:
            window_min = self.window_min

        window_max: int | None | Unset
        if isinstance(self.window_max, Unset):
            window_max = UNSET
        else:
            window_max = self.window_max

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "version": version,
            }
        )
        if name is not UNSET:
            field_dict["name"] = name
        if model is not UNSET:
            field_dict["model"] = model
        if system is not UNSET:
            field_dict["system"] = system
        if tools is not UNSET:
            field_dict["tools"] = tools
        if skills is not UNSET:
            field_dict["skills"] = skills
        if mcp_servers is not UNSET:
            field_dict["mcp_servers"] = mcp_servers
        if http_servers is not UNSET:
            field_dict["http_servers"] = http_servers
        if description is not UNSET:
            field_dict["description"] = description
        if metadata is not UNSET:
            field_dict["metadata"] = metadata
        if litellm_extra is not UNSET:
            field_dict["litellm_extra"] = litellm_extra
        if window_min is not UNSET:
            field_dict["window_min"] = window_min
        if window_max is not UNSET:
            field_dict["window_max"] = window_max

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.agent_skill_ref import AgentSkillRef
        from ..models.agent_update_litellm_extra_type_0 import (
            AgentUpdateLitellmExtraType0,
        )
        from ..models.agent_update_metadata_type_0 import AgentUpdateMetadataType0
        from ..models.http_server_spec import HttpServerSpec
        from ..models.mcp_server_spec import McpServerSpec
        from ..models.tool_spec import ToolSpec

        d = dict(src_dict)
        version = d.pop("version")

        def _parse_name(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        name = _parse_name(d.pop("name", UNSET))

        def _parse_model(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        model = _parse_model(d.pop("model", UNSET))

        def _parse_system(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        system = _parse_system(d.pop("system", UNSET))

        def _parse_tools(data: object) -> list[ToolSpec] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                tools_type_0 = []
                _tools_type_0 = data
                for tools_type_0_item_data in _tools_type_0:
                    tools_type_0_item = ToolSpec.from_dict(tools_type_0_item_data)

                    tools_type_0.append(tools_type_0_item)

                return tools_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[ToolSpec] | None | Unset, data)

        tools = _parse_tools(d.pop("tools", UNSET))

        def _parse_skills(data: object) -> list[AgentSkillRef] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                skills_type_0 = []
                _skills_type_0 = data
                for skills_type_0_item_data in _skills_type_0:
                    skills_type_0_item = AgentSkillRef.from_dict(
                        skills_type_0_item_data
                    )

                    skills_type_0.append(skills_type_0_item)

                return skills_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[AgentSkillRef] | None | Unset, data)

        skills = _parse_skills(d.pop("skills", UNSET))

        def _parse_mcp_servers(data: object) -> list[McpServerSpec] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                mcp_servers_type_0 = []
                _mcp_servers_type_0 = data
                for mcp_servers_type_0_item_data in _mcp_servers_type_0:
                    mcp_servers_type_0_item = McpServerSpec.from_dict(
                        mcp_servers_type_0_item_data
                    )

                    mcp_servers_type_0.append(mcp_servers_type_0_item)

                return mcp_servers_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[McpServerSpec] | None | Unset, data)

        mcp_servers = _parse_mcp_servers(d.pop("mcp_servers", UNSET))

        def _parse_http_servers(data: object) -> list[HttpServerSpec] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                http_servers_type_0 = []
                _http_servers_type_0 = data
                for http_servers_type_0_item_data in _http_servers_type_0:
                    http_servers_type_0_item = HttpServerSpec.from_dict(
                        http_servers_type_0_item_data
                    )

                    http_servers_type_0.append(http_servers_type_0_item)

                return http_servers_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[HttpServerSpec] | None | Unset, data)

        http_servers = _parse_http_servers(d.pop("http_servers", UNSET))

        def _parse_description(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        description = _parse_description(d.pop("description", UNSET))

        def _parse_metadata(data: object) -> AgentUpdateMetadataType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                metadata_type_0 = AgentUpdateMetadataType0.from_dict(data)

                return metadata_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(AgentUpdateMetadataType0 | None | Unset, data)

        metadata = _parse_metadata(d.pop("metadata", UNSET))

        def _parse_litellm_extra(
            data: object,
        ) -> AgentUpdateLitellmExtraType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                litellm_extra_type_0 = AgentUpdateLitellmExtraType0.from_dict(data)

                return litellm_extra_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(AgentUpdateLitellmExtraType0 | None | Unset, data)

        litellm_extra = _parse_litellm_extra(d.pop("litellm_extra", UNSET))

        def _parse_window_min(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        window_min = _parse_window_min(d.pop("window_min", UNSET))

        def _parse_window_max(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        window_max = _parse_window_max(d.pop("window_max", UNSET))

        agent_update = cls(
            version=version,
            name=name,
            model=model,
            system=system,
            tools=tools,
            skills=skills,
            mcp_servers=mcp_servers,
            http_servers=http_servers,
            description=description,
            metadata=metadata,
            litellm_extra=litellm_extra,
            window_min=window_min,
            window_max=window_max,
        )

        return agent_update
