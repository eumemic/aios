from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..models.agent_create_preempt_policy import AgentCreatePreemptPolicy
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.agent_create_litellm_extra import AgentCreateLitellmExtra
    from ..models.agent_create_metadata import AgentCreateMetadata
    from ..models.agent_skill_ref import AgentSkillRef
    from ..models.http_server_spec import HttpServerSpec
    from ..models.mcp_server_spec import McpServerSpec
    from ..models.tool_spec import ToolSpec


T = TypeVar("T", bound="AgentCreate")


@_attrs_define
class AgentCreate:
    """Request body for `POST /v1/agents`.

    Attributes:
        name (str):
        model (str): LiteLLM model string, e.g. 'anthropic/claude-opus-4-6'.
        system (str | Unset): System prompt; empty by default. Default: ''.
        tools (list[ToolSpec] | Unset):
        skills (list[AgentSkillRef] | Unset):
        mcp_servers (list[McpServerSpec] | Unset):
        http_servers (list[HttpServerSpec] | Unset):
        description (None | str | Unset):
        metadata (AgentCreateMetadata | Unset):
        litellm_extra (AgentCreateLitellmExtra | Unset): Provider-specific LiteLLM kwargs merged into every model
            request for this agent.  Common shapes: OpenRouter ``extra_body.provider.order`` for provider pinning, Anthropic
            ``thinking``, OpenAI ``reasoning_effort``, raw sampling knobs (``temperature``, ``max_tokens``), ``api_base``
            for self-hosted inference.  Validated by LiteLLM / the provider; bad kwargs surface as tool-path errors the
            model sees.  Security: ``api_base`` redirects the model call — treat operator-set agents as trusted and don't
            accept this field from untrusted principals.
        window_min (int | Unset):  Default: 50000.
        window_max (int | Unset):  Default: 150000.
        preempt_policy (AgentCreatePreemptPolicy | Unset): Whether a new wake-eligible event (e.g. a user message)
            arriving mid-step cancels the in-flight model call so the step restarts against fresh context ('preempt'), or
            waits for the step to finish ('wait', default). Default: AgentCreatePreemptPolicy.WAIT.
    """

    name: str
    model: str
    system: str | Unset = ""
    tools: list[ToolSpec] | Unset = UNSET
    skills: list[AgentSkillRef] | Unset = UNSET
    mcp_servers: list[McpServerSpec] | Unset = UNSET
    http_servers: list[HttpServerSpec] | Unset = UNSET
    description: None | str | Unset = UNSET
    metadata: AgentCreateMetadata | Unset = UNSET
    litellm_extra: AgentCreateLitellmExtra | Unset = UNSET
    window_min: int | Unset = 50000
    window_max: int | Unset = 150000
    preempt_policy: AgentCreatePreemptPolicy | Unset = AgentCreatePreemptPolicy.WAIT

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        model = self.model

        system = self.system

        tools: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.tools, Unset):
            tools = []
            for tools_item_data in self.tools:
                tools_item = tools_item_data.to_dict()
                tools.append(tools_item)

        skills: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.skills, Unset):
            skills = []
            for skills_item_data in self.skills:
                skills_item = skills_item_data.to_dict()
                skills.append(skills_item)

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

        description: None | str | Unset
        if isinstance(self.description, Unset):
            description = UNSET
        else:
            description = self.description

        metadata: dict[str, Any] | Unset = UNSET
        if not isinstance(self.metadata, Unset):
            metadata = self.metadata.to_dict()

        litellm_extra: dict[str, Any] | Unset = UNSET
        if not isinstance(self.litellm_extra, Unset):
            litellm_extra = self.litellm_extra.to_dict()

        window_min = self.window_min

        window_max = self.window_max

        preempt_policy: str | Unset = UNSET
        if not isinstance(self.preempt_policy, Unset):
            preempt_policy = self.preempt_policy.value

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "name": name,
                "model": model,
            }
        )
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
        if preempt_policy is not UNSET:
            field_dict["preempt_policy"] = preempt_policy

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.agent_create_litellm_extra import AgentCreateLitellmExtra
        from ..models.agent_create_metadata import AgentCreateMetadata
        from ..models.agent_skill_ref import AgentSkillRef
        from ..models.http_server_spec import HttpServerSpec
        from ..models.mcp_server_spec import McpServerSpec
        from ..models.tool_spec import ToolSpec

        d = dict(src_dict)
        name = d.pop("name")

        model = d.pop("model")

        system = d.pop("system", UNSET)

        _tools = d.pop("tools", UNSET)
        tools: list[ToolSpec] | Unset = UNSET
        if _tools is not UNSET:
            tools = []
            for tools_item_data in _tools:
                tools_item = ToolSpec.from_dict(tools_item_data)

                tools.append(tools_item)

        _skills = d.pop("skills", UNSET)
        skills: list[AgentSkillRef] | Unset = UNSET
        if _skills is not UNSET:
            skills = []
            for skills_item_data in _skills:
                skills_item = AgentSkillRef.from_dict(skills_item_data)

                skills.append(skills_item)

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

        def _parse_description(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        description = _parse_description(d.pop("description", UNSET))

        _metadata = d.pop("metadata", UNSET)
        metadata: AgentCreateMetadata | Unset
        if isinstance(_metadata, Unset):
            metadata = UNSET
        else:
            metadata = AgentCreateMetadata.from_dict(_metadata)

        _litellm_extra = d.pop("litellm_extra", UNSET)
        litellm_extra: AgentCreateLitellmExtra | Unset
        if isinstance(_litellm_extra, Unset):
            litellm_extra = UNSET
        else:
            litellm_extra = AgentCreateLitellmExtra.from_dict(_litellm_extra)

        window_min = d.pop("window_min", UNSET)

        window_max = d.pop("window_max", UNSET)

        _preempt_policy = d.pop("preempt_policy", UNSET)
        preempt_policy: AgentCreatePreemptPolicy | Unset
        if isinstance(_preempt_policy, Unset):
            preempt_policy = UNSET
        else:
            preempt_policy = AgentCreatePreemptPolicy(_preempt_policy)

        agent_create = cls(
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
            preempt_policy=preempt_policy,
        )

        return agent_create
