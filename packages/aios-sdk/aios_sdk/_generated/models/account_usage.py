from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="AccountUsage")


@_attrs_define
class AccountUsage:
    """Per-account resource counts as returned by ``GET /v1/accounts/{id}/usage``.

    Attributes:
        spent_usd (float):
        spend_limit_usd (float | None):
        input_tokens (int):
        output_tokens (int):
        cache_read_input_tokens (int):
        cache_creation_input_tokens (int):
        agents (int):
        environments (int):
        sessions (int):
        vaults (int):
        memory_stores (int):
        skills (int):
        session_templates (int):
        connections (int):
    """

    spent_usd: float
    spend_limit_usd: float | None
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    agents: int
    environments: int
    sessions: int
    vaults: int
    memory_stores: int
    skills: int
    session_templates: int
    connections: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        spent_usd = self.spent_usd

        spend_limit_usd: float | None
        spend_limit_usd = self.spend_limit_usd

        input_tokens = self.input_tokens

        output_tokens = self.output_tokens

        cache_read_input_tokens = self.cache_read_input_tokens

        cache_creation_input_tokens = self.cache_creation_input_tokens

        agents = self.agents

        environments = self.environments

        sessions = self.sessions

        vaults = self.vaults

        memory_stores = self.memory_stores

        skills = self.skills

        session_templates = self.session_templates

        connections = self.connections

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "spent_usd": spent_usd,
                "spend_limit_usd": spend_limit_usd,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": cache_read_input_tokens,
                "cache_creation_input_tokens": cache_creation_input_tokens,
                "agents": agents,
                "environments": environments,
                "sessions": sessions,
                "vaults": vaults,
                "memory_stores": memory_stores,
                "skills": skills,
                "session_templates": session_templates,
                "connections": connections,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        spent_usd = d.pop("spent_usd")

        def _parse_spend_limit_usd(data: object) -> float | None:
            if data is None:
                return data
            return cast(float | None, data)

        spend_limit_usd = _parse_spend_limit_usd(d.pop("spend_limit_usd"))

        input_tokens = d.pop("input_tokens")

        output_tokens = d.pop("output_tokens")

        cache_read_input_tokens = d.pop("cache_read_input_tokens")

        cache_creation_input_tokens = d.pop("cache_creation_input_tokens")

        agents = d.pop("agents")

        environments = d.pop("environments")

        sessions = d.pop("sessions")

        vaults = d.pop("vaults")

        memory_stores = d.pop("memory_stores")

        skills = d.pop("skills")

        session_templates = d.pop("session_templates")

        connections = d.pop("connections")

        account_usage = cls(
            spent_usd=spent_usd,
            spend_limit_usd=spend_limit_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            agents=agents,
            environments=environments,
            sessions=sessions,
            vaults=vaults,
            memory_stores=memory_stores,
            skills=skills,
            session_templates=session_templates,
            connections=connections,
        )

        account_usage.additional_properties = d
        return account_usage

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
