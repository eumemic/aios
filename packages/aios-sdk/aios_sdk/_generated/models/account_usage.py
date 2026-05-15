from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="AccountUsage")


@_attrs_define
class AccountUsage:
    """Per-account resource counts as returned by ``GET /v1/accounts/{id}/usage``.

    Attributes:
        agents (int):
        environments (int):
        sessions (int):
        vaults (int):
        memory_stores (int):
        skills (int):
        session_templates (int):
        connections (int):
    """

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
        agents = d.pop("agents")

        environments = d.pop("environments")

        sessions = d.pop("sessions")

        vaults = d.pop("vaults")

        memory_stores = d.pop("memory_stores")

        skills = d.pop("skills")

        session_templates = d.pop("session_templates")

        connections = d.pop("connections")

        account_usage = cls(
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
