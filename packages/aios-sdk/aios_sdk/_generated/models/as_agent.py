from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

T = TypeVar("T", bound="AsAgent")


@_attrs_define
class AsAgent:
    """An agent version whose surface an operator re-rooted a sub-run to
    (``invoke_workflow``'s ``as_agent``), so an eval arm runs with the authority that
    agent would give it.

        Attributes:
            agent_id (str):
            version (int):
    """

    agent_id: str
    version: int

    def to_dict(self) -> dict[str, Any]:
        agent_id = self.agent_id

        version = self.version

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "agent_id": agent_id,
                "version": version,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        agent_id = d.pop("agent_id")

        version = d.pop("version")

        as_agent = cls(
            agent_id=agent_id,
            version=version,
        )

        return as_agent
