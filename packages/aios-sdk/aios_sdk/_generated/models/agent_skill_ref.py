from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="AgentSkillRef")


@_attrs_define
class AgentSkillRef:
    """One entry in an agent's ``skills`` list.

    ``version`` is ``None`` for "latest" (auto-updating — the session uses
    whatever version is current at step time). When an agent version is
    snapshotted, null versions are resolved to the concrete latest version
    at that moment.

        Attributes:
            skill_id (str):
            version (int | None | Unset): Pin to a specific version. Omit or null for latest.
    """

    skill_id: str
    version: int | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        skill_id = self.skill_id

        version: int | None | Unset
        if isinstance(self.version, Unset):
            version = UNSET
        else:
            version = self.version

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "skill_id": skill_id,
            }
        )
        if version is not UNSET:
            field_dict["version"] = version

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        skill_id = d.pop("skill_id")

        def _parse_version(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        version = _parse_version(d.pop("version", UNSET))

        agent_skill_ref = cls(
            skill_id=skill_id,
            version=version,
        )

        return agent_skill_ref
