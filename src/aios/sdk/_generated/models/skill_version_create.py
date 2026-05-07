from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define

if TYPE_CHECKING:
    from ..models.skill_version_create_files import SkillVersionCreateFiles


T = TypeVar("T", bound="SkillVersionCreate")


@_attrs_define
class SkillVersionCreate:
    """Request body for ``POST /v1/skills/{skill_id}/versions``.

    Same file format as :class:`SkillCreate`. The directory, name, and
    description are re-extracted from the new SKILL.md.

        Attributes:
            files (SkillVersionCreateFiles):
    """

    files: SkillVersionCreateFiles

    def to_dict(self) -> dict[str, Any]:
        files = self.files.to_dict()

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "files": files,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.skill_version_create_files import SkillVersionCreateFiles

        d = dict(src_dict)
        files = SkillVersionCreateFiles.from_dict(d.pop("files"))

        skill_version_create = cls(
            files=files,
        )

        return skill_version_create
