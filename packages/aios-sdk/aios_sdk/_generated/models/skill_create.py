from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define

if TYPE_CHECKING:
    from ..models.skill_create_files import SkillCreateFiles


T = TypeVar("T", bound="SkillCreate")


@_attrs_define
class SkillCreate:
    """Request body for ``POST /v1/skills``.

    ``files`` must include exactly one ``{directory}/SKILL.md`` entry.
    The server extracts ``name``, ``description``, and ``directory`` from
    the SKILL.md frontmatter and file paths.

        Attributes:
            display_title (str):
            files (SkillCreateFiles): Skill files as {path: content}. Must include exactly one {directory}/SKILL.md entry
                with YAML frontmatter.
    """

    display_title: str
    files: SkillCreateFiles

    def to_dict(self) -> dict[str, Any]:
        display_title = self.display_title

        files = self.files.to_dict()

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "display_title": display_title,
                "files": files,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.skill_create_files import SkillCreateFiles

        d = dict(src_dict)
        display_title = d.pop("display_title")

        files = SkillCreateFiles.from_dict(d.pop("files"))

        skill_create = cls(
            display_title=display_title,
            files=files,
        )

        return skill_create
