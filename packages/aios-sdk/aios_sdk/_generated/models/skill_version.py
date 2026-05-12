from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

if TYPE_CHECKING:
    from ..models.skill_version_files import SkillVersionFiles


T = TypeVar("T", bound="SkillVersion")


@_attrs_define
class SkillVersion:
    """Read view of a specific skill version.

    Attributes:
        skill_id (str):
        version (int):
        directory (str):
        name (str):
        description (str):
        files (SkillVersionFiles):
        created_at (datetime.datetime):
    """

    skill_id: str
    version: int
    directory: str
    name: str
    description: str
    files: SkillVersionFiles
    created_at: datetime.datetime
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        skill_id = self.skill_id

        version = self.version

        directory = self.directory

        name = self.name

        description = self.description

        files = self.files.to_dict()

        created_at = self.created_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "skill_id": skill_id,
                "version": version,
                "directory": directory,
                "name": name,
                "description": description,
                "files": files,
                "created_at": created_at,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.skill_version_files import SkillVersionFiles

        d = dict(src_dict)
        skill_id = d.pop("skill_id")

        version = d.pop("version")

        directory = d.pop("directory")

        name = d.pop("name")

        description = d.pop("description")

        files = SkillVersionFiles.from_dict(d.pop("files"))

        created_at = isoparse(d.pop("created_at"))

        skill_version = cls(
            skill_id=skill_id,
            version=version,
            directory=directory,
            name=name,
            description=description,
            files=files,
            created_at=created_at,
        )

        skill_version.additional_properties = d
        return skill_version

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
