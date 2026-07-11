from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.actor import Actor


T = TypeVar("T", bound="Skill")


@_attrs_define
class Skill:
    """Read view of a skill (head row).

    Attributes:
        id (str):
        display_title (str):
        latest_version (int):
        created_at (datetime.datetime):
        updated_at (datetime.datetime):
        created_by (Actor | None | Unset):
        archived_at (datetime.datetime | None | Unset):
    """

    id: str
    display_title: str
    latest_version: int
    created_at: datetime.datetime
    updated_at: datetime.datetime
    created_by: Actor | None | Unset = UNSET
    archived_at: datetime.datetime | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.actor import Actor

        id = self.id

        display_title = self.display_title

        latest_version = self.latest_version

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        created_by: dict[str, Any] | None | Unset
        if isinstance(self.created_by, Unset):
            created_by = UNSET
        elif isinstance(self.created_by, Actor):
            created_by = self.created_by.to_dict()
        else:
            created_by = self.created_by

        archived_at: None | str | Unset
        if isinstance(self.archived_at, Unset):
            archived_at = UNSET
        elif isinstance(self.archived_at, datetime.datetime):
            archived_at = self.archived_at.isoformat()
        else:
            archived_at = self.archived_at

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "display_title": display_title,
                "latest_version": latest_version,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        if created_by is not UNSET:
            field_dict["created_by"] = created_by
        if archived_at is not UNSET:
            field_dict["archived_at"] = archived_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.actor import Actor

        d = dict(src_dict)
        id = d.pop("id")

        display_title = d.pop("display_title")

        latest_version = d.pop("latest_version")

        created_at = isoparse(d.pop("created_at"))

        updated_at = isoparse(d.pop("updated_at"))

        def _parse_created_by(data: object) -> Actor | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                created_by_type_0 = Actor.from_dict(data)

                return created_by_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(Actor | None | Unset, data)

        created_by = _parse_created_by(d.pop("created_by", UNSET))

        def _parse_archived_at(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                archived_at_type_0 = isoparse(data)

                return archived_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        archived_at = _parse_archived_at(d.pop("archived_at", UNSET))

        skill = cls(
            id=id,
            display_title=display_title,
            latest_version=latest_version,
            created_at=created_at,
            updated_at=updated_at,
            created_by=created_by,
            archived_at=archived_at,
        )

        skill.additional_properties = d
        return skill

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
