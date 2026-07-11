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
    from ..models.environment_config import EnvironmentConfig


T = TypeVar("T", bound="Environment")


@_attrs_define
class Environment:
    """Read view of an environment.

    Attributes:
        id (str):
        name (str):
        config (EnvironmentConfig): Container configuration for an environment.
        created_at (datetime.datetime):
        created_by (Actor | None | Unset):
        archived_at (datetime.datetime | None | Unset):
    """

    id: str
    name: str
    config: EnvironmentConfig
    created_at: datetime.datetime
    created_by: Actor | None | Unset = UNSET
    archived_at: datetime.datetime | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.actor import Actor

        id = self.id

        name = self.name

        config = self.config.to_dict()

        created_at = self.created_at.isoformat()

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
                "name": name,
                "config": config,
                "created_at": created_at,
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
        from ..models.environment_config import EnvironmentConfig

        d = dict(src_dict)
        id = d.pop("id")

        name = d.pop("name")

        config = EnvironmentConfig.from_dict(d.pop("config"))

        created_at = isoparse(d.pop("created_at"))

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

        environment = cls(
            id=id,
            name=name,
            config=config,
            created_at=created_at,
            created_by=created_by,
            archived_at=archived_at,
        )

        environment.additional_properties = d
        return environment

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
