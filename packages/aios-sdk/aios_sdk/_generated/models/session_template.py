from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.session_template_metadata import SessionTemplateMetadata


T = TypeVar("T", bound="SessionTemplate")


@_attrs_define
class SessionTemplate:
    """Read view of a session template.

    Attributes:
        id (str):
        name (str):
        agent_id (str):
        agent_version (int | None):
        environment_id (str):
        vault_ids (list[str]):
        memory_store_ids (list[str]):
        metadata (SessionTemplateMetadata):
        created_at (datetime.datetime):
        updated_at (datetime.datetime):
        archive_when_idle (bool | Unset):  Default: False.
        archived_at (datetime.datetime | None | Unset):
    """

    id: str
    name: str
    agent_id: str
    agent_version: int | None
    environment_id: str
    vault_ids: list[str]
    memory_store_ids: list[str]
    metadata: SessionTemplateMetadata
    created_at: datetime.datetime
    updated_at: datetime.datetime
    archive_when_idle: bool | Unset = False
    archived_at: datetime.datetime | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        name = self.name

        agent_id = self.agent_id

        agent_version: int | None
        agent_version = self.agent_version

        environment_id = self.environment_id

        vault_ids = self.vault_ids

        memory_store_ids = self.memory_store_ids

        metadata = self.metadata.to_dict()

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        archive_when_idle = self.archive_when_idle

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
                "agent_id": agent_id,
                "agent_version": agent_version,
                "environment_id": environment_id,
                "vault_ids": vault_ids,
                "memory_store_ids": memory_store_ids,
                "metadata": metadata,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        if archive_when_idle is not UNSET:
            field_dict["archive_when_idle"] = archive_when_idle
        if archived_at is not UNSET:
            field_dict["archived_at"] = archived_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.session_template_metadata import SessionTemplateMetadata

        d = dict(src_dict)
        id = d.pop("id")

        name = d.pop("name")

        agent_id = d.pop("agent_id")

        def _parse_agent_version(data: object) -> int | None:
            if data is None:
                return data
            return cast(int | None, data)

        agent_version = _parse_agent_version(d.pop("agent_version"))

        environment_id = d.pop("environment_id")

        vault_ids = cast(list[str], d.pop("vault_ids"))

        memory_store_ids = cast(list[str], d.pop("memory_store_ids"))

        metadata = SessionTemplateMetadata.from_dict(d.pop("metadata"))

        created_at = isoparse(d.pop("created_at"))

        updated_at = isoparse(d.pop("updated_at"))

        archive_when_idle = d.pop("archive_when_idle", UNSET)

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

        session_template = cls(
            id=id,
            name=name,
            agent_id=agent_id,
            agent_version=agent_version,
            environment_id=environment_id,
            vault_ids=vault_ids,
            memory_store_ids=memory_store_ids,
            metadata=metadata,
            created_at=created_at,
            updated_at=updated_at,
            archive_when_idle=archive_when_idle,
            archived_at=archived_at,
        )

        session_template.additional_properties = d
        return session_template

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
