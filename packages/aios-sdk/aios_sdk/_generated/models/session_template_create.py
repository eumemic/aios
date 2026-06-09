from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.session_template_create_metadata import SessionTemplateCreateMetadata


T = TypeVar("T", bound="SessionTemplateCreate")


@_attrs_define
class SessionTemplateCreate:
    """Request body for ``POST /v1/session-templates``.

    Attributes:
        name (str):
        agent_id (str):
        environment_id (str):
        agent_version (int | None | Unset): Pin to a specific agent version. Omit or pass null for 'latest' — the spawn
            captures whatever version is current at spawn time.
        vault_ids (list[str] | Unset):
        memory_store_ids (list[str] | Unset):
        metadata (SessionTemplateCreateMetadata | Unset):
        archive_when_idle (bool | Unset): Copied down to every session this template spawns: when true, each spawned
            session self-archives the first time it goes idle. Default: False.
    """

    name: str
    agent_id: str
    environment_id: str
    agent_version: int | None | Unset = UNSET
    vault_ids: list[str] | Unset = UNSET
    memory_store_ids: list[str] | Unset = UNSET
    metadata: SessionTemplateCreateMetadata | Unset = UNSET
    archive_when_idle: bool | Unset = False

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        agent_id = self.agent_id

        environment_id = self.environment_id

        agent_version: int | None | Unset
        if isinstance(self.agent_version, Unset):
            agent_version = UNSET
        else:
            agent_version = self.agent_version

        vault_ids: list[str] | Unset = UNSET
        if not isinstance(self.vault_ids, Unset):
            vault_ids = self.vault_ids

        memory_store_ids: list[str] | Unset = UNSET
        if not isinstance(self.memory_store_ids, Unset):
            memory_store_ids = self.memory_store_ids

        metadata: dict[str, Any] | Unset = UNSET
        if not isinstance(self.metadata, Unset):
            metadata = self.metadata.to_dict()

        archive_when_idle = self.archive_when_idle

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "name": name,
                "agent_id": agent_id,
                "environment_id": environment_id,
            }
        )
        if agent_version is not UNSET:
            field_dict["agent_version"] = agent_version
        if vault_ids is not UNSET:
            field_dict["vault_ids"] = vault_ids
        if memory_store_ids is not UNSET:
            field_dict["memory_store_ids"] = memory_store_ids
        if metadata is not UNSET:
            field_dict["metadata"] = metadata
        if archive_when_idle is not UNSET:
            field_dict["archive_when_idle"] = archive_when_idle

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.session_template_create_metadata import (
            SessionTemplateCreateMetadata,
        )

        d = dict(src_dict)
        name = d.pop("name")

        agent_id = d.pop("agent_id")

        environment_id = d.pop("environment_id")

        def _parse_agent_version(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        agent_version = _parse_agent_version(d.pop("agent_version", UNSET))

        vault_ids = cast(list[str], d.pop("vault_ids", UNSET))

        memory_store_ids = cast(list[str], d.pop("memory_store_ids", UNSET))

        _metadata = d.pop("metadata", UNSET)
        metadata: SessionTemplateCreateMetadata | Unset
        if isinstance(_metadata, Unset):
            metadata = UNSET
        else:
            metadata = SessionTemplateCreateMetadata.from_dict(_metadata)

        archive_when_idle = d.pop("archive_when_idle", UNSET)

        session_template_create = cls(
            name=name,
            agent_id=agent_id,
            environment_id=environment_id,
            agent_version=agent_version,
            vault_ids=vault_ids,
            memory_store_ids=memory_store_ids,
            metadata=metadata,
            archive_when_idle=archive_when_idle,
        )

        return session_template_create
