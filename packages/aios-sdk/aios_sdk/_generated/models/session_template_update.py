from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.session_template_update_metadata_type_0 import (
        SessionTemplateUpdateMetadataType0,
    )


T = TypeVar("T", bound="SessionTemplateUpdate")


@_attrs_define
class SessionTemplateUpdate:
    """Request body for ``PUT /v1/session-templates/{id}``.

    Updates apply to future spawns only — already-spawned sessions are
    not retroactively migrated (see module docstring).

        Attributes:
            name (None | str | Unset):
            agent_id (None | str | Unset):
            agent_version (int | None | Unset):
            environment_id (None | str | Unset):
            vault_ids (list[str] | None | Unset):
            memory_store_ids (list[str] | None | Unset):
            metadata (None | SessionTemplateUpdateMetadataType0 | Unset):
            archive_when_idle (bool | None | Unset):
    """

    name: None | str | Unset = UNSET
    agent_id: None | str | Unset = UNSET
    agent_version: int | None | Unset = UNSET
    environment_id: None | str | Unset = UNSET
    vault_ids: list[str] | None | Unset = UNSET
    memory_store_ids: list[str] | None | Unset = UNSET
    metadata: None | SessionTemplateUpdateMetadataType0 | Unset = UNSET
    archive_when_idle: bool | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.session_template_update_metadata_type_0 import (
            SessionTemplateUpdateMetadataType0,
        )

        name: None | str | Unset
        if isinstance(self.name, Unset):
            name = UNSET
        else:
            name = self.name

        agent_id: None | str | Unset
        if isinstance(self.agent_id, Unset):
            agent_id = UNSET
        else:
            agent_id = self.agent_id

        agent_version: int | None | Unset
        if isinstance(self.agent_version, Unset):
            agent_version = UNSET
        else:
            agent_version = self.agent_version

        environment_id: None | str | Unset
        if isinstance(self.environment_id, Unset):
            environment_id = UNSET
        else:
            environment_id = self.environment_id

        vault_ids: list[str] | None | Unset
        if isinstance(self.vault_ids, Unset):
            vault_ids = UNSET
        elif isinstance(self.vault_ids, list):
            vault_ids = self.vault_ids

        else:
            vault_ids = self.vault_ids

        memory_store_ids: list[str] | None | Unset
        if isinstance(self.memory_store_ids, Unset):
            memory_store_ids = UNSET
        elif isinstance(self.memory_store_ids, list):
            memory_store_ids = self.memory_store_ids

        else:
            memory_store_ids = self.memory_store_ids

        metadata: dict[str, Any] | None | Unset
        if isinstance(self.metadata, Unset):
            metadata = UNSET
        elif isinstance(self.metadata, SessionTemplateUpdateMetadataType0):
            metadata = self.metadata.to_dict()
        else:
            metadata = self.metadata

        archive_when_idle: bool | None | Unset
        if isinstance(self.archive_when_idle, Unset):
            archive_when_idle = UNSET
        else:
            archive_when_idle = self.archive_when_idle

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if name is not UNSET:
            field_dict["name"] = name
        if agent_id is not UNSET:
            field_dict["agent_id"] = agent_id
        if agent_version is not UNSET:
            field_dict["agent_version"] = agent_version
        if environment_id is not UNSET:
            field_dict["environment_id"] = environment_id
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
        from ..models.session_template_update_metadata_type_0 import (
            SessionTemplateUpdateMetadataType0,
        )

        d = dict(src_dict)

        def _parse_name(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        name = _parse_name(d.pop("name", UNSET))

        def _parse_agent_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        agent_id = _parse_agent_id(d.pop("agent_id", UNSET))

        def _parse_agent_version(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        agent_version = _parse_agent_version(d.pop("agent_version", UNSET))

        def _parse_environment_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        environment_id = _parse_environment_id(d.pop("environment_id", UNSET))

        def _parse_vault_ids(data: object) -> list[str] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                vault_ids_type_0 = cast(list[str], data)

                return vault_ids_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None | Unset, data)

        vault_ids = _parse_vault_ids(d.pop("vault_ids", UNSET))

        def _parse_memory_store_ids(data: object) -> list[str] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                memory_store_ids_type_0 = cast(list[str], data)

                return memory_store_ids_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None | Unset, data)

        memory_store_ids = _parse_memory_store_ids(d.pop("memory_store_ids", UNSET))

        def _parse_metadata(
            data: object,
        ) -> None | SessionTemplateUpdateMetadataType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                metadata_type_0 = SessionTemplateUpdateMetadataType0.from_dict(data)

                return metadata_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | SessionTemplateUpdateMetadataType0 | Unset, data)

        metadata = _parse_metadata(d.pop("metadata", UNSET))

        def _parse_archive_when_idle(data: object) -> bool | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(bool | None | Unset, data)

        archive_when_idle = _parse_archive_when_idle(d.pop("archive_when_idle", UNSET))

        session_template_update = cls(
            name=name,
            agent_id=agent_id,
            agent_version=agent_version,
            environment_id=environment_id,
            vault_ids=vault_ids,
            memory_store_ids=memory_store_ids,
            metadata=metadata,
            archive_when_idle=archive_when_idle,
        )

        return session_template_update
