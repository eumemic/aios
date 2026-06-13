from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.github_repository_resource import GithubRepositoryResource
    from ..models.memory_store_resource import MemoryStoreResource
    from ..models.session_update_metadata_type_0 import SessionUpdateMetadataType0


T = TypeVar("T", bound="SessionUpdate")


@_attrs_define
class SessionUpdate:
    """Request body for ``PUT /v1/sessions/{id}``.

    All fields are optional; omitted fields are preserved. Changing
    ``agent_id`` resets ``agent_version`` to null (latest) unless
    ``agent_version`` is also provided. ``resources`` and ``vault_ids``
    use full-list-replacement semantics: ``None`` (the default) leaves
    the current set alone, ``[]`` detaches everything, and a non-empty
    list replaces the bound set entirely.

    To add or remove a SINGLE resource without re-supplying the rest of
    the list, use the granular sub-collection endpoints —
    ``POST /v1/sessions/{id}/resources`` (attach one) and
    ``DELETE /v1/sessions/{id}/resources/{resource_id}`` (detach one).
    A one-resource ``resources`` list here silently detaches everything
    else; the granular endpoints are the safe add/remove path (#270).

        Attributes:
            agent_id (None | str | Unset):
            agent_version (int | None | Unset):
            title (None | str | Unset):
            metadata (None | SessionUpdateMetadataType0 | Unset):
            vault_ids (list[str] | None | Unset):
            resources (list[GithubRepositoryResource | MemoryStoreResource] | None | Unset):
    """

    agent_id: None | str | Unset = UNSET
    agent_version: int | None | Unset = UNSET
    title: None | str | Unset = UNSET
    metadata: None | SessionUpdateMetadataType0 | Unset = UNSET
    vault_ids: list[str] | None | Unset = UNSET
    resources: list[GithubRepositoryResource | MemoryStoreResource] | None | Unset = (
        UNSET
    )

    def to_dict(self) -> dict[str, Any]:
        from ..models.memory_store_resource import MemoryStoreResource
        from ..models.session_update_metadata_type_0 import SessionUpdateMetadataType0

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

        title: None | str | Unset
        if isinstance(self.title, Unset):
            title = UNSET
        else:
            title = self.title

        metadata: dict[str, Any] | None | Unset
        if isinstance(self.metadata, Unset):
            metadata = UNSET
        elif isinstance(self.metadata, SessionUpdateMetadataType0):
            metadata = self.metadata.to_dict()
        else:
            metadata = self.metadata

        vault_ids: list[str] | None | Unset
        if isinstance(self.vault_ids, Unset):
            vault_ids = UNSET
        elif isinstance(self.vault_ids, list):
            vault_ids = self.vault_ids

        else:
            vault_ids = self.vault_ids

        resources: list[dict[str, Any]] | None | Unset
        if isinstance(self.resources, Unset):
            resources = UNSET
        elif isinstance(self.resources, list):
            resources = []
            for resources_type_0_item_data in self.resources:
                resources_type_0_item: dict[str, Any]
                if isinstance(resources_type_0_item_data, MemoryStoreResource):
                    resources_type_0_item = resources_type_0_item_data.to_dict()
                else:
                    resources_type_0_item = resources_type_0_item_data.to_dict()

                resources.append(resources_type_0_item)

        else:
            resources = self.resources

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if agent_id is not UNSET:
            field_dict["agent_id"] = agent_id
        if agent_version is not UNSET:
            field_dict["agent_version"] = agent_version
        if title is not UNSET:
            field_dict["title"] = title
        if metadata is not UNSET:
            field_dict["metadata"] = metadata
        if vault_ids is not UNSET:
            field_dict["vault_ids"] = vault_ids
        if resources is not UNSET:
            field_dict["resources"] = resources

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.github_repository_resource import GithubRepositoryResource
        from ..models.memory_store_resource import MemoryStoreResource
        from ..models.session_update_metadata_type_0 import SessionUpdateMetadataType0

        d = dict(src_dict)

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

        def _parse_title(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        title = _parse_title(d.pop("title", UNSET))

        def _parse_metadata(data: object) -> None | SessionUpdateMetadataType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                metadata_type_0 = SessionUpdateMetadataType0.from_dict(data)

                return metadata_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | SessionUpdateMetadataType0 | Unset, data)

        metadata = _parse_metadata(d.pop("metadata", UNSET))

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

        def _parse_resources(
            data: object,
        ) -> list[GithubRepositoryResource | MemoryStoreResource] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                resources_type_0 = []
                _resources_type_0 = data
                for resources_type_0_item_data in _resources_type_0:

                    def _parse_resources_type_0_item(
                        data: object,
                    ) -> GithubRepositoryResource | MemoryStoreResource:
                        try:
                            if not isinstance(data, dict):
                                raise TypeError()
                            resources_type_0_item_type_0 = (
                                MemoryStoreResource.from_dict(data)
                            )

                            return resources_type_0_item_type_0
                        except (TypeError, ValueError, AttributeError, KeyError):
                            pass
                        if not isinstance(data, dict):
                            raise TypeError()
                        resources_type_0_item_type_1 = (
                            GithubRepositoryResource.from_dict(data)
                        )

                        return resources_type_0_item_type_1

                    resources_type_0_item = _parse_resources_type_0_item(
                        resources_type_0_item_data
                    )

                    resources_type_0.append(resources_type_0_item)

                return resources_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(
                list[GithubRepositoryResource | MemoryStoreResource] | None | Unset,
                data,
            )

        resources = _parse_resources(d.pop("resources", UNSET))

        session_update = cls(
            agent_id=agent_id,
            agent_version=agent_version,
            title=title,
            metadata=metadata,
            vault_ids=vault_ids,
            resources=resources,
        )

        return session_update
