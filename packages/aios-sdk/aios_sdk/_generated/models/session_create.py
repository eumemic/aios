from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.github_repository_resource import GithubRepositoryResource
    from ..models.memory_store_resource import MemoryStoreResource
    from ..models.session_create_env import SessionCreateEnv
    from ..models.session_create_metadata import SessionCreateMetadata
    from ..models.trigger_create import TriggerCreate


T = TypeVar("T", bound="SessionCreate")


@_attrs_define
class SessionCreate:
    """Request body for `POST /v1/sessions`.

    Attributes:
        agent_id (str):
        environment_id (str):
        agent_version (int | None | Unset): Pin to a specific agent version. Omit or pass null for 'latest' (auto-
            updating — the session uses whatever version is current).
        title (None | str | Unset):
        metadata (SessionCreateMetadata | Unset):
        vault_ids (list[str] | Unset): Vault ids to bind to this session for MCP credential resolution.
        archive_when_idle (bool | Unset): When true, the session is soft-archived the first time it goes idle owing
            nothing — a self-reclaiming, one-shot session. Immutable after launch. Workflow agent() children launch with
            this set. Default: False.
        workspace_path (None | str | Unset): Absolute host path to use as the session workspace. If omitted, defaults to
            workspace_root/<account_id>/<session_id>. Must resolve within the account's workspace subdirectory. The
            directory must exist; aios will not create it.
        env (SessionCreateEnv | Unset): Environment variables injected into the sandbox container. A vaulted
            environment_variable credential whose secret_name matches a key here takes precedence: that key resolves to the
            credential's opaque placeholder, not the value set here.
        initial_message (None | str | Unset): Convenience: when set, the server appends a user.message event with this
            content immediately after creating the session and enqueues a wake job. Equivalent to a follow-up POST
            /messages.
        resources (list[GithubRepositoryResource | MemoryStoreResource] | Unset): Resources to attach. Mix of memory
            stores (mounted under /mnt/memory/<name>/) and github repositories (cloned to a user-specified mount_path). Each
            type has its own per-session cap; duplicates within a type are rejected. After creation, prefer the granular
            sub-collection endpoints — ``POST /v1/sessions/{id}/resources`` to attach one resource without touching the
            others, and ``DELETE /v1/sessions/{id}/resources/{resource_id}`` to detach one. ``PUT /v1/sessions/{id}`` with
            ``resources`` replaces the WHOLE list (omitting a resource detaches it), so the granular endpoints are the safe
            path for add/remove (#270).
        triggers (list[TriggerCreate] | Unset): Triggers attached at session creation. Each pairs a ``source`` (cron /
            one_shot) with an ``action`` (a ``sandbox_command`` bash task that fires without waking the model, or a
            ``wake_owner`` message that wakes it). Manage after creation via ``POST/DELETE/PUT /v1/sessions/{id}/triggers``;
            ``SessionUpdate`` deliberately does not accept this field (granular ops only).
    """

    agent_id: str
    environment_id: str
    agent_version: int | None | Unset = UNSET
    title: None | str | Unset = UNSET
    metadata: SessionCreateMetadata | Unset = UNSET
    vault_ids: list[str] | Unset = UNSET
    archive_when_idle: bool | Unset = False
    workspace_path: None | str | Unset = UNSET
    env: SessionCreateEnv | Unset = UNSET
    initial_message: None | str | Unset = UNSET
    resources: list[GithubRepositoryResource | MemoryStoreResource] | Unset = UNSET
    triggers: list[TriggerCreate] | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.memory_store_resource import MemoryStoreResource

        agent_id = self.agent_id

        environment_id = self.environment_id

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

        metadata: dict[str, Any] | Unset = UNSET
        if not isinstance(self.metadata, Unset):
            metadata = self.metadata.to_dict()

        vault_ids: list[str] | Unset = UNSET
        if not isinstance(self.vault_ids, Unset):
            vault_ids = self.vault_ids

        archive_when_idle = self.archive_when_idle

        workspace_path: None | str | Unset
        if isinstance(self.workspace_path, Unset):
            workspace_path = UNSET
        else:
            workspace_path = self.workspace_path

        env: dict[str, Any] | Unset = UNSET
        if not isinstance(self.env, Unset):
            env = self.env.to_dict()

        initial_message: None | str | Unset
        if isinstance(self.initial_message, Unset):
            initial_message = UNSET
        else:
            initial_message = self.initial_message

        resources: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.resources, Unset):
            resources = []
            for resources_item_data in self.resources:
                resources_item: dict[str, Any]
                if isinstance(resources_item_data, MemoryStoreResource):
                    resources_item = resources_item_data.to_dict()
                else:
                    resources_item = resources_item_data.to_dict()

                resources.append(resources_item)

        triggers: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.triggers, Unset):
            triggers = []
            for triggers_item_data in self.triggers:
                triggers_item = triggers_item_data.to_dict()
                triggers.append(triggers_item)

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "agent_id": agent_id,
                "environment_id": environment_id,
            }
        )
        if agent_version is not UNSET:
            field_dict["agent_version"] = agent_version
        if title is not UNSET:
            field_dict["title"] = title
        if metadata is not UNSET:
            field_dict["metadata"] = metadata
        if vault_ids is not UNSET:
            field_dict["vault_ids"] = vault_ids
        if archive_when_idle is not UNSET:
            field_dict["archive_when_idle"] = archive_when_idle
        if workspace_path is not UNSET:
            field_dict["workspace_path"] = workspace_path
        if env is not UNSET:
            field_dict["env"] = env
        if initial_message is not UNSET:
            field_dict["initial_message"] = initial_message
        if resources is not UNSET:
            field_dict["resources"] = resources
        if triggers is not UNSET:
            field_dict["triggers"] = triggers

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.github_repository_resource import GithubRepositoryResource
        from ..models.memory_store_resource import MemoryStoreResource
        from ..models.session_create_env import SessionCreateEnv
        from ..models.session_create_metadata import SessionCreateMetadata
        from ..models.trigger_create import TriggerCreate

        d = dict(src_dict)
        agent_id = d.pop("agent_id")

        environment_id = d.pop("environment_id")

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

        _metadata = d.pop("metadata", UNSET)
        metadata: SessionCreateMetadata | Unset
        if isinstance(_metadata, Unset):
            metadata = UNSET
        else:
            metadata = SessionCreateMetadata.from_dict(_metadata)

        vault_ids = cast(list[str], d.pop("vault_ids", UNSET))

        archive_when_idle = d.pop("archive_when_idle", UNSET)

        def _parse_workspace_path(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        workspace_path = _parse_workspace_path(d.pop("workspace_path", UNSET))

        _env = d.pop("env", UNSET)
        env: SessionCreateEnv | Unset
        if isinstance(_env, Unset):
            env = UNSET
        else:
            env = SessionCreateEnv.from_dict(_env)

        def _parse_initial_message(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        initial_message = _parse_initial_message(d.pop("initial_message", UNSET))

        _resources = d.pop("resources", UNSET)
        resources: list[GithubRepositoryResource | MemoryStoreResource] | Unset = UNSET
        if _resources is not UNSET:
            resources = []
            for resources_item_data in _resources:

                def _parse_resources_item(
                    data: object,
                ) -> GithubRepositoryResource | MemoryStoreResource:
                    try:
                        if not isinstance(data, dict):
                            raise TypeError()
                        resources_item_type_0 = MemoryStoreResource.from_dict(data)

                        return resources_item_type_0
                    except (TypeError, ValueError, AttributeError, KeyError):
                        pass
                    if not isinstance(data, dict):
                        raise TypeError()
                    resources_item_type_1 = GithubRepositoryResource.from_dict(data)

                    return resources_item_type_1

                resources_item = _parse_resources_item(resources_item_data)

                resources.append(resources_item)

        _triggers = d.pop("triggers", UNSET)
        triggers: list[TriggerCreate] | Unset = UNSET
        if _triggers is not UNSET:
            triggers = []
            for triggers_item_data in _triggers:
                triggers_item = TriggerCreate.from_dict(triggers_item_data)

                triggers.append(triggers_item)

        session_create = cls(
            agent_id=agent_id,
            environment_id=environment_id,
            agent_version=agent_version,
            title=title,
            metadata=metadata,
            vault_ids=vault_ids,
            archive_when_idle=archive_when_idle,
            workspace_path=workspace_path,
            env=env,
            initial_message=initial_message,
            resources=resources,
            triggers=triggers,
        )

        return session_create
