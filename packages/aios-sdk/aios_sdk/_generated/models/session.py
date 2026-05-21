from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.session_status import SessionStatus
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.awaiting_tool_call import AwaitingToolCall
    from ..models.github_repository_resource_echo import GithubRepositoryResourceEcho
    from ..models.memory_store_resource_echo import MemoryStoreResourceEcho
    from ..models.session_metadata import SessionMetadata
    from ..models.session_stop_reason_type_0 import SessionStopReasonType0
    from ..models.session_usage import SessionUsage


T = TypeVar("T", bound="Session")


@_attrs_define
class Session:
    """Read view of a session. Internal-only columns are not exposed.

    ``stop_reason`` records why the most recent step ended. Possible
    ``type`` values: ``"end_turn"``, ``"interrupt"``, ``"rescheduling"``,
    ``"error"``. ``awaiting`` lists tool calls the session is blocked
    on (derived per read from the event log + agent tool specs).

        Attributes:
            id (str):
            agent_id (str):
            environment_id (str):
            agent_version (int | None):
            title (None | str):
            metadata (SessionMetadata):
            status (SessionStatus):
            stop_reason (None | SessionStopReasonType0):
            last_event_seq (int):
            created_at (datetime.datetime):
            updated_at (datetime.datetime):
            awaiting (list[AwaitingToolCall] | Unset):
            vault_ids (list[str] | Unset):
            usage (SessionUsage | Unset): Cumulative token usage across all model calls in a session.
            resources (list[GithubRepositoryResourceEcho | MemoryStoreResourceEcho] | Unset):
            archived_at (datetime.datetime | None | Unset):
            focal_channel (None | str | Unset):
            focal_locked (bool | Unset):  Default: False.
            last_event_at (datetime.datetime | None | Unset):
            total_events (int | Unset):  Default: 0.
    """

    id: str
    agent_id: str
    environment_id: str
    agent_version: int | None
    title: None | str
    metadata: SessionMetadata
    status: SessionStatus
    stop_reason: None | SessionStopReasonType0
    last_event_seq: int
    created_at: datetime.datetime
    updated_at: datetime.datetime
    awaiting: list[AwaitingToolCall] | Unset = UNSET
    vault_ids: list[str] | Unset = UNSET
    usage: SessionUsage | Unset = UNSET
    resources: list[GithubRepositoryResourceEcho | MemoryStoreResourceEcho] | Unset = (
        UNSET
    )
    archived_at: datetime.datetime | None | Unset = UNSET
    focal_channel: None | str | Unset = UNSET
    focal_locked: bool | Unset = False
    last_event_at: datetime.datetime | None | Unset = UNSET
    total_events: int | Unset = 0
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.memory_store_resource_echo import MemoryStoreResourceEcho
        from ..models.session_stop_reason_type_0 import SessionStopReasonType0

        id = self.id

        agent_id = self.agent_id

        environment_id = self.environment_id

        agent_version: int | None
        agent_version = self.agent_version

        title: None | str
        title = self.title

        metadata = self.metadata.to_dict()

        status = self.status.value

        stop_reason: dict[str, Any] | None
        if isinstance(self.stop_reason, SessionStopReasonType0):
            stop_reason = self.stop_reason.to_dict()
        else:
            stop_reason = self.stop_reason

        last_event_seq = self.last_event_seq

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        awaiting: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.awaiting, Unset):
            awaiting = []
            for awaiting_item_data in self.awaiting:
                awaiting_item = awaiting_item_data.to_dict()
                awaiting.append(awaiting_item)

        vault_ids: list[str] | Unset = UNSET
        if not isinstance(self.vault_ids, Unset):
            vault_ids = self.vault_ids

        usage: dict[str, Any] | Unset = UNSET
        if not isinstance(self.usage, Unset):
            usage = self.usage.to_dict()

        resources: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.resources, Unset):
            resources = []
            for resources_item_data in self.resources:
                resources_item: dict[str, Any]
                if isinstance(resources_item_data, MemoryStoreResourceEcho):
                    resources_item = resources_item_data.to_dict()
                else:
                    resources_item = resources_item_data.to_dict()

                resources.append(resources_item)

        archived_at: None | str | Unset
        if isinstance(self.archived_at, Unset):
            archived_at = UNSET
        elif isinstance(self.archived_at, datetime.datetime):
            archived_at = self.archived_at.isoformat()
        else:
            archived_at = self.archived_at

        focal_channel: None | str | Unset
        if isinstance(self.focal_channel, Unset):
            focal_channel = UNSET
        else:
            focal_channel = self.focal_channel

        focal_locked = self.focal_locked

        last_event_at: None | str | Unset
        if isinstance(self.last_event_at, Unset):
            last_event_at = UNSET
        elif isinstance(self.last_event_at, datetime.datetime):
            last_event_at = self.last_event_at.isoformat()
        else:
            last_event_at = self.last_event_at

        total_events = self.total_events

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "agent_id": agent_id,
                "environment_id": environment_id,
                "agent_version": agent_version,
                "title": title,
                "metadata": metadata,
                "status": status,
                "stop_reason": stop_reason,
                "last_event_seq": last_event_seq,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        if awaiting is not UNSET:
            field_dict["awaiting"] = awaiting
        if vault_ids is not UNSET:
            field_dict["vault_ids"] = vault_ids
        if usage is not UNSET:
            field_dict["usage"] = usage
        if resources is not UNSET:
            field_dict["resources"] = resources
        if archived_at is not UNSET:
            field_dict["archived_at"] = archived_at
        if focal_channel is not UNSET:
            field_dict["focal_channel"] = focal_channel
        if focal_locked is not UNSET:
            field_dict["focal_locked"] = focal_locked
        if last_event_at is not UNSET:
            field_dict["last_event_at"] = last_event_at
        if total_events is not UNSET:
            field_dict["total_events"] = total_events

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.awaiting_tool_call import AwaitingToolCall
        from ..models.github_repository_resource_echo import (
            GithubRepositoryResourceEcho,
        )
        from ..models.memory_store_resource_echo import MemoryStoreResourceEcho
        from ..models.session_metadata import SessionMetadata
        from ..models.session_stop_reason_type_0 import SessionStopReasonType0
        from ..models.session_usage import SessionUsage

        d = dict(src_dict)
        id = d.pop("id")

        agent_id = d.pop("agent_id")

        environment_id = d.pop("environment_id")

        def _parse_agent_version(data: object) -> int | None:
            if data is None:
                return data
            return cast(int | None, data)

        agent_version = _parse_agent_version(d.pop("agent_version"))

        def _parse_title(data: object) -> None | str:
            if data is None:
                return data
            return cast(None | str, data)

        title = _parse_title(d.pop("title"))

        metadata = SessionMetadata.from_dict(d.pop("metadata"))

        status = SessionStatus(d.pop("status"))

        def _parse_stop_reason(data: object) -> None | SessionStopReasonType0:
            if data is None:
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                stop_reason_type_0 = SessionStopReasonType0.from_dict(data)

                return stop_reason_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | SessionStopReasonType0, data)

        stop_reason = _parse_stop_reason(d.pop("stop_reason"))

        last_event_seq = d.pop("last_event_seq")

        created_at = isoparse(d.pop("created_at"))

        updated_at = isoparse(d.pop("updated_at"))

        _awaiting = d.pop("awaiting", UNSET)
        awaiting: list[AwaitingToolCall] | Unset = UNSET
        if _awaiting is not UNSET:
            awaiting = []
            for awaiting_item_data in _awaiting:
                awaiting_item = AwaitingToolCall.from_dict(awaiting_item_data)

                awaiting.append(awaiting_item)

        vault_ids = cast(list[str], d.pop("vault_ids", UNSET))

        _usage = d.pop("usage", UNSET)
        usage: SessionUsage | Unset
        if isinstance(_usage, Unset):
            usage = UNSET
        else:
            usage = SessionUsage.from_dict(_usage)

        _resources = d.pop("resources", UNSET)
        resources: (
            list[GithubRepositoryResourceEcho | MemoryStoreResourceEcho] | Unset
        ) = UNSET
        if _resources is not UNSET:
            resources = []
            for resources_item_data in _resources:

                def _parse_resources_item(
                    data: object,
                ) -> GithubRepositoryResourceEcho | MemoryStoreResourceEcho:
                    try:
                        if not isinstance(data, dict):
                            raise TypeError()
                        resources_item_type_0 = MemoryStoreResourceEcho.from_dict(data)

                        return resources_item_type_0
                    except (TypeError, ValueError, AttributeError, KeyError):
                        pass
                    if not isinstance(data, dict):
                        raise TypeError()
                    resources_item_type_1 = GithubRepositoryResourceEcho.from_dict(data)

                    return resources_item_type_1

                resources_item = _parse_resources_item(resources_item_data)

                resources.append(resources_item)

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

        def _parse_focal_channel(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        focal_channel = _parse_focal_channel(d.pop("focal_channel", UNSET))

        focal_locked = d.pop("focal_locked", UNSET)

        def _parse_last_event_at(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                last_event_at_type_0 = isoparse(data)

                return last_event_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        last_event_at = _parse_last_event_at(d.pop("last_event_at", UNSET))

        total_events = d.pop("total_events", UNSET)

        session = cls(
            id=id,
            agent_id=agent_id,
            environment_id=environment_id,
            agent_version=agent_version,
            title=title,
            metadata=metadata,
            status=status,
            stop_reason=stop_reason,
            last_event_seq=last_event_seq,
            created_at=created_at,
            updated_at=updated_at,
            awaiting=awaiting,
            vault_ids=vault_ids,
            usage=usage,
            resources=resources,
            archived_at=archived_at,
            focal_channel=focal_channel,
            focal_locked=focal_locked,
            last_event_at=last_event_at,
            total_events=total_events,
        )

        session.additional_properties = d
        return session

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
