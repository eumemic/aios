from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.scheduled_task_create_metadata import ScheduledTaskCreateMetadata


T = TypeVar("T", bound="ScheduledTaskCreate")


@_attrs_define
class ScheduledTaskCreate:
    """Request body for adding a scheduled task to a session.

    Also accepted in :class:`SessionCreate.scheduled_tasks` for initial
    attachment at session creation.

        Attributes:
            name (str): Stable user-chosen identifier; unique per session.
            schedule (str): Standard 5-field cron expression in UTC.
            command (str): Bash command run in the session's sandbox at each fire.
            enabled (bool | Unset):  Default: True.
            timeout_seconds (int | Unset):  Default: 300.
            max_output_bytes (int | Unset):  Default: 65536.
            metadata (ScheduledTaskCreateMetadata | Unset):
    """

    name: str
    schedule: str
    command: str
    enabled: bool | Unset = True
    timeout_seconds: int | Unset = 300
    max_output_bytes: int | Unset = 65536
    metadata: ScheduledTaskCreateMetadata | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        schedule = self.schedule

        command = self.command

        enabled = self.enabled

        timeout_seconds = self.timeout_seconds

        max_output_bytes = self.max_output_bytes

        metadata: dict[str, Any] | Unset = UNSET
        if not isinstance(self.metadata, Unset):
            metadata = self.metadata.to_dict()

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "name": name,
                "schedule": schedule,
                "command": command,
            }
        )
        if enabled is not UNSET:
            field_dict["enabled"] = enabled
        if timeout_seconds is not UNSET:
            field_dict["timeout_seconds"] = timeout_seconds
        if max_output_bytes is not UNSET:
            field_dict["max_output_bytes"] = max_output_bytes
        if metadata is not UNSET:
            field_dict["metadata"] = metadata

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.scheduled_task_create_metadata import ScheduledTaskCreateMetadata

        d = dict(src_dict)
        name = d.pop("name")

        schedule = d.pop("schedule")

        command = d.pop("command")

        enabled = d.pop("enabled", UNSET)

        timeout_seconds = d.pop("timeout_seconds", UNSET)

        max_output_bytes = d.pop("max_output_bytes", UNSET)

        _metadata = d.pop("metadata", UNSET)
        metadata: ScheduledTaskCreateMetadata | Unset
        if isinstance(_metadata, Unset):
            metadata = UNSET
        else:
            metadata = ScheduledTaskCreateMetadata.from_dict(_metadata)

        scheduled_task_create = cls(
            name=name,
            schedule=schedule,
            command=command,
            enabled=enabled,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
            metadata=metadata,
        )

        return scheduled_task_create
