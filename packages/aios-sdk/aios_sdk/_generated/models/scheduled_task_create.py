from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.scheduled_task_create_metadata import ScheduledTaskCreateMetadata


T = TypeVar("T", bound="ScheduledTaskCreate")


@_attrs_define
class ScheduledTaskCreate:
    """Request body for adding a scheduled task to a session.

    Each row carries either a cron ``schedule`` (recurring) or a
    ``fire_at`` absolute time (one-shot — self-deletes after firing).
    Exactly one must be set; enforced by both a Pydantic ``model_validator``
    here and a DB CHECK constraint.

    Also accepted in :class:`SessionCreate.scheduled_tasks` for initial
    attachment at session creation.

        Attributes:
            name (str): Stable user-chosen identifier; unique per session.
            command (str): Bash command run in the session's sandbox at each fire.
            schedule (None | str | Unset): Standard 5-field cron expression in UTC for recurring rows. Mutually exclusive
                with ``fire_at``.
            fire_at (datetime.datetime | None | Unset): Absolute UTC time for one-shot rows; the row self-deletes after
                firing. Mutually exclusive with ``schedule``.
            enabled (bool | Unset):  Default: True.
            timeout_seconds (int | Unset):  Default: 300.
            max_output_bytes (int | Unset):  Default: 65536.
            metadata (ScheduledTaskCreateMetadata | Unset):
    """

    name: str
    command: str
    schedule: None | str | Unset = UNSET
    fire_at: datetime.datetime | None | Unset = UNSET
    enabled: bool | Unset = True
    timeout_seconds: int | Unset = 300
    max_output_bytes: int | Unset = 65536
    metadata: ScheduledTaskCreateMetadata | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        command = self.command

        schedule: None | str | Unset
        if isinstance(self.schedule, Unset):
            schedule = UNSET
        else:
            schedule = self.schedule

        fire_at: None | str | Unset
        if isinstance(self.fire_at, Unset):
            fire_at = UNSET
        elif isinstance(self.fire_at, datetime.datetime):
            fire_at = self.fire_at.isoformat()
        else:
            fire_at = self.fire_at

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
                "command": command,
            }
        )
        if schedule is not UNSET:
            field_dict["schedule"] = schedule
        if fire_at is not UNSET:
            field_dict["fire_at"] = fire_at
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

        command = d.pop("command")

        def _parse_schedule(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        schedule = _parse_schedule(d.pop("schedule", UNSET))

        def _parse_fire_at(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                fire_at_type_0 = isoparse(data)

                return fire_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        fire_at = _parse_fire_at(d.pop("fire_at", UNSET))

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
            command=command,
            schedule=schedule,
            fire_at=fire_at,
            enabled=enabled,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
            metadata=metadata,
        )

        return scheduled_task_create
