from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.scheduled_task_update_metadata_type_0 import (
        ScheduledTaskUpdateMetadataType0,
    )


T = TypeVar("T", bound="ScheduledTaskUpdate")


@_attrs_define
class ScheduledTaskUpdate:
    """PATCH body for updating a scheduled task.

    ``name`` cannot be changed — it is the addressable identifier within
    a session. All other fields are optional; omitted fields are left
    unchanged. Toggling ``enabled`` true→false clears ``next_fire``;
    false→true recomputes it from now.

    Updates can adjust the trigger by setting ``schedule`` (cron) or
    ``fire_at`` (one-shot) — but not both in the same PATCH; the DB
    CHECK constraint enforces the XOR invariant after the merged write.

        Attributes:
            schedule (None | str | Unset):
            fire_at (datetime.datetime | None | Unset): Update the one-shot fire time. Mutually exclusive with `schedule`.
            command (None | str | Unset):
            enabled (bool | None | Unset):
            timeout_seconds (int | None | Unset):
            max_output_bytes (int | None | Unset):
            metadata (None | ScheduledTaskUpdateMetadataType0 | Unset):
    """

    schedule: None | str | Unset = UNSET
    fire_at: datetime.datetime | None | Unset = UNSET
    command: None | str | Unset = UNSET
    enabled: bool | None | Unset = UNSET
    timeout_seconds: int | None | Unset = UNSET
    max_output_bytes: int | None | Unset = UNSET
    metadata: None | ScheduledTaskUpdateMetadataType0 | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.scheduled_task_update_metadata_type_0 import (
            ScheduledTaskUpdateMetadataType0,
        )

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

        command: None | str | Unset
        if isinstance(self.command, Unset):
            command = UNSET
        else:
            command = self.command

        enabled: bool | None | Unset
        if isinstance(self.enabled, Unset):
            enabled = UNSET
        else:
            enabled = self.enabled

        timeout_seconds: int | None | Unset
        if isinstance(self.timeout_seconds, Unset):
            timeout_seconds = UNSET
        else:
            timeout_seconds = self.timeout_seconds

        max_output_bytes: int | None | Unset
        if isinstance(self.max_output_bytes, Unset):
            max_output_bytes = UNSET
        else:
            max_output_bytes = self.max_output_bytes

        metadata: dict[str, Any] | None | Unset
        if isinstance(self.metadata, Unset):
            metadata = UNSET
        elif isinstance(self.metadata, ScheduledTaskUpdateMetadataType0):
            metadata = self.metadata.to_dict()
        else:
            metadata = self.metadata

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if schedule is not UNSET:
            field_dict["schedule"] = schedule
        if fire_at is not UNSET:
            field_dict["fire_at"] = fire_at
        if command is not UNSET:
            field_dict["command"] = command
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
        from ..models.scheduled_task_update_metadata_type_0 import (
            ScheduledTaskUpdateMetadataType0,
        )

        d = dict(src_dict)

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

        def _parse_command(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        command = _parse_command(d.pop("command", UNSET))

        def _parse_enabled(data: object) -> bool | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(bool | None | Unset, data)

        enabled = _parse_enabled(d.pop("enabled", UNSET))

        def _parse_timeout_seconds(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        timeout_seconds = _parse_timeout_seconds(d.pop("timeout_seconds", UNSET))

        def _parse_max_output_bytes(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        max_output_bytes = _parse_max_output_bytes(d.pop("max_output_bytes", UNSET))

        def _parse_metadata(
            data: object,
        ) -> None | ScheduledTaskUpdateMetadataType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                metadata_type_0 = ScheduledTaskUpdateMetadataType0.from_dict(data)

                return metadata_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | ScheduledTaskUpdateMetadataType0 | Unset, data)

        metadata = _parse_metadata(d.pop("metadata", UNSET))

        scheduled_task_update = cls(
            schedule=schedule,
            fire_at=fire_at,
            command=command,
            enabled=enabled,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
            metadata=metadata,
        )

        return scheduled_task_update
