from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.scheduled_task_echo_last_fire_status_type_0 import (
    ScheduledTaskEchoLastFireStatusType0,
)

if TYPE_CHECKING:
    from ..models.scheduled_task_echo_metadata import ScheduledTaskEchoMetadata


T = TypeVar("T", bound="ScheduledTaskEcho")


@_attrs_define
class ScheduledTaskEcho:
    """Read view of a scheduled task as echoed on ``Session.scheduled_tasks``.

    Runtime fields (``last_fire_at`` / ``last_fire_status`` /
    ``consecutive_failures``) reflect the most recent fire outcome.
    ``running_since`` is internal scheduler bookkeeping and is not
    exposed here.

        Attributes:
            id (str):
            name (str):
            schedule (str):
            command (str):
            enabled (bool):
            timeout_seconds (int):
            max_output_bytes (int):
            next_fire (datetime.datetime | None):
            last_fire_at (datetime.datetime | None):
            last_fire_status (None | ScheduledTaskEchoLastFireStatusType0):
            consecutive_failures (int):
            metadata (ScheduledTaskEchoMetadata):
            created_at (datetime.datetime):
            updated_at (datetime.datetime):
    """

    id: str
    name: str
    schedule: str
    command: str
    enabled: bool
    timeout_seconds: int
    max_output_bytes: int
    next_fire: datetime.datetime | None
    last_fire_at: datetime.datetime | None
    last_fire_status: None | ScheduledTaskEchoLastFireStatusType0
    consecutive_failures: int
    metadata: ScheduledTaskEchoMetadata
    created_at: datetime.datetime
    updated_at: datetime.datetime
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        name = self.name

        schedule = self.schedule

        command = self.command

        enabled = self.enabled

        timeout_seconds = self.timeout_seconds

        max_output_bytes = self.max_output_bytes

        next_fire: None | str
        if isinstance(self.next_fire, datetime.datetime):
            next_fire = self.next_fire.isoformat()
        else:
            next_fire = self.next_fire

        last_fire_at: None | str
        if isinstance(self.last_fire_at, datetime.datetime):
            last_fire_at = self.last_fire_at.isoformat()
        else:
            last_fire_at = self.last_fire_at

        last_fire_status: None | str
        if isinstance(self.last_fire_status, ScheduledTaskEchoLastFireStatusType0):
            last_fire_status = self.last_fire_status.value
        else:
            last_fire_status = self.last_fire_status

        consecutive_failures = self.consecutive_failures

        metadata = self.metadata.to_dict()

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "name": name,
                "schedule": schedule,
                "command": command,
                "enabled": enabled,
                "timeout_seconds": timeout_seconds,
                "max_output_bytes": max_output_bytes,
                "next_fire": next_fire,
                "last_fire_at": last_fire_at,
                "last_fire_status": last_fire_status,
                "consecutive_failures": consecutive_failures,
                "metadata": metadata,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.scheduled_task_echo_metadata import ScheduledTaskEchoMetadata

        d = dict(src_dict)
        id = d.pop("id")

        name = d.pop("name")

        schedule = d.pop("schedule")

        command = d.pop("command")

        enabled = d.pop("enabled")

        timeout_seconds = d.pop("timeout_seconds")

        max_output_bytes = d.pop("max_output_bytes")

        def _parse_next_fire(data: object) -> datetime.datetime | None:
            if data is None:
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                next_fire_type_0 = isoparse(data)

                return next_fire_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None, data)

        next_fire = _parse_next_fire(d.pop("next_fire"))

        def _parse_last_fire_at(data: object) -> datetime.datetime | None:
            if data is None:
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                last_fire_at_type_0 = isoparse(data)

                return last_fire_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None, data)

        last_fire_at = _parse_last_fire_at(d.pop("last_fire_at"))

        def _parse_last_fire_status(
            data: object,
        ) -> None | ScheduledTaskEchoLastFireStatusType0:
            if data is None:
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                last_fire_status_type_0 = ScheduledTaskEchoLastFireStatusType0(data)

                return last_fire_status_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | ScheduledTaskEchoLastFireStatusType0, data)

        last_fire_status = _parse_last_fire_status(d.pop("last_fire_status"))

        consecutive_failures = d.pop("consecutive_failures")

        metadata = ScheduledTaskEchoMetadata.from_dict(d.pop("metadata"))

        created_at = isoparse(d.pop("created_at"))

        updated_at = isoparse(d.pop("updated_at"))

        scheduled_task_echo = cls(
            id=id,
            name=name,
            schedule=schedule,
            command=command,
            enabled=enabled,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
            next_fire=next_fire,
            last_fire_at=last_fire_at,
            last_fire_status=last_fire_status,
            consecutive_failures=consecutive_failures,
            metadata=metadata,
            created_at=created_at,
            updated_at=updated_at,
        )

        scheduled_task_echo.additional_properties = d
        return scheduled_task_echo

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
