from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

if TYPE_CHECKING:
    from ..models.trigger_run_echo_event_type_0 import TriggerRunEchoEventType0


T = TypeVar("T", bound="TriggerRunEcho")


@_attrs_define
class TriggerRunEcho:
    """Read view of one ``trigger_runs`` row — a single fire of a trigger.

    ``trigger_context`` echoes the firing source (``cron`` / ``one_shot`` /
    ``run_completion``); ``event`` carries the per-event context for
    ``run_completion`` fires (``{run_id, workflow_id, status}``) and is
    ``None`` for timer fires. ``status`` is an open string on read (rows
    written by future writers must always read back); the current writer
    vocabulary is ``pending``/``running``/``ok``/``error``/``timeout``/
    ``skipped``. ``result_id`` is the prefixed id of the resource the fire
    created (a ``wfr_…`` run today), or ``None`` when the fire produced no
    resource or failed. ``created_at`` is when the fire INTENT was created
    (the match transaction for event fires; insert time for timer rows);
    ``started_at`` is when execution claimed it.

        Attributes:
            id (str):
            trigger_id (str):
            trigger_context (str):
            event (None | TriggerRunEchoEventType0):
            status (str):
            result_id (None | str):
            error_summary (None | str):
            created_at (datetime.datetime):
            started_at (datetime.datetime | None):
            finished_at (datetime.datetime | None):
    """

    id: str
    trigger_id: str
    trigger_context: str
    event: None | TriggerRunEchoEventType0
    status: str
    result_id: None | str
    error_summary: None | str
    created_at: datetime.datetime
    started_at: datetime.datetime | None
    finished_at: datetime.datetime | None
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.trigger_run_echo_event_type_0 import TriggerRunEchoEventType0

        id = self.id

        trigger_id = self.trigger_id

        trigger_context = self.trigger_context

        event: dict[str, Any] | None
        if isinstance(self.event, TriggerRunEchoEventType0):
            event = self.event.to_dict()
        else:
            event = self.event

        status = self.status

        result_id: None | str
        result_id = self.result_id

        error_summary: None | str
        error_summary = self.error_summary

        created_at = self.created_at.isoformat()

        started_at: None | str
        if isinstance(self.started_at, datetime.datetime):
            started_at = self.started_at.isoformat()
        else:
            started_at = self.started_at

        finished_at: None | str
        if isinstance(self.finished_at, datetime.datetime):
            finished_at = self.finished_at.isoformat()
        else:
            finished_at = self.finished_at

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "trigger_id": trigger_id,
                "trigger_context": trigger_context,
                "event": event,
                "status": status,
                "result_id": result_id,
                "error_summary": error_summary,
                "created_at": created_at,
                "started_at": started_at,
                "finished_at": finished_at,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.trigger_run_echo_event_type_0 import TriggerRunEchoEventType0

        d = dict(src_dict)
        id = d.pop("id")

        trigger_id = d.pop("trigger_id")

        trigger_context = d.pop("trigger_context")

        def _parse_event(data: object) -> None | TriggerRunEchoEventType0:
            if data is None:
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                event_type_0 = TriggerRunEchoEventType0.from_dict(data)

                return event_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | TriggerRunEchoEventType0, data)

        event = _parse_event(d.pop("event"))

        status = d.pop("status")

        def _parse_result_id(data: object) -> None | str:
            if data is None:
                return data
            return cast(None | str, data)

        result_id = _parse_result_id(d.pop("result_id"))

        def _parse_error_summary(data: object) -> None | str:
            if data is None:
                return data
            return cast(None | str, data)

        error_summary = _parse_error_summary(d.pop("error_summary"))

        created_at = isoparse(d.pop("created_at"))

        def _parse_started_at(data: object) -> datetime.datetime | None:
            if data is None:
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                started_at_type_0 = isoparse(data)

                return started_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None, data)

        started_at = _parse_started_at(d.pop("started_at"))

        def _parse_finished_at(data: object) -> datetime.datetime | None:
            if data is None:
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                finished_at_type_0 = isoparse(data)

                return finished_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None, data)

        finished_at = _parse_finished_at(d.pop("finished_at"))

        trigger_run_echo = cls(
            id=id,
            trigger_id=trigger_id,
            trigger_context=trigger_context,
            event=event,
            status=status,
            result_id=result_id,
            error_summary=error_summary,
            created_at=created_at,
            started_at=started_at,
            finished_at=finished_at,
        )

        trigger_run_echo.additional_properties = d
        return trigger_run_echo

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
