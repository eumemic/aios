from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.trigger_echo_last_fire_status_type_0 import TriggerEchoLastFireStatusType0

if TYPE_CHECKING:
    from ..models.cron_source import CronSource
    from ..models.external_event_source import ExternalEventSource
    from ..models.one_shot_source import OneShotSource
    from ..models.run_completion_source import RunCompletionSource
    from ..models.sandbox_command_action import SandboxCommandAction
    from ..models.trigger_echo_metadata import TriggerEchoMetadata
    from ..models.wake_owner_action import WakeOwnerAction
    from ..models.wake_session_action import WakeSessionAction
    from ..models.workflow_action import WorkflowAction


T = TypeVar("T", bound="TriggerEcho")


@_attrs_define
class TriggerEcho:
    """Read view of a trigger as echoed on ``Session.triggers``.

    Runtime fields (``last_fire_at`` / ``last_fire_status`` /
    ``consecutive_failures``) reflect the most recent fire outcome.
    ``running_since`` is internal scheduler bookkeeping and is not exposed
    here.

        Attributes:
            id (str):
            name (str):
            source (CronSource | ExternalEventSource | OneShotSource | RunCompletionSource):
            action (SandboxCommandAction | WakeOwnerAction | WakeSessionAction | WorkflowAction):
            enabled (bool):
            next_fire (datetime.datetime | None):
            last_fire_at (datetime.datetime | None):
            last_fire_status (None | TriggerEchoLastFireStatusType0):
            consecutive_failures (int):
            metadata (TriggerEchoMetadata):
            created_at (datetime.datetime):
            updated_at (datetime.datetime):
    """

    id: str
    name: str
    source: CronSource | ExternalEventSource | OneShotSource | RunCompletionSource
    action: SandboxCommandAction | WakeOwnerAction | WakeSessionAction | WorkflowAction
    enabled: bool
    next_fire: datetime.datetime | None
    last_fire_at: datetime.datetime | None
    last_fire_status: None | TriggerEchoLastFireStatusType0
    consecutive_failures: int
    metadata: TriggerEchoMetadata
    created_at: datetime.datetime
    updated_at: datetime.datetime
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.cron_source import CronSource
        from ..models.one_shot_source import OneShotSource
        from ..models.run_completion_source import RunCompletionSource
        from ..models.sandbox_command_action import SandboxCommandAction
        from ..models.wake_owner_action import WakeOwnerAction
        from ..models.wake_session_action import WakeSessionAction

        id = self.id

        name = self.name

        source: dict[str, Any]
        if isinstance(self.source, CronSource):
            source = self.source.to_dict()
        elif isinstance(self.source, OneShotSource):
            source = self.source.to_dict()
        elif isinstance(self.source, RunCompletionSource):
            source = self.source.to_dict()
        else:
            source = self.source.to_dict()

        action: dict[str, Any]
        if isinstance(self.action, SandboxCommandAction):
            action = self.action.to_dict()
        elif isinstance(self.action, WakeOwnerAction):
            action = self.action.to_dict()
        elif isinstance(self.action, WakeSessionAction):
            action = self.action.to_dict()
        else:
            action = self.action.to_dict()

        enabled = self.enabled

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
        if isinstance(self.last_fire_status, TriggerEchoLastFireStatusType0):
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
                "source": source,
                "action": action,
                "enabled": enabled,
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
        from ..models.cron_source import CronSource
        from ..models.external_event_source import ExternalEventSource
        from ..models.one_shot_source import OneShotSource
        from ..models.run_completion_source import RunCompletionSource
        from ..models.sandbox_command_action import SandboxCommandAction
        from ..models.trigger_echo_metadata import TriggerEchoMetadata
        from ..models.wake_owner_action import WakeOwnerAction
        from ..models.wake_session_action import WakeSessionAction
        from ..models.workflow_action import WorkflowAction

        d = dict(src_dict)
        id = d.pop("id")

        name = d.pop("name")

        def _parse_source(
            data: object,
        ) -> CronSource | ExternalEventSource | OneShotSource | RunCompletionSource:
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                source_type_0 = CronSource.from_dict(data)

                return source_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                source_type_1 = OneShotSource.from_dict(data)

                return source_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                source_type_2 = RunCompletionSource.from_dict(data)

                return source_type_2
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            if not isinstance(data, dict):
                raise TypeError()
            source_type_3 = ExternalEventSource.from_dict(data)

            return source_type_3

        source = _parse_source(d.pop("source"))

        def _parse_action(
            data: object,
        ) -> (
            SandboxCommandAction | WakeOwnerAction | WakeSessionAction | WorkflowAction
        ):
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                action_type_0 = SandboxCommandAction.from_dict(data)

                return action_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                action_type_1 = WakeOwnerAction.from_dict(data)

                return action_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                action_type_2 = WakeSessionAction.from_dict(data)

                return action_type_2
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            if not isinstance(data, dict):
                raise TypeError()
            action_type_3 = WorkflowAction.from_dict(data)

            return action_type_3

        action = _parse_action(d.pop("action"))

        enabled = d.pop("enabled")

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
        ) -> None | TriggerEchoLastFireStatusType0:
            if data is None:
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                last_fire_status_type_0 = TriggerEchoLastFireStatusType0(data)

                return last_fire_status_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | TriggerEchoLastFireStatusType0, data)

        last_fire_status = _parse_last_fire_status(d.pop("last_fire_status"))

        consecutive_failures = d.pop("consecutive_failures")

        metadata = TriggerEchoMetadata.from_dict(d.pop("metadata"))

        created_at = isoparse(d.pop("created_at"))

        updated_at = isoparse(d.pop("updated_at"))

        trigger_echo = cls(
            id=id,
            name=name,
            source=source,
            action=action,
            enabled=enabled,
            next_fire=next_fire,
            last_fire_at=last_fire_at,
            last_fire_status=last_fire_status,
            consecutive_failures=consecutive_failures,
            metadata=metadata,
            created_at=created_at,
            updated_at=updated_at,
        )

        trigger_echo.additional_properties = d
        return trigger_echo

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
