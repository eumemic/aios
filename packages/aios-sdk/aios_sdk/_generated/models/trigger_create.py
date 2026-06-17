from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.cron_source import CronSource
    from ..models.one_shot_source import OneShotSource
    from ..models.run_completion_source import RunCompletionSource
    from ..models.sandbox_command_action import SandboxCommandAction
    from ..models.trigger_create_metadata import TriggerCreateMetadata
    from ..models.wake_owner_action import WakeOwnerAction
    from ..models.wake_session_action import WakeSessionAction
    from ..models.workflow_action import WorkflowAction


T = TypeVar("T", bound="TriggerCreate")


@_attrs_define
class TriggerCreate:
    """Request body for adding a trigger to a session.

    Carries a ``source`` (cron / one_shot) and an ``action``
    (sandbox_command / wake_owner). Also accepted in
    :class:`SessionCreate.triggers` for initial attachment at session
    creation.

        Attributes:
            name (str): Stable user-chosen identifier; unique per session.
            source (CronSource | OneShotSource | RunCompletionSource):
            action (SandboxCommandAction | WakeOwnerAction | WakeSessionAction | WorkflowAction):
            enabled (bool | Unset):  Default: True.
            metadata (TriggerCreateMetadata | Unset):
    """

    name: str
    source: CronSource | OneShotSource | RunCompletionSource
    action: SandboxCommandAction | WakeOwnerAction | WakeSessionAction | WorkflowAction
    enabled: bool | Unset = True
    metadata: TriggerCreateMetadata | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.cron_source import CronSource
        from ..models.one_shot_source import OneShotSource
        from ..models.sandbox_command_action import SandboxCommandAction
        from ..models.wake_owner_action import WakeOwnerAction
        from ..models.wake_session_action import WakeSessionAction

        name = self.name

        source: dict[str, Any]
        if isinstance(self.source, CronSource):
            source = self.source.to_dict()
        elif isinstance(self.source, OneShotSource):
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

        metadata: dict[str, Any] | Unset = UNSET
        if not isinstance(self.metadata, Unset):
            metadata = self.metadata.to_dict()

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "name": name,
                "source": source,
                "action": action,
            }
        )
        if enabled is not UNSET:
            field_dict["enabled"] = enabled
        if metadata is not UNSET:
            field_dict["metadata"] = metadata

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.cron_source import CronSource
        from ..models.one_shot_source import OneShotSource
        from ..models.run_completion_source import RunCompletionSource
        from ..models.sandbox_command_action import SandboxCommandAction
        from ..models.trigger_create_metadata import TriggerCreateMetadata
        from ..models.wake_owner_action import WakeOwnerAction
        from ..models.wake_session_action import WakeSessionAction
        from ..models.workflow_action import WorkflowAction

        d = dict(src_dict)
        name = d.pop("name")

        def _parse_source(
            data: object,
        ) -> CronSource | OneShotSource | RunCompletionSource:
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
            if not isinstance(data, dict):
                raise TypeError()
            source_type_2 = RunCompletionSource.from_dict(data)

            return source_type_2

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

        enabled = d.pop("enabled", UNSET)

        _metadata = d.pop("metadata", UNSET)
        metadata: TriggerCreateMetadata | Unset
        if isinstance(_metadata, Unset):
            metadata = UNSET
        else:
            metadata = TriggerCreateMetadata.from_dict(_metadata)

        trigger_create = cls(
            name=name,
            source=source,
            action=action,
            enabled=enabled,
            metadata=metadata,
        )

        return trigger_create
