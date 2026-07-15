from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.cron_source import CronSource
    from ..models.external_event_source import ExternalEventSource
    from ..models.one_shot_source import OneShotSource
    from ..models.run_completion_source_replace import RunCompletionSourceReplace
    from ..models.sandbox_command_action_replace import SandboxCommandActionReplace
    from ..models.trigger_update_metadata_type_0 import TriggerUpdateMetadataType0
    from ..models.wake_owner_action import WakeOwnerAction
    from ..models.wake_session_action import WakeSessionAction
    from ..models.workflow_action_replace import WorkflowActionReplace


T = TypeVar("T", bound="TriggerUpdate")


@_attrs_define
class TriggerUpdate:
    """Update body. ``source`` / ``action`` are replaced WHOLESALE when
    provided (a cron↔one-shot or sandbox↔wake conversion is just a
    different object) — via the Replace union variants, whose
    optional-at-create fields are required so a partial object 422s instead
    of silently re-defaulting. ``None`` = leave alone; there is no
    clear-to-null (both columns are NOT NULL). The next_fire / cap /
    past-fire_at business rules are enforced in the service layer (§2.4).

        Attributes:
            source (CronSource | ExternalEventSource | None | OneShotSource | RunCompletionSourceReplace | Unset):
            action (None | SandboxCommandActionReplace | Unset | WakeOwnerAction | WakeSessionAction |
                WorkflowActionReplace):
            enabled (bool | None | Unset):
            metadata (None | TriggerUpdateMetadataType0 | Unset):
    """

    source: (
        CronSource
        | ExternalEventSource
        | None
        | OneShotSource
        | RunCompletionSourceReplace
        | Unset
    ) = UNSET
    action: (
        None
        | SandboxCommandActionReplace
        | Unset
        | WakeOwnerAction
        | WakeSessionAction
        | WorkflowActionReplace
    ) = UNSET
    enabled: bool | None | Unset = UNSET
    metadata: None | TriggerUpdateMetadataType0 | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.cron_source import CronSource
        from ..models.external_event_source import ExternalEventSource
        from ..models.one_shot_source import OneShotSource
        from ..models.run_completion_source_replace import RunCompletionSourceReplace
        from ..models.sandbox_command_action_replace import SandboxCommandActionReplace
        from ..models.trigger_update_metadata_type_0 import TriggerUpdateMetadataType0
        from ..models.wake_owner_action import WakeOwnerAction
        from ..models.wake_session_action import WakeSessionAction
        from ..models.workflow_action_replace import WorkflowActionReplace

        source: dict[str, Any] | None | Unset
        if isinstance(self.source, Unset):
            source = UNSET
        elif isinstance(self.source, CronSource) or isinstance(self.source, OneShotSource) or isinstance(self.source, RunCompletionSourceReplace) or isinstance(self.source, ExternalEventSource):
            source = self.source.to_dict()
        else:
            source = self.source

        action: dict[str, Any] | None | Unset
        if isinstance(self.action, Unset):
            action = UNSET
        elif isinstance(self.action, SandboxCommandActionReplace) or isinstance(self.action, WakeOwnerAction) or isinstance(self.action, WakeSessionAction) or isinstance(self.action, WorkflowActionReplace):
            action = self.action.to_dict()
        else:
            action = self.action

        enabled: bool | None | Unset
        if isinstance(self.enabled, Unset):
            enabled = UNSET
        else:
            enabled = self.enabled

        metadata: dict[str, Any] | None | Unset
        if isinstance(self.metadata, Unset):
            metadata = UNSET
        elif isinstance(self.metadata, TriggerUpdateMetadataType0):
            metadata = self.metadata.to_dict()
        else:
            metadata = self.metadata

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if source is not UNSET:
            field_dict["source"] = source
        if action is not UNSET:
            field_dict["action"] = action
        if enabled is not UNSET:
            field_dict["enabled"] = enabled
        if metadata is not UNSET:
            field_dict["metadata"] = metadata

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.cron_source import CronSource
        from ..models.external_event_source import ExternalEventSource
        from ..models.one_shot_source import OneShotSource
        from ..models.run_completion_source_replace import RunCompletionSourceReplace
        from ..models.sandbox_command_action_replace import SandboxCommandActionReplace
        from ..models.trigger_update_metadata_type_0 import TriggerUpdateMetadataType0
        from ..models.wake_owner_action import WakeOwnerAction
        from ..models.wake_session_action import WakeSessionAction
        from ..models.workflow_action_replace import WorkflowActionReplace

        d = dict(src_dict)

        def _parse_source(
            data: object,
        ) -> (
            CronSource
            | ExternalEventSource
            | None
            | OneShotSource
            | RunCompletionSourceReplace
            | Unset
        ):
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                source_type_0_type_0 = CronSource.from_dict(data)

                return source_type_0_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                source_type_0_type_1 = OneShotSource.from_dict(data)

                return source_type_0_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                source_type_0_type_2 = RunCompletionSourceReplace.from_dict(data)

                return source_type_0_type_2
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                source_type_0_type_3 = ExternalEventSource.from_dict(data)

                return source_type_0_type_3
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(
                CronSource
                | ExternalEventSource
                | None
                | OneShotSource
                | RunCompletionSourceReplace
                | Unset,
                data,
            )

        source = _parse_source(d.pop("source", UNSET))

        def _parse_action(
            data: object,
        ) -> (
            None
            | SandboxCommandActionReplace
            | Unset
            | WakeOwnerAction
            | WakeSessionAction
            | WorkflowActionReplace
        ):
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                action_type_0_type_0 = SandboxCommandActionReplace.from_dict(data)

                return action_type_0_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                action_type_0_type_1 = WakeOwnerAction.from_dict(data)

                return action_type_0_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                action_type_0_type_2 = WakeSessionAction.from_dict(data)

                return action_type_0_type_2
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                action_type_0_type_3 = WorkflowActionReplace.from_dict(data)

                return action_type_0_type_3
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(
                None
                | SandboxCommandActionReplace
                | Unset
                | WakeOwnerAction
                | WakeSessionAction
                | WorkflowActionReplace,
                data,
            )

        action = _parse_action(d.pop("action", UNSET))

        def _parse_enabled(data: object) -> bool | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(bool | None | Unset, data)

        enabled = _parse_enabled(d.pop("enabled", UNSET))

        def _parse_metadata(data: object) -> None | TriggerUpdateMetadataType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                metadata_type_0 = TriggerUpdateMetadataType0.from_dict(data)

                return metadata_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | TriggerUpdateMetadataType0 | Unset, data)

        metadata = _parse_metadata(d.pop("metadata", UNSET))

        trigger_update = cls(
            source=source,
            action=action,
            enabled=enabled,
            metadata=metadata,
        )

        return trigger_update
