from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.cron_source import CronSource
    from ..models.one_shot_source import OneShotSource
    from ..models.sandbox_command_action_replace import SandboxCommandActionReplace
    from ..models.trigger_update_metadata_type_0 import TriggerUpdateMetadataType0
    from ..models.wake_owner_action import WakeOwnerAction


T = TypeVar("T", bound="TriggerUpdate")


@_attrs_define
class TriggerUpdate:
    """Update body. ``source`` / ``action`` are replaced WHOLESALE when
    provided (a cron↔one-shot or sandbox↔wake conversion is just a
    different object). ``None`` = leave alone; there is no clear-to-null
    (both columns are NOT NULL). The next_fire / cap / past-fire_at
    business rules are enforced in the service layer (§2.4).

        Attributes:
            source (CronSource | None | OneShotSource | Unset):
            action (None | SandboxCommandActionReplace | Unset | WakeOwnerAction):
            enabled (bool | None | Unset):
            metadata (None | TriggerUpdateMetadataType0 | Unset):
    """

    source: CronSource | None | OneShotSource | Unset = UNSET
    action: None | SandboxCommandActionReplace | Unset | WakeOwnerAction = UNSET
    enabled: bool | None | Unset = UNSET
    metadata: None | TriggerUpdateMetadataType0 | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.cron_source import CronSource
        from ..models.one_shot_source import OneShotSource
        from ..models.sandbox_command_action_replace import SandboxCommandActionReplace
        from ..models.trigger_update_metadata_type_0 import TriggerUpdateMetadataType0
        from ..models.wake_owner_action import WakeOwnerAction

        source: dict[str, Any] | None | Unset
        if isinstance(self.source, Unset):
            source = UNSET
        elif isinstance(self.source, CronSource):
            source = self.source.to_dict()
        elif isinstance(self.source, OneShotSource):
            source = self.source.to_dict()
        else:
            source = self.source

        action: dict[str, Any] | None | Unset
        if isinstance(self.action, Unset):
            action = UNSET
        elif isinstance(self.action, SandboxCommandActionReplace):
            action = self.action.to_dict()
        elif isinstance(self.action, WakeOwnerAction):
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
        from ..models.one_shot_source import OneShotSource
        from ..models.sandbox_command_action_replace import SandboxCommandActionReplace
        from ..models.trigger_update_metadata_type_0 import TriggerUpdateMetadataType0
        from ..models.wake_owner_action import WakeOwnerAction

        d = dict(src_dict)

        def _parse_source(data: object) -> CronSource | None | OneShotSource | Unset:
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
            return cast(CronSource | None | OneShotSource | Unset, data)

        source = _parse_source(d.pop("source", UNSET))

        def _parse_action(
            data: object,
        ) -> None | SandboxCommandActionReplace | Unset | WakeOwnerAction:
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
            return cast(
                None | SandboxCommandActionReplace | Unset | WakeOwnerAction, data
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
