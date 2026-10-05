from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.cron_source_replace import CronSourceReplace
    from ..models.one_shot_source import OneShotSource
    from ..models.operator_trigger_update_metadata_type_0 import (
        OperatorTriggerUpdateMetadataType0,
    )
    from ..models.operator_workflow_action_replace import OperatorWorkflowActionReplace


T = TypeVar("T", bound="OperatorTriggerUpdate")


@_attrs_define
class OperatorTriggerUpdate:
    """Update body for ``PUT /v1/triggers/{name}``: the same Replace semantics as
    :class:`TriggerUpdate`, limited to the operator shapes.

        Attributes:
            source (CronSourceReplace | None | OneShotSource | Unset):
            action (None | OperatorWorkflowActionReplace | Unset):
            enabled (bool | None | Unset):
            metadata (None | OperatorTriggerUpdateMetadataType0 | Unset):
    """

    source: CronSourceReplace | None | OneShotSource | Unset = UNSET
    action: None | OperatorWorkflowActionReplace | Unset = UNSET
    enabled: bool | None | Unset = UNSET
    metadata: None | OperatorTriggerUpdateMetadataType0 | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.cron_source_replace import CronSourceReplace
        from ..models.one_shot_source import OneShotSource
        from ..models.operator_trigger_update_metadata_type_0 import (
            OperatorTriggerUpdateMetadataType0,
        )
        from ..models.operator_workflow_action_replace import (
            OperatorWorkflowActionReplace,
        )

        source: dict[str, Any] | None | Unset
        if isinstance(self.source, Unset):
            source = UNSET
        elif isinstance(self.source, CronSourceReplace):
            source = self.source.to_dict()
        elif isinstance(self.source, OneShotSource):
            source = self.source.to_dict()
        else:
            source = self.source

        action: dict[str, Any] | None | Unset
        if isinstance(self.action, Unset):
            action = UNSET
        elif isinstance(self.action, OperatorWorkflowActionReplace):
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
        elif isinstance(self.metadata, OperatorTriggerUpdateMetadataType0):
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
        from ..models.cron_source_replace import CronSourceReplace
        from ..models.one_shot_source import OneShotSource
        from ..models.operator_trigger_update_metadata_type_0 import (
            OperatorTriggerUpdateMetadataType0,
        )
        from ..models.operator_workflow_action_replace import (
            OperatorWorkflowActionReplace,
        )

        d = dict(src_dict)

        def _parse_source(
            data: object,
        ) -> CronSourceReplace | None | OneShotSource | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                source_type_0_type_0 = CronSourceReplace.from_dict(data)

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
            return cast(CronSourceReplace | None | OneShotSource | Unset, data)

        source = _parse_source(d.pop("source", UNSET))

        def _parse_action(data: object) -> None | OperatorWorkflowActionReplace | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                action_type_0 = OperatorWorkflowActionReplace.from_dict(data)

                return action_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | OperatorWorkflowActionReplace | Unset, data)

        action = _parse_action(d.pop("action", UNSET))

        def _parse_enabled(data: object) -> bool | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(bool | None | Unset, data)

        enabled = _parse_enabled(d.pop("enabled", UNSET))

        def _parse_metadata(
            data: object,
        ) -> None | OperatorTriggerUpdateMetadataType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                metadata_type_0 = OperatorTriggerUpdateMetadataType0.from_dict(data)

                return metadata_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | OperatorTriggerUpdateMetadataType0 | Unset, data)

        metadata = _parse_metadata(d.pop("metadata", UNSET))

        operator_trigger_update = cls(
            source=source,
            action=action,
            enabled=enabled,
            metadata=metadata,
        )

        return operator_trigger_update
