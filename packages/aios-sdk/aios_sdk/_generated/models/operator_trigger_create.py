from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.cron_source import CronSource
    from ..models.one_shot_source import OneShotSource
    from ..models.operator_trigger_create_metadata import OperatorTriggerCreateMetadata
    from ..models.operator_workflow_action import OperatorWorkflowAction


T = TypeVar("T", bound="OperatorTriggerCreate")


@_attrs_define
class OperatorTriggerCreate:
    """Request body for ``POST /v1/triggers``. ``environment_id`` is the
    environment the trigger's runs bind to, named by the operator as on
    ``POST /v1/runs``; it can't be changed later.

        Attributes:
            name (str): Stable identifier; unique among the account's operator triggers.
            source (CronSource | OneShotSource):
            action (OperatorWorkflowAction): The workflow action of an operator trigger: ``budget_usd`` is required,
                so every operator run it launches is bounded.
            environment_id (str):
            enabled (bool | Unset):  Default: True.
            metadata (OperatorTriggerCreateMetadata | Unset):
    """

    name: str
    source: CronSource | OneShotSource
    action: OperatorWorkflowAction
    environment_id: str
    enabled: bool | Unset = True
    metadata: OperatorTriggerCreateMetadata | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.cron_source import CronSource

        name = self.name

        source: dict[str, Any]
        if isinstance(self.source, CronSource):
            source = self.source.to_dict()
        else:
            source = self.source.to_dict()

        action = self.action.to_dict()

        environment_id = self.environment_id

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
                "environment_id": environment_id,
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
        from ..models.operator_trigger_create_metadata import (
            OperatorTriggerCreateMetadata,
        )
        from ..models.operator_workflow_action import OperatorWorkflowAction

        d = dict(src_dict)
        name = d.pop("name")

        def _parse_source(data: object) -> CronSource | OneShotSource:
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                source_type_0 = CronSource.from_dict(data)

                return source_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            if not isinstance(data, dict):
                raise TypeError()
            source_type_1 = OneShotSource.from_dict(data)

            return source_type_1

        source = _parse_source(d.pop("source"))

        action = OperatorWorkflowAction.from_dict(d.pop("action"))

        environment_id = d.pop("environment_id")

        enabled = d.pop("enabled", UNSET)

        _metadata = d.pop("metadata", UNSET)
        metadata: OperatorTriggerCreateMetadata | Unset
        if isinstance(_metadata, Unset):
            metadata = UNSET
        else:
            metadata = OperatorTriggerCreateMetadata.from_dict(_metadata)

        operator_trigger_create = cls(
            name=name,
            source=source,
            action=action,
            environment_id=environment_id,
            enabled=enabled,
            metadata=metadata,
        )

        return operator_trigger_create
