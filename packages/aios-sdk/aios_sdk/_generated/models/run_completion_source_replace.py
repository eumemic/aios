from __future__ import annotations

from collections.abc import Mapping
from typing import (
    Any,
    Literal,
    TypeVar,
    cast,
)

from attrs import define as _attrs_define

from ..models.run_completion_source_replace_statuses_item import (
    RunCompletionSourceReplaceStatusesItem,
)
from ..types import UNSET, Unset

T = TypeVar("T", bound="RunCompletionSourceReplace")


@_attrs_define
class RunCompletionSourceReplace:
    """Update-side variant (§2.2 Replace rule): ``statuses`` is REQUIRED, so a
    partial source on update 422s instead of silently resetting a narrowed
    filter back to all-three. (The first SOURCE member with a defaulted field,
    hence the first ``TriggerSourceReplace`` union.)

        Attributes:
            workflow_id (str):
            statuses (list[RunCompletionSourceReplaceStatusesItem]):
            kind (Literal['run_completion'] | Unset):  Default: 'run_completion'.
    """

    workflow_id: str
    statuses: list[RunCompletionSourceReplaceStatusesItem]
    kind: Literal["run_completion"] | Unset = "run_completion"

    def to_dict(self) -> dict[str, Any]:
        workflow_id = self.workflow_id

        statuses = []
        for statuses_item_data in self.statuses:
            statuses_item = statuses_item_data.value
            statuses.append(statuses_item)

        kind = self.kind

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "workflow_id": workflow_id,
                "statuses": statuses,
            }
        )
        if kind is not UNSET:
            field_dict["kind"] = kind

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        workflow_id = d.pop("workflow_id")

        statuses = []
        _statuses = d.pop("statuses")
        for statuses_item_data in _statuses:
            statuses_item = RunCompletionSourceReplaceStatusesItem(statuses_item_data)

            statuses.append(statuses_item)

        kind = cast(Literal["run_completion"] | Unset, d.pop("kind", UNSET))
        if kind != "run_completion" and not isinstance(kind, Unset):
            raise ValueError(f"kind must match const 'run_completion', got '{kind}'")

        run_completion_source_replace = cls(
            workflow_id=workflow_id,
            statuses=statuses,
            kind=kind,
        )

        return run_completion_source_replace
