from __future__ import annotations

from collections.abc import Mapping
from typing import (
    Any,
    Literal,
    TypeVar,
    cast,
)

from attrs import define as _attrs_define

from ..models.run_completion_source_statuses_item import RunCompletionSourceStatusesItem
from ..types import UNSET, Unset

T = TypeVar("T", bound="RunCompletionSource")


@_attrs_define
class RunCompletionSource:
    """Reactive source: fires once per terminal completion of any run of the
    watched workflow whose status is in ``statuses``.

    No ``next_fire`` — never scheduled by the tick (the claim/MIN queries'
    ``next_fire IS NOT NULL`` predicate plus the DB guard make a reactive row
    unschedulable by construction); fires are dispatched from the watched
    run's completion transaction instead. The watch is account-scoped: the
    trigger is only ever handed run data its owner could already read via the
    account-scoped run reads.

        Attributes:
            workflow_id (str):
            kind (Literal['run_completion'] | Unset):  Default: 'run_completion'.
            statuses (list[RunCompletionSourceStatusesItem] | Unset):
    """

    workflow_id: str
    kind: Literal["run_completion"] | Unset = "run_completion"
    statuses: list[RunCompletionSourceStatusesItem] | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        workflow_id = self.workflow_id

        kind = self.kind

        statuses: list[str] | Unset = UNSET
        if not isinstance(self.statuses, Unset):
            statuses = []
            for statuses_item_data in self.statuses:
                statuses_item = statuses_item_data.value
                statuses.append(statuses_item)

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "workflow_id": workflow_id,
            }
        )
        if kind is not UNSET:
            field_dict["kind"] = kind
        if statuses is not UNSET:
            field_dict["statuses"] = statuses

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        workflow_id = d.pop("workflow_id")

        kind = cast(Literal["run_completion"] | Unset, d.pop("kind", UNSET))
        if kind != "run_completion" and not isinstance(kind, Unset):
            raise ValueError(f"kind must match const 'run_completion', got '{kind}'")

        _statuses = d.pop("statuses", UNSET)
        statuses: list[RunCompletionSourceStatusesItem] | Unset = UNSET
        if _statuses is not UNSET:
            statuses = []
            for statuses_item_data in _statuses:
                statuses_item = RunCompletionSourceStatusesItem(statuses_item_data)

                statuses.append(statuses_item)

        run_completion_source = cls(
            workflow_id=workflow_id,
            kind=kind,
            statuses=statuses,
        )

        return run_completion_source
