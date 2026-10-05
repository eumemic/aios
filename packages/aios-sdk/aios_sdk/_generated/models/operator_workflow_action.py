from __future__ import annotations

from collections.abc import Mapping
from typing import (
    Any,
    Literal,
    TypeVar,
    cast,
)

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="OperatorWorkflowAction")


@_attrs_define
class OperatorWorkflowAction:
    """The workflow action of an operator trigger: ``budget_usd`` is required,
    so every operator run it launches is bounded.

        Attributes:
            workflow_id (str):
            budget_usd (float):
            kind (Literal['workflow'] | Unset):  Default: 'workflow'.
            workflow_version (int | None | Unset):
            version (int | None | Unset):
            input_template (Any | Unset):
            vault_ids (list[str] | Unset):
            max_outstanding_runs (int | None | Unset):
    """

    workflow_id: str
    budget_usd: float
    kind: Literal["workflow"] | Unset = "workflow"
    workflow_version: int | None | Unset = UNSET
    version: int | None | Unset = UNSET
    input_template: Any | Unset = UNSET
    vault_ids: list[str] | Unset = UNSET
    max_outstanding_runs: int | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        workflow_id = self.workflow_id

        budget_usd = self.budget_usd

        kind = self.kind

        workflow_version: int | None | Unset
        if isinstance(self.workflow_version, Unset):
            workflow_version = UNSET
        else:
            workflow_version = self.workflow_version

        version: int | None | Unset
        if isinstance(self.version, Unset):
            version = UNSET
        else:
            version = self.version

        input_template = self.input_template

        vault_ids: list[str] | Unset = UNSET
        if not isinstance(self.vault_ids, Unset):
            vault_ids = self.vault_ids

        max_outstanding_runs: int | None | Unset
        if isinstance(self.max_outstanding_runs, Unset):
            max_outstanding_runs = UNSET
        else:
            max_outstanding_runs = self.max_outstanding_runs

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "workflow_id": workflow_id,
                "budget_usd": budget_usd,
            }
        )
        if kind is not UNSET:
            field_dict["kind"] = kind
        if workflow_version is not UNSET:
            field_dict["workflow_version"] = workflow_version
        if version is not UNSET:
            field_dict["version"] = version
        if input_template is not UNSET:
            field_dict["input_template"] = input_template
        if vault_ids is not UNSET:
            field_dict["vault_ids"] = vault_ids
        if max_outstanding_runs is not UNSET:
            field_dict["max_outstanding_runs"] = max_outstanding_runs

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        workflow_id = d.pop("workflow_id")

        budget_usd = d.pop("budget_usd")

        kind = cast(Literal["workflow"] | Unset, d.pop("kind", UNSET))
        if kind != "workflow" and not isinstance(kind, Unset):
            raise ValueError(f"kind must match const 'workflow', got '{kind}'")

        def _parse_workflow_version(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        workflow_version = _parse_workflow_version(d.pop("workflow_version", UNSET))

        def _parse_version(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        version = _parse_version(d.pop("version", UNSET))

        input_template = d.pop("input_template", UNSET)

        vault_ids = cast(list[str], d.pop("vault_ids", UNSET))

        def _parse_max_outstanding_runs(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        max_outstanding_runs = _parse_max_outstanding_runs(
            d.pop("max_outstanding_runs", UNSET)
        )

        operator_workflow_action = cls(
            workflow_id=workflow_id,
            budget_usd=budget_usd,
            kind=kind,
            workflow_version=workflow_version,
            version=version,
            input_template=input_template,
            vault_ids=vault_ids,
            max_outstanding_runs=max_outstanding_runs,
        )

        return operator_workflow_action
