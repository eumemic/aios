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

T = TypeVar("T", bound="WorkflowActionReplace")


@_attrs_define
class WorkflowActionReplace:
    """Update-side variant (§2.2): optional-at-create fields are REQUIRED, so
    a partial action 422s instead of silently flipping a pin to float, nulling
    the template, or dropping vault bindings. Explicit null/[] are explicit.

        Attributes:
            workflow_id (str):
            workflow_version (int | None):
            version (int | None):
            input_template (Any):
            vault_ids (list[str]):
            kind (Literal['workflow'] | Unset):  Default: 'workflow'.
    """

    workflow_id: str
    workflow_version: int | None
    version: int | None
    input_template: Any
    vault_ids: list[str]
    kind: Literal["workflow"] | Unset = "workflow"

    def to_dict(self) -> dict[str, Any]:
        workflow_id = self.workflow_id

        workflow_version: int | None
        workflow_version = self.workflow_version

        version: int | None
        version = self.version

        input_template = self.input_template

        vault_ids = self.vault_ids

        kind = self.kind

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "workflow_id": workflow_id,
                "workflow_version": workflow_version,
                "version": version,
                "input_template": input_template,
                "vault_ids": vault_ids,
            }
        )
        if kind is not UNSET:
            field_dict["kind"] = kind

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        workflow_id = d.pop("workflow_id")

        def _parse_workflow_version(data: object) -> int | None:
            if data is None:
                return data
            return cast(int | None, data)

        workflow_version = _parse_workflow_version(d.pop("workflow_version"))

        def _parse_version(data: object) -> int | None:
            if data is None:
                return data
            return cast(int | None, data)

        version = _parse_version(d.pop("version"))

        input_template = d.pop("input_template")

        vault_ids = cast(list[str], d.pop("vault_ids"))

        kind = cast(Literal["workflow"] | Unset, d.pop("kind", UNSET))
        if kind != "workflow" and not isinstance(kind, Unset):
            raise ValueError(f"kind must match const 'workflow', got '{kind}'")

        workflow_action_replace = cls(
            workflow_id=workflow_id,
            workflow_version=workflow_version,
            version=version,
            input_template=input_template,
            vault_ids=vault_ids,
            kind=kind,
        )

        return workflow_action_replace
