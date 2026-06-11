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

T = TypeVar("T", bound="WorkflowAction")


@_attrs_define
class WorkflowAction:
    """Launch a run of ``workflow_id`` at fire time — deterministic, no model
    wake.

    The run's input is ALWAYS the envelope ``{"trigger": <firing context>,
    "input": <input_template verbatim>}`` — no placeholder substitution. A
    workflow built to be triggered reads ``input["trigger"]`` for the firing
    context (for run_completion fires: the completing run's id, status,
    output, and error kind) and ``input["input"]`` for this template.

    ``environment_id`` is deliberately NOT a field: the run always binds to
    the owner session's environment, resolved at write time into the
    first-class ``triggers.environment_id`` column (sessions' environment is
    immutable, so write-time freeze equals fire-time resolution). Anything on
    this model is agent-reachable through the ``trigger_create`` tool — a
    caller-chosen environment would bypass the same-stance refusal on the
    ``create_run`` builtin.

    ``workflow_version``: ``None`` = run the workflow's CURRENT version at
    each fire (float); an integer is a DRIFT ASSERTION — it must equal the
    workflow's current version at write, and a fire whose workflow has since
    been edited records an error instead of running the unreviewed script
    (workflows have no version-history table: a pin cannot resolve an old
    script, only refuse a new one).

    This member is STRUCTURE-ONLY: the ``input_template`` size bound lives on
    the write models — see :func:`_validate_input_template_bound` for why a
    read-side byte bound is unsafe.

        Attributes:
            workflow_id (str):
            kind (Literal['workflow'] | Unset):  Default: 'workflow'.
            workflow_version (int | None | Unset):
            input_template (Any | Unset):
            vault_ids (list[str] | Unset):
    """

    workflow_id: str
    kind: Literal["workflow"] | Unset = "workflow"
    workflow_version: int | None | Unset = UNSET
    input_template: Any | Unset = UNSET
    vault_ids: list[str] | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        workflow_id = self.workflow_id

        kind = self.kind

        workflow_version: int | None | Unset
        if isinstance(self.workflow_version, Unset):
            workflow_version = UNSET
        else:
            workflow_version = self.workflow_version

        input_template = self.input_template

        vault_ids: list[str] | Unset = UNSET
        if not isinstance(self.vault_ids, Unset):
            vault_ids = self.vault_ids

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "workflow_id": workflow_id,
            }
        )
        if kind is not UNSET:
            field_dict["kind"] = kind
        if workflow_version is not UNSET:
            field_dict["workflow_version"] = workflow_version
        if input_template is not UNSET:
            field_dict["input_template"] = input_template
        if vault_ids is not UNSET:
            field_dict["vault_ids"] = vault_ids

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        workflow_id = d.pop("workflow_id")

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

        input_template = d.pop("input_template", UNSET)

        vault_ids = cast(list[str], d.pop("vault_ids", UNSET))

        workflow_action = cls(
            workflow_id=workflow_id,
            kind=kind,
            workflow_version=workflow_version,
            input_template=input_template,
            vault_ids=vault_ids,
        )

        return workflow_action
