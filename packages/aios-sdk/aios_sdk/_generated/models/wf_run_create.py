from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="WfRunCreate")


@_attrs_define
class WfRunCreate:
    """Request body for ``POST /v1/runs`` — launch a run of a workflow.

    ``input`` is arbitrary JSON (a workflow's input need not be an object). The run
    binds to ``environment_id`` (like a session), into which its ``agent()`` children
    spawn. (``launcher_session_id`` is deliberately NOT a field — trusted ids never
    ride in request bodies; the HTTP path is always an operator launch.)

        Attributes:
            workflow_id (str):
            environment_id (str):
            input_ (Any | Unset):
            vault_ids (list[str] | Unset): Vault ids to bind to the run for credential resolution. When an agent launches
                the run, these must be a subset of the launcher's own vaults; the HTTP path is unattenuated operator authority.
            budget_usd (float | None | Unset): Optional shared USD spend ceiling for this run's direct agent() children.
            default_child_model (None | str | Unset): Optional model used by generic agent() children when they omit model=.
    """

    workflow_id: str
    environment_id: str
    input_: Any | Unset = UNSET
    vault_ids: list[str] | Unset = UNSET
    budget_usd: float | None | Unset = UNSET
    default_child_model: None | str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        workflow_id = self.workflow_id

        environment_id = self.environment_id

        input_ = self.input_

        vault_ids: list[str] | Unset = UNSET
        if not isinstance(self.vault_ids, Unset):
            vault_ids = self.vault_ids

        budget_usd: float | None | Unset
        if isinstance(self.budget_usd, Unset):
            budget_usd = UNSET
        else:
            budget_usd = self.budget_usd

        default_child_model: None | str | Unset
        if isinstance(self.default_child_model, Unset):
            default_child_model = UNSET
        else:
            default_child_model = self.default_child_model

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "workflow_id": workflow_id,
                "environment_id": environment_id,
            }
        )
        if input_ is not UNSET:
            field_dict["input"] = input_
        if vault_ids is not UNSET:
            field_dict["vault_ids"] = vault_ids
        if budget_usd is not UNSET:
            field_dict["budget_usd"] = budget_usd
        if default_child_model is not UNSET:
            field_dict["default_child_model"] = default_child_model

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        workflow_id = d.pop("workflow_id")

        environment_id = d.pop("environment_id")

        input_ = d.pop("input", UNSET)

        vault_ids = cast(list[str], d.pop("vault_ids", UNSET))

        def _parse_budget_usd(data: object) -> float | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(float | None | Unset, data)

        budget_usd = _parse_budget_usd(d.pop("budget_usd", UNSET))

        def _parse_default_child_model(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        default_child_model = _parse_default_child_model(
            d.pop("default_child_model", UNSET)
        )

        wf_run_create = cls(
            workflow_id=workflow_id,
            environment_id=environment_id,
            input_=input_,
            vault_ids=vault_ids,
            budget_usd=budget_usd,
            default_child_model=default_child_model,
        )

        return wf_run_create
