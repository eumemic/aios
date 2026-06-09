from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="WfRunCreate")


@_attrs_define
class WfRunCreate:
    """Request body for ``POST /v1/runs`` — launch a run of a workflow.

    ``input`` is arbitrary JSON (a workflow's input need not be an object). The run
    binds to ``environment_id`` (like a session), into which its ``agent()`` children
    spawn.

        Attributes:
            workflow_id (str):
            environment_id (str):
            input_ (Any | Unset):
            vault_ids (list[str] | Unset): Vault ids to bind to the run for credential resolution. When an agent launches
                the run, these must be a subset of the launcher's own vaults; the HTTP path is unattenuated operator authority.
    """

    workflow_id: str
    environment_id: str
    input_: Any | Unset = UNSET
    vault_ids: list[str] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        workflow_id = self.workflow_id

        environment_id = self.environment_id

        input_ = self.input_

        vault_ids: list[str] | Unset = UNSET
        if not isinstance(self.vault_ids, Unset):
            vault_ids = self.vault_ids

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
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

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        workflow_id = d.pop("workflow_id")

        environment_id = d.pop("environment_id")

        input_ = d.pop("input", UNSET)

        vault_ids = cast(list[str], d.pop("vault_ids", UNSET))

        wf_run_create = cls(
            workflow_id=workflow_id,
            environment_id=environment_id,
            input_=input_,
            vault_ids=vault_ids,
        )

        wf_run_create.additional_properties = d
        return wf_run_create

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
