from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

<<<<<<<< HEAD:packages/aios-sdk/aios_sdk/_generated/models/wf_run_terminal_summary_type_0.py
T = TypeVar("T", bound="WfRunTerminalSummaryType0")


@_attrs_define
class WfRunTerminalSummaryType0:
========
T = TypeVar("T", bound="VaultCredentialMetadataType0")


@_attrs_define
class VaultCredentialMetadataType0:
>>>>>>>> fd5ad37a (Regenerate OpenAPI and SDK for nullable vault metadata):packages/aios-sdk/aios_sdk/_generated/models/vault_credential_metadata_type_0.py
    """ """

    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
<<<<<<<< HEAD:packages/aios-sdk/aios_sdk/_generated/models/wf_run_terminal_summary_type_0.py
        wf_run_terminal_summary_type_0 = cls()

        wf_run_terminal_summary_type_0.additional_properties = d
        return wf_run_terminal_summary_type_0
========
        vault_credential_metadata_type_0 = cls()

        vault_credential_metadata_type_0.additional_properties = d
        return vault_credential_metadata_type_0
>>>>>>>> fd5ad37a (Regenerate OpenAPI and SDK for nullable vault metadata):packages/aios-sdk/aios_sdk/_generated/models/vault_credential_metadata_type_0.py

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
