from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="MintAccountResponse")


@_attrs_define
class MintAccountResponse:
    """
    Attributes:
        account_id (str):
        key_id (str):
        plaintext_key (str):
    """

    account_id: str
    key_id: str
    plaintext_key: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        account_id = self.account_id

        key_id = self.key_id

        plaintext_key = self.plaintext_key

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "account_id": account_id,
                "key_id": key_id,
                "plaintext_key": plaintext_key,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        account_id = d.pop("account_id")

        key_id = d.pop("key_id")

        plaintext_key = d.pop("plaintext_key")

        mint_account_response = cls(
            account_id=account_id,
            key_id=key_id,
            plaintext_key=plaintext_key,
        )

        mint_account_response.additional_properties = d
        return mint_account_response

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
