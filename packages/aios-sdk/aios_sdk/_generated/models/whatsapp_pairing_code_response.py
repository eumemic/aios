from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="WhatsappPairingCodeResponse")


@_attrs_define
class WhatsappPairingCodeResponse:
    """The QR code currently live for the in-flight pairing attempt.

    ``rotation_seq`` increments each time whatsmeow rotates the code
    (~every 20 s); operators poll this endpoint every few seconds and
    re-render the QR when ``rotation_seq`` changes.

        Attributes:
            external_account_id (str):
            code (str):
            rotation_seq (int):
    """

    external_account_id: str
    code: str
    rotation_seq: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        external_account_id = self.external_account_id

        code = self.code

        rotation_seq = self.rotation_seq

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "external_account_id": external_account_id,
                "code": code,
                "rotation_seq": rotation_seq,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        external_account_id = d.pop("external_account_id")

        code = d.pop("code")

        rotation_seq = d.pop("rotation_seq")

        whatsapp_pairing_code_response = cls(
            external_account_id=external_account_id,
            code=code,
            rotation_seq=rotation_seq,
        )

        whatsapp_pairing_code_response.additional_properties = d
        return whatsapp_pairing_code_response

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
