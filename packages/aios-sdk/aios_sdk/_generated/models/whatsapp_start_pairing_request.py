from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

T = TypeVar("T", bound="WhatsappStartPairingRequest")


@_attrs_define
class WhatsappStartPairingRequest:
    """
    Attributes:
        external_account_id (str):
    """

    external_account_id: str

    def to_dict(self) -> dict[str, Any]:
        external_account_id = self.external_account_id

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "external_account_id": external_account_id,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        external_account_id = d.pop("external_account_id")

        whatsapp_start_pairing_request = cls(
            external_account_id=external_account_id,
        )

        return whatsapp_start_pairing_request
