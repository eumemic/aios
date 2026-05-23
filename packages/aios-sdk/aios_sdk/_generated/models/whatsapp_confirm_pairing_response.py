from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.whatsapp_confirm_pairing_response_status import (
    WhatsappConfirmPairingResponseStatus,
)
from ..types import UNSET, Unset

T = TypeVar("T", bound="WhatsappConfirmPairingResponse")


@_attrs_define
class WhatsappConfirmPairingResponse:
    """``status`` is the terminal pairing outcome; ``jid`` / ``push_name``
    are populated on success, ``reason`` on error / timeout.

        Attributes:
            external_account_id (str):
            status (WhatsappConfirmPairingResponseStatus):
            jid (None | str | Unset):
            push_name (None | str | Unset):
            reason (None | str | Unset):
    """

    external_account_id: str
    status: WhatsappConfirmPairingResponseStatus
    jid: None | str | Unset = UNSET
    push_name: None | str | Unset = UNSET
    reason: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        external_account_id = self.external_account_id

        status = self.status.value

        jid: None | str | Unset
        if isinstance(self.jid, Unset):
            jid = UNSET
        else:
            jid = self.jid

        push_name: None | str | Unset
        if isinstance(self.push_name, Unset):
            push_name = UNSET
        else:
            push_name = self.push_name

        reason: None | str | Unset
        if isinstance(self.reason, Unset):
            reason = UNSET
        else:
            reason = self.reason

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "external_account_id": external_account_id,
                "status": status,
            }
        )
        if jid is not UNSET:
            field_dict["jid"] = jid
        if push_name is not UNSET:
            field_dict["push_name"] = push_name
        if reason is not UNSET:
            field_dict["reason"] = reason

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        external_account_id = d.pop("external_account_id")

        status = WhatsappConfirmPairingResponseStatus(d.pop("status"))

        def _parse_jid(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        jid = _parse_jid(d.pop("jid", UNSET))

        def _parse_push_name(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        push_name = _parse_push_name(d.pop("push_name", UNSET))

        def _parse_reason(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        reason = _parse_reason(d.pop("reason", UNSET))

        whatsapp_confirm_pairing_response = cls(
            external_account_id=external_account_id,
            status=status,
            jid=jid,
            push_name=push_name,
            reason=reason,
        )

        whatsapp_confirm_pairing_response.additional_properties = d
        return whatsapp_confirm_pairing_response

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
