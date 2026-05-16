from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="SignalVerifyRequest")


@_attrs_define
class SignalVerifyRequest:
    """
    Attributes:
        external_account_id (str):
        code (str):
        pin (None | str | Unset):
    """

    external_account_id: str
    code: str
    pin: None | str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        external_account_id = self.external_account_id

        code = self.code

        pin: None | str | Unset
        if isinstance(self.pin, Unset):
            pin = UNSET
        else:
            pin = self.pin

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "external_account_id": external_account_id,
                "code": code,
            }
        )
        if pin is not UNSET:
            field_dict["pin"] = pin

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        external_account_id = d.pop("external_account_id")

        code = d.pop("code")

        def _parse_pin(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        pin = _parse_pin(d.pop("pin", UNSET))

        signal_verify_request = cls(
            external_account_id=external_account_id,
            code=code,
            pin=pin,
        )

        return signal_verify_request
