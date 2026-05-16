from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.signal_register_response_status import SignalRegisterResponseStatus
from ..types import UNSET, Unset

T = TypeVar("T", bound="SignalRegisterResponse")


@_attrs_define
class SignalRegisterResponse:
    """``status="captcha_required"`` is a 200, not a 4xx — it's an actionable
    next step (solve the captcha, repost with the token), and 4xx would bury
    the URL inside FastAPI's error envelope.

        Attributes:
            external_account_id (str):
            status (SignalRegisterResponseStatus):
            captcha_url (None | str | Unset):
    """

    external_account_id: str
    status: SignalRegisterResponseStatus
    captcha_url: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        external_account_id = self.external_account_id

        status = self.status.value

        captcha_url: None | str | Unset
        if isinstance(self.captcha_url, Unset):
            captcha_url = UNSET
        else:
            captcha_url = self.captcha_url

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "external_account_id": external_account_id,
                "status": status,
            }
        )
        if captcha_url is not UNSET:
            field_dict["captcha_url"] = captcha_url

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        external_account_id = d.pop("external_account_id")

        status = SignalRegisterResponseStatus(d.pop("status"))

        def _parse_captcha_url(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        captcha_url = _parse_captcha_url(d.pop("captcha_url", UNSET))

        signal_register_response = cls(
            external_account_id=external_account_id,
            status=status,
            captcha_url=captcha_url,
        )

        signal_register_response.additional_properties = d
        return signal_register_response

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
