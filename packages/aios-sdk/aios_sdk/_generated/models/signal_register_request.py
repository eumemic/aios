from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="SignalRegisterRequest")


@_attrs_define
class SignalRegisterRequest:
    """
    Attributes:
        external_account_id (str):
        captcha (None | str | Unset):
        voice (bool | Unset):  Default: False.
    """

    external_account_id: str
    captcha: None | str | Unset = UNSET
    voice: bool | Unset = False

    def to_dict(self) -> dict[str, Any]:
        external_account_id = self.external_account_id

        captcha: None | str | Unset
        if isinstance(self.captcha, Unset):
            captcha = UNSET
        else:
            captcha = self.captcha

        voice = self.voice

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "external_account_id": external_account_id,
            }
        )
        if captcha is not UNSET:
            field_dict["captcha"] = captcha
        if voice is not UNSET:
            field_dict["voice"] = voice

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        external_account_id = d.pop("external_account_id")

        def _parse_captcha(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        captcha = _parse_captcha(d.pop("captcha", UNSET))

        voice = d.pop("voice", UNSET)

        signal_register_request = cls(
            external_account_id=external_account_id,
            captcha=captcha,
            voice=voice,
        )

        return signal_register_request
