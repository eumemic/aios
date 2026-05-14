from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="SignalRegisterRequest")


@_attrs_define
class SignalRegisterRequest:
    """Body for ``POST /v1/connectors/signal/register``.

    Attributes:
        account (str):
        captcha (None | str | Unset):
        voice (bool | Unset):  Default: False.
    """

    account: str
    captcha: None | str | Unset = UNSET
    voice: bool | Unset = False

    def to_dict(self) -> dict[str, Any]:
        account = self.account

        captcha: None | str | Unset
        if isinstance(self.captcha, Unset):
            captcha = UNSET
        else:
            captcha = self.captcha

        voice = self.voice

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "account": account,
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
        account = d.pop("account")

        def _parse_captcha(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        captcha = _parse_captcha(d.pop("captcha", UNSET))

        voice = d.pop("voice", UNSET)

        signal_register_request = cls(
            account=account,
            captcha=captcha,
            voice=voice,
        )

        return signal_register_request
