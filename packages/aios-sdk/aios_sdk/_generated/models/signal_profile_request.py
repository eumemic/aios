from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="SignalProfileRequest")


@_attrs_define
class SignalProfileRequest:
    """
    Attributes:
        external_account_id (str):
        given_name (None | str | Unset):
        family_name (None | str | Unset):
        about (None | str | Unset):
    """

    external_account_id: str
    given_name: None | str | Unset = UNSET
    family_name: None | str | Unset = UNSET
    about: None | str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        external_account_id = self.external_account_id

        given_name: None | str | Unset
        if isinstance(self.given_name, Unset):
            given_name = UNSET
        else:
            given_name = self.given_name

        family_name: None | str | Unset
        if isinstance(self.family_name, Unset):
            family_name = UNSET
        else:
            family_name = self.family_name

        about: None | str | Unset
        if isinstance(self.about, Unset):
            about = UNSET
        else:
            about = self.about

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "external_account_id": external_account_id,
            }
        )
        if given_name is not UNSET:
            field_dict["given_name"] = given_name
        if family_name is not UNSET:
            field_dict["family_name"] = family_name
        if about is not UNSET:
            field_dict["about"] = about

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        external_account_id = d.pop("external_account_id")

        def _parse_given_name(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        given_name = _parse_given_name(d.pop("given_name", UNSET))

        def _parse_family_name(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        family_name = _parse_family_name(d.pop("family_name", UNSET))

        def _parse_about(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        about = _parse_about(d.pop("about", UNSET))

        signal_profile_request = cls(
            external_account_id=external_account_id,
            given_name=given_name,
            family_name=family_name,
            about=about,
        )

        return signal_profile_request
