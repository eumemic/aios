from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="UpdateAccountRequest")


@_attrs_define
class UpdateAccountRequest:
    """Body for ``PATCH /v1/accounts/{id}``.

    Partial update: omitted fields are preserved. Both fields are
    optional; at least one must be non-null. Submitting both as null
    is a no-op that returns the account row unchanged.

        Attributes:
            display_name (None | str | Unset):
            can_mint_children (bool | None | Unset):
    """

    display_name: None | str | Unset = UNSET
    can_mint_children: bool | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        display_name: None | str | Unset
        if isinstance(self.display_name, Unset):
            display_name = UNSET
        else:
            display_name = self.display_name

        can_mint_children: bool | None | Unset
        if isinstance(self.can_mint_children, Unset):
            can_mint_children = UNSET
        else:
            can_mint_children = self.can_mint_children

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if display_name is not UNSET:
            field_dict["display_name"] = display_name
        if can_mint_children is not UNSET:
            field_dict["can_mint_children"] = can_mint_children

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)

        def _parse_display_name(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        display_name = _parse_display_name(d.pop("display_name", UNSET))

        def _parse_can_mint_children(data: object) -> bool | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(bool | None | Unset, data)

        can_mint_children = _parse_can_mint_children(d.pop("can_mint_children", UNSET))

        update_account_request = cls(
            display_name=display_name,
            can_mint_children=can_mint_children,
        )

        return update_account_request
