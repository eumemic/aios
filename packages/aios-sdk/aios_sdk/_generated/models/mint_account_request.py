from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="MintAccountRequest")


@_attrs_define
class MintAccountRequest:
    """
    Attributes:
        display_name (str):
        can_mint_children (bool | Unset):  Default: False.
    """

    display_name: str
    can_mint_children: bool | Unset = False

    def to_dict(self) -> dict[str, Any]:
        display_name = self.display_name

        can_mint_children = self.can_mint_children

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "display_name": display_name,
            }
        )
        if can_mint_children is not UNSET:
            field_dict["can_mint_children"] = can_mint_children

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        display_name = d.pop("display_name")

        can_mint_children = d.pop("can_mint_children", UNSET)

        mint_account_request = cls(
            display_name=display_name,
            can_mint_children=can_mint_children,
        )

        return mint_account_request
