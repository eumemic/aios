from __future__ import annotations

from collections.abc import Mapping
from typing import (
    Any,
    Literal,
    TypeVar,
    cast,
)

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="UnrestrictedNetworking")


@_attrs_define
class UnrestrictedNetworking:
    """Full outbound network access (default).

    Attributes:
        type_ (Literal['unrestricted'] | Unset):  Default: 'unrestricted'.
    """

    type_: Literal["unrestricted"] | Unset = "unrestricted"

    def to_dict(self) -> dict[str, Any]:
        type_ = self.type_

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if type_ is not UNSET:
            field_dict["type"] = type_

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        type_ = cast(Literal["unrestricted"] | Unset, d.pop("type", UNSET))
        if type_ != "unrestricted" and not isinstance(type_, Unset):
            raise ValueError(f"type must match const 'unrestricted', got '{type_}'")

        unrestricted_networking = cls(
            type_=type_,
        )

        return unrestricted_networking
