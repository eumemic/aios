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

T = TypeVar("T", bound="DenyAll")


@_attrs_define
class DenyAll:
    """Explicit fail-closed — admit no one. The server default.

    Attributes:
        kind (Literal['deny_all'] | Unset):  Default: 'deny_all'.
    """

    kind: Literal["deny_all"] | Unset = "deny_all"

    def to_dict(self) -> dict[str, Any]:
        kind = self.kind

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if kind is not UNSET:
            field_dict["kind"] = kind

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        kind = cast(Literal["deny_all"] | Unset, d.pop("kind", UNSET))
        if kind != "deny_all" and not isinstance(kind, Unset):
            raise ValueError(f"kind must match const 'deny_all', got '{kind}'")

        deny_all = cls(
            kind=kind,
        )

        return deny_all
