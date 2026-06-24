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

T = TypeVar("T", bound="AllowAll")


@_attrs_define
class AllowAll:
    """Explicit "anyone may talk to this agent" acknowledgement.

    Attributes:
        kind (Literal['allow_all'] | Unset):  Default: 'allow_all'.
    """

    kind: Literal["allow_all"] | Unset = "allow_all"

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
        kind = cast(Literal["allow_all"] | Unset, d.pop("kind", UNSET))
        if kind != "allow_all" and not isinstance(kind, Unset):
            raise ValueError(f"kind must match const 'allow_all', got '{kind}'")

        allow_all = cls(
            kind=kind,
        )

        return allow_all
