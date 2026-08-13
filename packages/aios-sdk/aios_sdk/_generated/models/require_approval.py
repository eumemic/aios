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

T = TypeVar("T", bound="RequireApproval")


@_attrs_define
class RequireApproval:
    """Admit only chat IDs promoted by an operator; others become pending.

    Attributes:
        approved (list[str]):
        kind (Literal['require_approval'] | Unset):  Default: 'require_approval'.
    """

    approved: list[str]
    kind: Literal["require_approval"] | Unset = "require_approval"

    def to_dict(self) -> dict[str, Any]:
        approved = self.approved

        kind = self.kind

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "approved": approved,
            }
        )
        if kind is not UNSET:
            field_dict["kind"] = kind

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        approved = cast(list[str], d.pop("approved"))

        kind = cast(Literal["require_approval"] | Unset, d.pop("kind", UNSET))
        if kind != "require_approval" and not isinstance(kind, Unset):
            raise ValueError(f"kind must match const 'require_approval', got '{kind}'")

        require_approval = cls(
            approved=approved,
            kind=kind,
        )

        return require_approval
