from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import (
    Any,
    Literal,
    TypeVar,
    cast,
)

from attrs import define as _attrs_define
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="OneShotSource")


@_attrs_define
class OneShotSource:
    """One-shot source: fires once at an absolute UTC time, then self-deletes.

    Attributes:
        fire_at (datetime.datetime):
        kind (Literal['one_shot'] | Unset):  Default: 'one_shot'.
    """

    fire_at: datetime.datetime
    kind: Literal["one_shot"] | Unset = "one_shot"

    def to_dict(self) -> dict[str, Any]:
        fire_at = self.fire_at.isoformat()

        kind = self.kind

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "fire_at": fire_at,
            }
        )
        if kind is not UNSET:
            field_dict["kind"] = kind

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        fire_at = isoparse(d.pop("fire_at"))

        kind = cast(Literal["one_shot"] | Unset, d.pop("kind", UNSET))
        if kind != "one_shot" and not isinstance(kind, Unset):
            raise ValueError(f"kind must match const 'one_shot', got '{kind}'")

        one_shot_source = cls(
            fire_at=fire_at,
            kind=kind,
        )

        return one_shot_source
