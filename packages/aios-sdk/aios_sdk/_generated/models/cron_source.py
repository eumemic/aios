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

T = TypeVar("T", bound="CronSource")


@_attrs_define
class CronSource:
    """Recurring source: a standard 5-field cron expression in UTC.

    Structure-only here — grammar/occurrence checks live on the write
    models (``TriggerCreate`` / ``TriggerUpdate``) so the read path accepts
    every row the write path ever accepted (see module docstring + §2.2 of
    the design contract). A future additive ``timezone`` field (IANA name;
    absent = UTC) lands without touching this model or the DB CHECK.

        Attributes:
            schedule (str):
            kind (Literal['cron'] | Unset):  Default: 'cron'.
    """

    schedule: str
    kind: Literal["cron"] | Unset = "cron"

    def to_dict(self) -> dict[str, Any]:
        schedule = self.schedule

        kind = self.kind

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "schedule": schedule,
            }
        )
        if kind is not UNSET:
            field_dict["kind"] = kind

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        schedule = d.pop("schedule")

        kind = cast(Literal["cron"] | Unset, d.pop("kind", UNSET))
        if kind != "cron" and not isinstance(kind, Unset):
            raise ValueError(f"kind must match const 'cron', got '{kind}'")

        cron_source = cls(
            schedule=schedule,
            kind=kind,
        )

        return cron_source
