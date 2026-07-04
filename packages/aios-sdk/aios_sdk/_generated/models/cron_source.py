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
    """Recurring source: a standard 5-field cron expression.

    Structure-only here — grammar/occurrence checks live on the write
    models (``TriggerCreate`` / ``TriggerUpdate``) so the read path accepts
    every row the write path ever accepted (see module docstring + §2.2 of
    the design contract). The additive ``timezone`` field (IANA name)
    interprets the cron wall-clock; absent = UTC.

        Attributes:
            schedule (str):
            kind (Literal['cron'] | Unset):  Default: 'cron'.
            timezone (None | str | Unset): IANA timezone name (e.g. 'America/Los_Angeles') the cron wall-clock is
                interpreted in. Unset = UTC. '0 9 * * *' with timezone 'America/New_York' fires at 9am Eastern, DST-aware.
    """

    schedule: str
    kind: Literal["cron"] | Unset = "cron"
    timezone: None | str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        schedule = self.schedule

        kind = self.kind

        timezone: None | str | Unset
        if isinstance(self.timezone, Unset):
            timezone = UNSET
        else:
            timezone = self.timezone

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "schedule": schedule,
            }
        )
        if kind is not UNSET:
            field_dict["kind"] = kind
        if timezone is not UNSET:
            field_dict["timezone"] = timezone

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        schedule = d.pop("schedule")

        kind = cast(Literal["cron"] | Unset, d.pop("kind", UNSET))
        if kind != "cron" and not isinstance(kind, Unset):
            raise ValueError(f"kind must match const 'cron', got '{kind}'")

        def _parse_timezone(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        timezone = _parse_timezone(d.pop("timezone", UNSET))

        cron_source = cls(
            schedule=schedule,
            kind=kind,
            timezone=timezone,
        )

        return cron_source
