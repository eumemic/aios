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

T = TypeVar("T", bound="CronSourceReplace")


@_attrs_define
class CronSourceReplace:
    """Update-side variant (§2.2 Replace rule): ``timezone`` is REQUIRED, so a
    partial cron source on update 422s instead of silently resetting a stored
    non-UTC zone to UTC (a silent shift of every future fire time). ``null`` is
    an explicit, in-band choice of UTC and must be sent deliberately — create
    keeps the default for tool ergonomics, mirroring the
    :class:`RunCompletionSourceReplace` / :class:`SandboxCommandActionReplace`
    / :class:`WorkflowActionReplace` siblings.

        Attributes:
            schedule (str):
            timezone (None | str): IANA timezone name (e.g. 'America/New_York') the cron wall-clock is interpreted in.
                Required on update — ``null`` explicitly means UTC (no implicit default; send the complete object).
            kind (Literal['cron'] | Unset):  Default: 'cron'.
    """

    schedule: str
    timezone: None | str
    kind: Literal["cron"] | Unset = "cron"

    def to_dict(self) -> dict[str, Any]:
        schedule = self.schedule

        timezone: None | str
        timezone = self.timezone

        kind = self.kind

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "schedule": schedule,
                "timezone": timezone,
            }
        )
        if kind is not UNSET:
            field_dict["kind"] = kind

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        schedule = d.pop("schedule")

        def _parse_timezone(data: object) -> None | str:
            if data is None:
                return data
            return cast(None | str, data)

        timezone = _parse_timezone(d.pop("timezone"))

        kind = cast(Literal["cron"] | Unset, d.pop("kind", UNSET))
        if kind != "cron" and not isinstance(kind, Unset):
            raise ValueError(f"kind must match const 'cron', got '{kind}'")

        cron_source_replace = cls(
            schedule=schedule,
            timezone=timezone,
            kind=kind,
        )

        return cron_source_replace
