from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="AccountConfig")


@_attrs_define
class AccountConfig:
    """Per-account configuration bag.

    An unset item inherits from the parent account; see
    ``queries.resolve_effective_timezone``. Update semantics (per-item merge)
    are documented on ``UpdateAccountRequest``.

        Attributes:
            timezone (None | str | Unset): IANA timezone name (e.g. 'America/Los_Angeles') used to render the per-message
                received-at timestamp for this account's agents. Unset inherits the parent account's timezone; the root falls
                back to UTC.
    """

    timezone: None | str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        timezone: None | str | Unset
        if isinstance(self.timezone, Unset):
            timezone = UNSET
        else:
            timezone = self.timezone

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if timezone is not UNSET:
            field_dict["timezone"] = timezone

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)

        def _parse_timezone(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        timezone = _parse_timezone(d.pop("timezone", UNSET))

        account_config = cls(
            timezone=timezone,
        )

        return account_config
