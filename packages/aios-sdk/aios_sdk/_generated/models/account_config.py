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
            spend_limit_usd (float | None | Unset): Lifetime USD spend limit for this account. Unset inherits the parent
                account's limit; the root falls back to the server default. The spend meter never resets — raise the limit to
                grant more spend.
            sandbox_snapshot_bytes (int | None | Unset): Per-account durable-sandbox snapshot cap, in unique bytes. When
                this account's total snapshot bytes exceed the cap, the snapshot GC evicts its MOST-DORMANT sessions' snapshots
                first (each with a model-visible sandbox_fs_expired {account_cap} notice) until the account is back under cap.
                Unset inherits the nearest configured ancestor's cap; no cap anywhere ⇒ unbounded (the per-host pool budget
                still applies).
    """

    timezone: None | str | Unset = UNSET
    spend_limit_usd: float | None | Unset = UNSET
    sandbox_snapshot_bytes: int | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        timezone: None | str | Unset
        if isinstance(self.timezone, Unset):
            timezone = UNSET
        else:
            timezone = self.timezone

        spend_limit_usd: float | None | Unset
        if isinstance(self.spend_limit_usd, Unset):
            spend_limit_usd = UNSET
        else:
            spend_limit_usd = self.spend_limit_usd

        sandbox_snapshot_bytes: int | None | Unset
        if isinstance(self.sandbox_snapshot_bytes, Unset):
            sandbox_snapshot_bytes = UNSET
        else:
            sandbox_snapshot_bytes = self.sandbox_snapshot_bytes

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if timezone is not UNSET:
            field_dict["timezone"] = timezone
        if spend_limit_usd is not UNSET:
            field_dict["spend_limit_usd"] = spend_limit_usd
        if sandbox_snapshot_bytes is not UNSET:
            field_dict["sandbox_snapshot_bytes"] = sandbox_snapshot_bytes

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

        def _parse_spend_limit_usd(data: object) -> float | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(float | None | Unset, data)

        spend_limit_usd = _parse_spend_limit_usd(d.pop("spend_limit_usd", UNSET))

        def _parse_sandbox_snapshot_bytes(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        sandbox_snapshot_bytes = _parse_sandbox_snapshot_bytes(
            d.pop("sandbox_snapshot_bytes", UNSET)
        )

        account_config = cls(
            timezone=timezone,
            spend_limit_usd=spend_limit_usd,
            sandbox_snapshot_bytes=sandbox_snapshot_bytes,
        )

        return account_config
