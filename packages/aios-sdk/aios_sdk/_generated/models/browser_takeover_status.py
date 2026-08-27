from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="BrowserTakeoverStatus")


@_attrs_define
class BrowserTakeoverStatus:
    """The open grant on this account's computer, if any.

    Attributes:
        grant_id (str):
        session_id (str):
        reason (str):
        epoch (int):
        boot (str):
        created_at (str):
    """

    grant_id: str
    session_id: str
    reason: str
    epoch: int
    boot: str
    created_at: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        grant_id = self.grant_id

        session_id = self.session_id

        reason = self.reason

        epoch = self.epoch

        boot = self.boot

        created_at = self.created_at

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "grant_id": grant_id,
                "session_id": session_id,
                "reason": reason,
                "epoch": epoch,
                "boot": boot,
                "created_at": created_at,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        grant_id = d.pop("grant_id")

        session_id = d.pop("session_id")

        reason = d.pop("reason")

        epoch = d.pop("epoch")

        boot = d.pop("boot")

        created_at = d.pop("created_at")

        browser_takeover_status = cls(
            grant_id=grant_id,
            session_id=session_id,
            reason=reason,
            epoch=epoch,
            boot=boot,
            created_at=created_at,
        )

        browser_takeover_status.additional_properties = d
        return browser_takeover_status

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
