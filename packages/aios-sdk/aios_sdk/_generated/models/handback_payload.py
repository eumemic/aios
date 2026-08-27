from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="HandbackPayload")


@_attrs_define
class HandbackPayload:
    """What the human left behind: the post-takeover page snapshot, an inline
    screenshot, and which sites are now signed in (the cookie-jar-derived
    delta). ``None`` fields mean the browser died before the handback could
    be captured — the grant still closed.

        Attributes:
            snapshot (None | str | Unset):
            screenshot_data_url (None | str | Unset):
            signed_in_hosts (list[str] | Unset):
            url (None | str | Unset):
    """

    snapshot: None | str | Unset = UNSET
    screenshot_data_url: None | str | Unset = UNSET
    signed_in_hosts: list[str] | Unset = UNSET
    url: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        snapshot: None | str | Unset
        if isinstance(self.snapshot, Unset):
            snapshot = UNSET
        else:
            snapshot = self.snapshot

        screenshot_data_url: None | str | Unset
        if isinstance(self.screenshot_data_url, Unset):
            screenshot_data_url = UNSET
        else:
            screenshot_data_url = self.screenshot_data_url

        signed_in_hosts: list[str] | Unset = UNSET
        if not isinstance(self.signed_in_hosts, Unset):
            signed_in_hosts = self.signed_in_hosts

        url: None | str | Unset
        if isinstance(self.url, Unset):
            url = UNSET
        else:
            url = self.url

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if snapshot is not UNSET:
            field_dict["snapshot"] = snapshot
        if screenshot_data_url is not UNSET:
            field_dict["screenshot_data_url"] = screenshot_data_url
        if signed_in_hosts is not UNSET:
            field_dict["signed_in_hosts"] = signed_in_hosts
        if url is not UNSET:
            field_dict["url"] = url

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)

        def _parse_snapshot(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        snapshot = _parse_snapshot(d.pop("snapshot", UNSET))

        def _parse_screenshot_data_url(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        screenshot_data_url = _parse_screenshot_data_url(
            d.pop("screenshot_data_url", UNSET)
        )

        signed_in_hosts = cast(list[str], d.pop("signed_in_hosts", UNSET))

        def _parse_url(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        url = _parse_url(d.pop("url", UNSET))

        handback_payload = cls(
            snapshot=snapshot,
            screenshot_data_url=screenshot_data_url,
            signed_in_hosts=signed_in_hosts,
            url=url,
        )

        handback_payload.additional_properties = d
        return handback_payload

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
