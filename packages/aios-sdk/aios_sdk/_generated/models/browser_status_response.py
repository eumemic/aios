from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.browser_takeover_status import BrowserTakeoverStatus


T = TypeVar("T", bound="BrowserStatusResponse")


@_attrs_define
class BrowserStatusResponse:
    """The account computer's state: not running, or running with its page
    and any open takeover + signed-in sites the driver reports.

        Attributes:
            running (bool):
            url (None | str | Unset):
            title (None | str | Unset):
            signed_in_hosts (list[str] | Unset):
            takeover (BrowserTakeoverStatus | None | Unset):
    """

    running: bool
    url: None | str | Unset = UNSET
    title: None | str | Unset = UNSET
    signed_in_hosts: list[str] | Unset = UNSET
    takeover: BrowserTakeoverStatus | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.browser_takeover_status import BrowserTakeoverStatus

        running = self.running

        url: None | str | Unset
        if isinstance(self.url, Unset):
            url = UNSET
        else:
            url = self.url

        title: None | str | Unset
        if isinstance(self.title, Unset):
            title = UNSET
        else:
            title = self.title

        signed_in_hosts: list[str] | Unset = UNSET
        if not isinstance(self.signed_in_hosts, Unset):
            signed_in_hosts = self.signed_in_hosts

        takeover: dict[str, Any] | None | Unset
        if isinstance(self.takeover, Unset):
            takeover = UNSET
        elif isinstance(self.takeover, BrowserTakeoverStatus):
            takeover = self.takeover.to_dict()
        else:
            takeover = self.takeover

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "running": running,
            }
        )
        if url is not UNSET:
            field_dict["url"] = url
        if title is not UNSET:
            field_dict["title"] = title
        if signed_in_hosts is not UNSET:
            field_dict["signed_in_hosts"] = signed_in_hosts
        if takeover is not UNSET:
            field_dict["takeover"] = takeover

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.browser_takeover_status import BrowserTakeoverStatus

        d = dict(src_dict)
        running = d.pop("running")

        def _parse_url(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        url = _parse_url(d.pop("url", UNSET))

        def _parse_title(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        title = _parse_title(d.pop("title", UNSET))

        signed_in_hosts = cast(list[str], d.pop("signed_in_hosts", UNSET))

        def _parse_takeover(data: object) -> BrowserTakeoverStatus | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                takeover_type_0 = BrowserTakeoverStatus.from_dict(data)

                return takeover_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(BrowserTakeoverStatus | None | Unset, data)

        takeover = _parse_takeover(d.pop("takeover", UNSET))

        browser_status_response = cls(
            running=running,
            url=url,
            title=title,
            signed_in_hosts=signed_in_hosts,
            takeover=takeover,
        )

        browser_status_response.additional_properties = d
        return browser_status_response

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
