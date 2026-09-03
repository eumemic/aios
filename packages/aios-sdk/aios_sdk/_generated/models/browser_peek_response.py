from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.browser_peek_page import BrowserPeekPage


T = TypeVar("T", bound="BrowserPeekResponse")


@_attrs_define
class BrowserPeekResponse:
    """A read-only look at the computer: not running, running with no page
    to show, or running with the page. Never provisions, never creates a
    page, and is refused while a human holds the computer.

        Attributes:
            running (bool):
            url (None | str | Unset):
            title (None | str | Unset):
            page (BrowserPeekPage | None | Unset):
    """

    running: bool
    url: None | str | Unset = UNSET
    title: None | str | Unset = UNSET
    page: BrowserPeekPage | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.browser_peek_page import BrowserPeekPage

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

        page: dict[str, Any] | None | Unset
        if isinstance(self.page, Unset):
            page = UNSET
        elif isinstance(self.page, BrowserPeekPage):
            page = self.page.to_dict()
        else:
            page = self.page

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
        if page is not UNSET:
            field_dict["page"] = page

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.browser_peek_page import BrowserPeekPage

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

        def _parse_page(data: object) -> BrowserPeekPage | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                page_type_0 = BrowserPeekPage.from_dict(data)

                return page_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(BrowserPeekPage | None | Unset, data)

        page = _parse_page(d.pop("page", UNSET))

        browser_peek_response = cls(
            running=running,
            url=url,
            title=title,
            page=page,
        )

        browser_peek_response.additional_properties = d
        return browser_peek_response

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
