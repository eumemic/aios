from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.browser_peek_page_security_type_0 import BrowserPeekPageSecurityType0
from ..types import UNSET, Unset

T = TypeVar("T", bound="BrowserPeekPage")


@_attrs_define
class BrowserPeekPage:
    """One JPEG of a page's viewport plus its trusted chrome. ``origin`` and
    ``security`` come from the driver's committed URL, never the pixels.

        Attributes:
            jpeg_b64 (str):
            w (int):
            h (int):
            origin (None | str | Unset):
            security (BrowserPeekPageSecurityType0 | None | Unset):
    """

    jpeg_b64: str
    w: int
    h: int
    origin: None | str | Unset = UNSET
    security: BrowserPeekPageSecurityType0 | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        jpeg_b64 = self.jpeg_b64

        w = self.w

        h = self.h

        origin: None | str | Unset
        if isinstance(self.origin, Unset):
            origin = UNSET
        else:
            origin = self.origin

        security: None | str | Unset
        if isinstance(self.security, Unset):
            security = UNSET
        elif isinstance(self.security, BrowserPeekPageSecurityType0):
            security = self.security.value
        else:
            security = self.security

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "jpeg_b64": jpeg_b64,
                "w": w,
                "h": h,
            }
        )
        if origin is not UNSET:
            field_dict["origin"] = origin
        if security is not UNSET:
            field_dict["security"] = security

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        jpeg_b64 = d.pop("jpeg_b64")

        w = d.pop("w")

        h = d.pop("h")

        def _parse_origin(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        origin = _parse_origin(d.pop("origin", UNSET))

        def _parse_security(
            data: object,
        ) -> BrowserPeekPageSecurityType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                security_type_0 = BrowserPeekPageSecurityType0(data)

                return security_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(BrowserPeekPageSecurityType0 | None | Unset, data)

        security = _parse_security(d.pop("security", UNSET))

        browser_peek_page = cls(
            jpeg_b64=jpeg_b64,
            w=w,
            h=h,
            origin=origin,
            security=security,
        )

        browser_peek_page.additional_properties = d
        return browser_peek_page

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
