from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.handback_payload import HandbackPayload


T = TypeVar("T", bound="TakeoverCloseResponse")


@_attrs_define
class TakeoverCloseResponse:
    """
    Attributes:
        handback (HandbackPayload): What the human left behind: the post-takeover page snapshot, an inline
            screenshot, and which sites are now signed in (the cookie-jar-derived
            delta). ``None`` fields mean the browser died before the handback could
            be captured — the grant still closed.
    """

    handback: HandbackPayload
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        handback = self.handback.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "handback": handback,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.handback_payload import HandbackPayload

        d = dict(src_dict)
        handback = HandbackPayload.from_dict(d.pop("handback"))

        takeover_close_response = cls(
            handback=handback,
        )

        takeover_close_response.additional_properties = d
        return takeover_close_response

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
