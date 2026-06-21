from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

T = TypeVar("T", bound="NativeButtons")


@_attrs_define
class NativeButtons:
    """Present == connector renders externally-executed tool-confirmations as
    inbound platform buttons; a button-tap maps to the existing
    ``POST /sessions/:id/tool-confirmations``.  Absent == text-prompt fallback.

        Attributes:
            max_buttons (int):
    """

    max_buttons: int

    def to_dict(self) -> dict[str, Any]:
        max_buttons = self.max_buttons

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "max_buttons": max_buttons,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        max_buttons = d.pop("max_buttons")

        native_buttons = cls(
            max_buttons=max_buttons,
        )

        return native_buttons
