from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.input_event_button_type_0 import InputEventButtonType0
from ..models.input_event_type import InputEventType
from ..types import UNSET, Unset

T = TypeVar("T", bound="InputEvent")


@_attrs_define
class InputEvent:
    """One raw viewer input event (§5.6 vocabulary).

    The nav quartet (``navigate``/``back``/``forward``/``reload``) is the
    viewer's browser chrome — URL bar and nav buttons — riding the same spool
    as raw input so a typed URL stays off the event log exactly like a typed
    password. The driver guards ``navigate`` (public http(s) only) and treats
    the rest as history moves needing no URL.

        Attributes:
            type_ (InputEventType):
            x (float | None | Unset):
            y (float | None | Unset):
            button (InputEventButtonType0 | None | Unset):
            dx (float | None | Unset):
            dy (float | None | Unset):
            key (None | str | Unset):
            text (None | str | Unset):
            url (None | str | Unset):
    """

    type_: InputEventType
    x: float | None | Unset = UNSET
    y: float | None | Unset = UNSET
    button: InputEventButtonType0 | None | Unset = UNSET
    dx: float | None | Unset = UNSET
    dy: float | None | Unset = UNSET
    key: None | str | Unset = UNSET
    text: None | str | Unset = UNSET
    url: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        type_ = self.type_.value

        x: float | None | Unset
        if isinstance(self.x, Unset):
            x = UNSET
        else:
            x = self.x

        y: float | None | Unset
        if isinstance(self.y, Unset):
            y = UNSET
        else:
            y = self.y

        button: None | str | Unset
        if isinstance(self.button, Unset):
            button = UNSET
        elif isinstance(self.button, InputEventButtonType0):
            button = self.button.value
        else:
            button = self.button

        dx: float | None | Unset
        if isinstance(self.dx, Unset):
            dx = UNSET
        else:
            dx = self.dx

        dy: float | None | Unset
        if isinstance(self.dy, Unset):
            dy = UNSET
        else:
            dy = self.dy

        key: None | str | Unset
        if isinstance(self.key, Unset):
            key = UNSET
        else:
            key = self.key

        text: None | str | Unset
        if isinstance(self.text, Unset):
            text = UNSET
        else:
            text = self.text

        url: None | str | Unset
        if isinstance(self.url, Unset):
            url = UNSET
        else:
            url = self.url

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "type": type_,
            }
        )
        if x is not UNSET:
            field_dict["x"] = x
        if y is not UNSET:
            field_dict["y"] = y
        if button is not UNSET:
            field_dict["button"] = button
        if dx is not UNSET:
            field_dict["dx"] = dx
        if dy is not UNSET:
            field_dict["dy"] = dy
        if key is not UNSET:
            field_dict["key"] = key
        if text is not UNSET:
            field_dict["text"] = text
        if url is not UNSET:
            field_dict["url"] = url

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        type_ = InputEventType(d.pop("type"))

        def _parse_x(data: object) -> float | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(float | None | Unset, data)

        x = _parse_x(d.pop("x", UNSET))

        def _parse_y(data: object) -> float | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(float | None | Unset, data)

        y = _parse_y(d.pop("y", UNSET))

        def _parse_button(data: object) -> InputEventButtonType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                button_type_0 = InputEventButtonType0(data)

                return button_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(InputEventButtonType0 | None | Unset, data)

        button = _parse_button(d.pop("button", UNSET))

        def _parse_dx(data: object) -> float | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(float | None | Unset, data)

        dx = _parse_dx(d.pop("dx", UNSET))

        def _parse_dy(data: object) -> float | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(float | None | Unset, data)

        dy = _parse_dy(d.pop("dy", UNSET))

        def _parse_key(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        key = _parse_key(d.pop("key", UNSET))

        def _parse_text(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        text = _parse_text(d.pop("text", UNSET))

        def _parse_url(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        url = _parse_url(d.pop("url", UNSET))

        input_event = cls(
            type_=type_,
            x=x,
            y=y,
            button=button,
            dx=dx,
            dy=dy,
            key=key,
            text=text,
            url=url,
        )

        input_event.additional_properties = d
        return input_event

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
