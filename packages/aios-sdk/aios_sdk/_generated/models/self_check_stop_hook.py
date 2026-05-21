from __future__ import annotations

from collections.abc import Mapping
from typing import (
    Any,
    Literal,
    TypeVar,
    cast,
)

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="SelfCheckStopHook")


@_attrs_define
class SelfCheckStopHook:
    """Stop hook: allow stop only when the agent's final message starts with ``stop_on``.

    The ``prompt`` is appended to the augmented system prompt so the model
    is aware of the termination criterion throughout the session.  At
    each conversational end-of-turn the harness compares the assistant
    message's first non-whitespace word (case-insensitive) to ``stop_on``;
    a match allows the idle transition, a mismatch wakes the session
    with ``continuation_message``.

        Attributes:
            prompt (str):
            stop_on (str):
            type_ (Literal['self_check'] | Unset):  Default: 'self_check'.
            continuation_message (None | str | Unset):
    """

    prompt: str
    stop_on: str
    type_: Literal["self_check"] | Unset = "self_check"
    continuation_message: None | str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        prompt = self.prompt

        stop_on = self.stop_on

        type_ = self.type_

        continuation_message: None | str | Unset
        if isinstance(self.continuation_message, Unset):
            continuation_message = UNSET
        else:
            continuation_message = self.continuation_message

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "prompt": prompt,
                "stop_on": stop_on,
            }
        )
        if type_ is not UNSET:
            field_dict["type"] = type_
        if continuation_message is not UNSET:
            field_dict["continuation_message"] = continuation_message

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        prompt = d.pop("prompt")

        stop_on = d.pop("stop_on")

        type_ = cast(Literal["self_check"] | Unset, d.pop("type", UNSET))
        if type_ != "self_check" and not isinstance(type_, Unset):
            raise ValueError(f"type must match const 'self_check', got '{type_}'")

        def _parse_continuation_message(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        continuation_message = _parse_continuation_message(
            d.pop("continuation_message", UNSET)
        )

        self_check_stop_hook = cls(
            prompt=prompt,
            stop_on=stop_on,
            type_=type_,
            continuation_message=continuation_message,
        )

        return self_check_stop_hook
