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

T = TypeVar("T", bound="AlwaysContinueStopHook")


@_attrs_define
class AlwaysContinueStopHook:
    """Stop hook: never honor conversational end-of-turn.

    For maintenance-mode lieutenants whose charter is to watch
    indefinitely.  The session can still be paused by external signals
    (supervisor message, interrupt, budget exhaustion).

        Attributes:
            type_ (Literal['always_continue'] | Unset):  Default: 'always_continue'.
            continuation_message (None | str | Unset):
    """

    type_: Literal["always_continue"] | Unset = "always_continue"
    continuation_message: None | str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        type_ = self.type_

        continuation_message: None | str | Unset
        if isinstance(self.continuation_message, Unset):
            continuation_message = UNSET
        else:
            continuation_message = self.continuation_message

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if type_ is not UNSET:
            field_dict["type"] = type_
        if continuation_message is not UNSET:
            field_dict["continuation_message"] = continuation_message

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        type_ = cast(Literal["always_continue"] | Unset, d.pop("type", UNSET))
        if type_ != "always_continue" and not isinstance(type_, Unset):
            raise ValueError(f"type must match const 'always_continue', got '{type_}'")

        def _parse_continuation_message(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        continuation_message = _parse_continuation_message(
            d.pop("continuation_message", UNSET)
        )

        always_continue_stop_hook = cls(
            type_=type_,
            continuation_message=continuation_message,
        )

        return always_continue_stop_hook
