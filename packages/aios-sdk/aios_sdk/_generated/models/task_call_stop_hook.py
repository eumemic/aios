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

T = TypeVar("T", bound="TaskCallStopHook")


@_attrs_define
class TaskCallStopHook:
    """Stop hook: allow stop only when the agent calls the ``task_complete`` tool.

    The harness injects ``task_complete`` into the tool list and the
    system prompt for sessions with this hook.  Conversational
    end-of-turn (final text without tool calls) is treated as
    continuation; the agent must explicitly call ``task_complete()``.

    ``tool_name`` is fixed to ``task_complete`` in v1; the field exists
    so a future v2 can parameterize the terminator without a schema
    break.

        Attributes:
            type_ (Literal['task_call'] | Unset):  Default: 'task_call'.
            tool_name (Literal['task_complete'] | Unset):  Default: 'task_complete'.
            continuation_message (None | str | Unset):
    """

    type_: Literal["task_call"] | Unset = "task_call"
    tool_name: Literal["task_complete"] | Unset = "task_complete"
    continuation_message: None | str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        type_ = self.type_

        tool_name = self.tool_name

        continuation_message: None | str | Unset
        if isinstance(self.continuation_message, Unset):
            continuation_message = UNSET
        else:
            continuation_message = self.continuation_message

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if type_ is not UNSET:
            field_dict["type"] = type_
        if tool_name is not UNSET:
            field_dict["tool_name"] = tool_name
        if continuation_message is not UNSET:
            field_dict["continuation_message"] = continuation_message

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        type_ = cast(Literal["task_call"] | Unset, d.pop("type", UNSET))
        if type_ != "task_call" and not isinstance(type_, Unset):
            raise ValueError(f"type must match const 'task_call', got '{type_}'")

        tool_name = cast(Literal["task_complete"] | Unset, d.pop("tool_name", UNSET))
        if tool_name != "task_complete" and not isinstance(tool_name, Unset):
            raise ValueError(
                f"tool_name must match const 'task_complete', got '{tool_name}'"
            )

        def _parse_continuation_message(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        continuation_message = _parse_continuation_message(
            d.pop("continuation_message", UNSET)
        )

        task_call_stop_hook = cls(
            type_=type_,
            tool_name=tool_name,
            continuation_message=continuation_message,
        )

        return task_call_stop_hook
