from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="ToolResultRequest")


@_attrs_define
class ToolResultRequest:
    """Request body for ``POST /v1/sessions/{id}/tool-results``.

    Attributes:
        tool_call_id (str): The tool_call_id from the assistant's tool_calls.
        content (str): The result of executing the tool.
        is_error (bool | Unset): True if the tool execution failed. Default: False.
    """

    tool_call_id: str
    content: str
    is_error: bool | Unset = False

    def to_dict(self) -> dict[str, Any]:
        tool_call_id = self.tool_call_id

        content = self.content

        is_error = self.is_error

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "tool_call_id": tool_call_id,
                "content": content,
            }
        )
        if is_error is not UNSET:
            field_dict["is_error"] = is_error

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        tool_call_id = d.pop("tool_call_id")

        content = d.pop("content")

        is_error = d.pop("is_error", UNSET)

        tool_result_request = cls(
            tool_call_id=tool_call_id,
            content=content,
            is_error=is_error,
        )

        return tool_result_request
