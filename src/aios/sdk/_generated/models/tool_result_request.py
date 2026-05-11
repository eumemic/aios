from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.tool_result_request_content_type_1_item import (
        ToolResultRequestContentType1Item,
    )


T = TypeVar("T", bound="ToolResultRequest")


@_attrs_define
class ToolResultRequest:
    """Request body for ``POST /v1/sessions/{id}/tool-results``.

    ``content`` accepts either a plain string OR a multimodal content
    array shaped per the OpenAI chat-completions tool-result format
    (e.g. ``[{"type": "text", "text": "..."}, {"type": "image_url",
    "image_url": {"url": "..."}}]``).  Built-in tools have always
    produced multimodal results; this widening lets external clients
    (#301 — connectors as HTTP clients) do the same when posting
    custom-tool results.

        Attributes:
            tool_call_id (str): The tool_call_id from the assistant's tool_calls.
            content (list[ToolResultRequestContentType1Item] | str): The result of executing the tool.  String or multimodal
                content array.
            is_error (bool | Unset): True if the tool execution failed. Default: False.
    """

    tool_call_id: str
    content: list[ToolResultRequestContentType1Item] | str
    is_error: bool | Unset = False

    def to_dict(self) -> dict[str, Any]:
        tool_call_id = self.tool_call_id

        content: list[dict[str, Any]] | str
        if isinstance(self.content, list):
            content = []
            for content_type_1_item_data in self.content:
                content_type_1_item = content_type_1_item_data.to_dict()
                content.append(content_type_1_item)

        else:
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
        from ..models.tool_result_request_content_type_1_item import (
            ToolResultRequestContentType1Item,
        )

        d = dict(src_dict)
        tool_call_id = d.pop("tool_call_id")

        def _parse_content(
            data: object,
        ) -> list[ToolResultRequestContentType1Item] | str:
            try:
                if not isinstance(data, list):
                    raise TypeError()
                content_type_1 = []
                _content_type_1 = data
                for content_type_1_item_data in _content_type_1:
                    content_type_1_item = ToolResultRequestContentType1Item.from_dict(
                        content_type_1_item_data
                    )

                    content_type_1.append(content_type_1_item)

                return content_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[ToolResultRequestContentType1Item] | str, data)

        content = _parse_content(d.pop("content"))

        is_error = d.pop("is_error", UNSET)

        tool_result_request = cls(
            tool_call_id=tool_call_id,
            content=content,
            is_error=is_error,
        )

        return tool_result_request
