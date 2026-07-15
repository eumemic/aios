from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.runtime_tool_result_request_content_type_1_item import (
        RuntimeToolResultRequestContentType1Item,
    )


T = TypeVar("T", bound="RuntimeToolResultRequest")


@_attrs_define
class RuntimeToolResultRequest:
    """Body for ``POST /v1/connectors/runtime/tool-results``.

    Carries ``connection_id`` explicitly — the bearer scopes the
    caller to a connector *type*, not to one connection, so the body
    has to name the target connection.

        Attributes:
            connection_id (str):
            session_id (str):
            tool_call_id (str):
            content (list[RuntimeToolResultRequestContentType1Item] | str):
            is_error (bool | Unset):  Default: False.
    """

    connection_id: str
    session_id: str
    tool_call_id: str
    content: list[RuntimeToolResultRequestContentType1Item] | str
    is_error: bool | Unset = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        connection_id = self.connection_id

        session_id = self.session_id

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
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "connection_id": connection_id,
                "session_id": session_id,
                "tool_call_id": tool_call_id,
                "content": content,
            }
        )
        if is_error is not UNSET:
            field_dict["is_error"] = is_error

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.runtime_tool_result_request_content_type_1_item import (
            RuntimeToolResultRequestContentType1Item,
        )

        d = dict(src_dict)
        connection_id = d.pop("connection_id")

        session_id = d.pop("session_id")

        tool_call_id = d.pop("tool_call_id")

        def _parse_content(
            data: object,
        ) -> list[RuntimeToolResultRequestContentType1Item] | str:
            try:
                if not isinstance(data, list):
                    raise TypeError()
                content_type_1 = []
                _content_type_1 = data
                for content_type_1_item_data in _content_type_1:
                    content_type_1_item = (
                        RuntimeToolResultRequestContentType1Item.from_dict(
                            content_type_1_item_data
                        )
                    )

                    content_type_1.append(content_type_1_item)

                return content_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[RuntimeToolResultRequestContentType1Item] | str, data)

        content = _parse_content(d.pop("content"))

        is_error = d.pop("is_error", UNSET)

        runtime_tool_result_request = cls(
            connection_id=connection_id,
            session_id=session_id,
            tool_call_id=tool_call_id,
            content=content,
            is_error=is_error,
        )

        runtime_tool_result_request.additional_properties = d
        return runtime_tool_result_request

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
