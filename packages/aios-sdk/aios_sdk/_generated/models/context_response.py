from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.context_response_messages_item import ContextResponseMessagesItem
    from ..models.context_response_tools_item import ContextResponseToolsItem


T = TypeVar("T", bound="ContextResponse")


@_attrs_define
class ContextResponse:
    """Response for ``GET /v1/sessions/{id}/context``.

    The exact chat-completions payload the worker would send to LiteLLM
    if a step ran right now.  Dry-run: no side effects — no events
    appended, no status bump, no skill files provisioned.

        Attributes:
            session_id (str):
            model (str):
            messages (list[ContextResponseMessagesItem]):
            tools (list[ContextResponseToolsItem]):
    """

    session_id: str
    model: str
    messages: list[ContextResponseMessagesItem]
    tools: list[ContextResponseToolsItem]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        session_id = self.session_id

        model = self.model

        messages = []
        for messages_item_data in self.messages:
            messages_item = messages_item_data.to_dict()
            messages.append(messages_item)

        tools = []
        for tools_item_data in self.tools:
            tools_item = tools_item_data.to_dict()
            tools.append(tools_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "session_id": session_id,
                "model": model,
                "messages": messages,
                "tools": tools,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.context_response_messages_item import ContextResponseMessagesItem
        from ..models.context_response_tools_item import ContextResponseToolsItem

        d = dict(src_dict)
        session_id = d.pop("session_id")

        model = d.pop("model")

        messages = []
        _messages = d.pop("messages")
        for messages_item_data in _messages:
            messages_item = ContextResponseMessagesItem.from_dict(messages_item_data)

            messages.append(messages_item)

        tools = []
        _tools = d.pop("tools")
        for tools_item_data in _tools:
            tools_item = ContextResponseToolsItem.from_dict(tools_item_data)

            tools.append(tools_item)

        context_response = cls(
            session_id=session_id,
            model=model,
            messages=messages,
            tools=tools,
        )

        context_response.additional_properties = d
        return context_response

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
