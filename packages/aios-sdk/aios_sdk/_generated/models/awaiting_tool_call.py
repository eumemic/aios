from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.awaiting_tool_call_kind import AwaitingToolCallKind

T = TypeVar("T", bound="AwaitingToolCall")


@_attrs_define
class AwaitingToolCall:
    """One pending tool call the harness will not dispatch itself.

    Derived view on session reads: each entry is a tool_call in the
    latest assistant message that has no paired tool_result event and
    no in-process executor. Two flavors share this state:

    * ``kind == "custom"`` — client-executed; awaits a POST to
      ``/sessions/:id/tool-results`` (operator-facing) or
      ``/connectors/runtime/tool-results`` (runtime-container-facing).
    * ``kind == "builtin" | "mcp"`` with ``needs_confirm=True`` —
      ``always_ask``-gated; awaits a POST to ``/sessions/:id/tool-confirmations``.

        Attributes:
            tool_call_id (str):
            name (str):
            kind (AwaitingToolCallKind):
            needs_confirm (bool):
    """

    tool_call_id: str
    name: str
    kind: AwaitingToolCallKind
    needs_confirm: bool
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        tool_call_id = self.tool_call_id

        name = self.name

        kind = self.kind.value

        needs_confirm = self.needs_confirm

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "tool_call_id": tool_call_id,
                "name": name,
                "kind": kind,
                "needs_confirm": needs_confirm,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        tool_call_id = d.pop("tool_call_id")

        name = d.pop("name")

        kind = AwaitingToolCallKind(d.pop("kind"))

        needs_confirm = d.pop("needs_confirm")

        awaiting_tool_call = cls(
            tool_call_id=tool_call_id,
            name=name,
            kind=kind,
            needs_confirm=needs_confirm,
        )

        awaiting_tool_call.additional_properties = d
        return awaiting_tool_call

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
