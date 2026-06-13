from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.awaiting_tool_call_kind import AwaitingToolCallKind

T = TypeVar("T", bound="AwaitingToolCall")


@_attrs_define
class AwaitingToolCall:
    """One pending tool call the harness will not dispatch itself.

    Derived view on session reads. Each entry is a tool_call in the
    latest assistant message with no paired tool_result and no
    in-process executor:

    * ``kind == "custom"`` — client-executed; awaits POST to
      ``/sessions/:id/tool-results`` (operator-facing) or
      ``/connectors/runtime/tool-results`` (runtime-container-facing).
    * ``kind == "builtin" | "mcp"`` — ``always_ask``-gated and not yet
      confirmed; awaits POST to ``/sessions/:id/tool-confirmations``.
      Confirmed-but-not-yet-dispatched and ``always_allow`` calls don't
      appear here — they're harness-internal.

    ``pending_since`` is the ``created_at`` of the assistant event that
    declared the call — the moment the call became pending. Consumers use
    it to distinguish a healthy in-flight custom call (fresh) from a stuck
    one whose client died (stale), without loading the transcript. For a
    tool_call spanning multiple unresolved turns it is the declaring
    assistant's timestamp, not "now".

        Attributes:
            tool_call_id (str):
            name (str):
            kind (AwaitingToolCallKind):
            pending_since (datetime.datetime):
    """

    tool_call_id: str
    name: str
    kind: AwaitingToolCallKind
    pending_since: datetime.datetime
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        tool_call_id = self.tool_call_id

        name = self.name

        kind = self.kind.value

        pending_since = self.pending_since.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "tool_call_id": tool_call_id,
                "name": name,
                "kind": kind,
                "pending_since": pending_since,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        tool_call_id = d.pop("tool_call_id")

        name = d.pop("name")

        kind = AwaitingToolCallKind(d.pop("kind"))

        pending_since = isoparse(d.pop("pending_since"))

        awaiting_tool_call = cls(
            tool_call_id=tool_call_id,
            name=name,
            kind=kind,
            pending_since=pending_since,
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
