from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..models.tool_confirmation_request_result import ToolConfirmationRequestResult
from ..types import UNSET, Unset

T = TypeVar("T", bound="ToolConfirmationRequest")


@_attrs_define
class ToolConfirmationRequest:
    """Request body for ``POST /v1/sessions/{id}/tool-confirmations``.

    Used for built-in tools with ``permission: "always_ask"``. The client
    inspects the pending tool call and either allows it (the worker will
    execute it) or denies it (the model receives an error with the deny
    message).

        Attributes:
            tool_call_id (str): The tool_call_id to confirm or deny.
            result (ToolConfirmationRequestResult):
            deny_message (None | str | Unset): When result='deny', an optional message explaining why. Shown to the model.
    """

    tool_call_id: str
    result: ToolConfirmationRequestResult
    deny_message: None | str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        tool_call_id = self.tool_call_id

        result = self.result.value

        deny_message: None | str | Unset
        if isinstance(self.deny_message, Unset):
            deny_message = UNSET
        else:
            deny_message = self.deny_message

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "tool_call_id": tool_call_id,
                "result": result,
            }
        )
        if deny_message is not UNSET:
            field_dict["deny_message"] = deny_message

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        tool_call_id = d.pop("tool_call_id")

        result = ToolConfirmationRequestResult(d.pop("result"))

        def _parse_deny_message(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        deny_message = _parse_deny_message(d.pop("deny_message", UNSET))

        tool_confirmation_request = cls(
            tool_call_id=tool_call_id,
            result=result,
            deny_message=deny_message,
        )

        return tool_confirmation_request
