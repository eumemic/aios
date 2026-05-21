from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.wait_response_session_status import WaitResponseSessionStatus
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.awaiting_tool_call import AwaitingToolCall
    from ..models.event import Event
    from ..models.wait_response_session_stop_reason_type_0 import (
        WaitResponseSessionStopReasonType0,
    )


T = TypeVar("T", bound="WaitResponse")


@_attrs_define
class WaitResponse:
    """Response for ``GET /v1/sessions/{id}/wait``.

    Attributes:
        events (list[Event]):
        session_status (WaitResponseSessionStatus):
        session_stop_reason (None | WaitResponseSessionStopReasonType0):
        next_after (int):
        session_awaiting (list[AwaitingToolCall] | Unset):
    """

    events: list[Event]
    session_status: WaitResponseSessionStatus
    session_stop_reason: None | WaitResponseSessionStopReasonType0
    next_after: int
    session_awaiting: list[AwaitingToolCall] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.wait_response_session_stop_reason_type_0 import (
            WaitResponseSessionStopReasonType0,
        )

        events = []
        for events_item_data in self.events:
            events_item = events_item_data.to_dict()
            events.append(events_item)

        session_status = self.session_status.value

        session_stop_reason: dict[str, Any] | None
        if isinstance(self.session_stop_reason, WaitResponseSessionStopReasonType0):
            session_stop_reason = self.session_stop_reason.to_dict()
        else:
            session_stop_reason = self.session_stop_reason

        next_after = self.next_after

        session_awaiting: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.session_awaiting, Unset):
            session_awaiting = []
            for session_awaiting_item_data in self.session_awaiting:
                session_awaiting_item = session_awaiting_item_data.to_dict()
                session_awaiting.append(session_awaiting_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "events": events,
                "session_status": session_status,
                "session_stop_reason": session_stop_reason,
                "next_after": next_after,
            }
        )
        if session_awaiting is not UNSET:
            field_dict["session_awaiting"] = session_awaiting

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.awaiting_tool_call import AwaitingToolCall
        from ..models.event import Event
        from ..models.wait_response_session_stop_reason_type_0 import (
            WaitResponseSessionStopReasonType0,
        )

        d = dict(src_dict)
        events = []
        _events = d.pop("events")
        for events_item_data in _events:
            events_item = Event.from_dict(events_item_data)

            events.append(events_item)

        session_status = WaitResponseSessionStatus(d.pop("session_status"))

        def _parse_session_stop_reason(
            data: object,
        ) -> None | WaitResponseSessionStopReasonType0:
            if data is None:
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                session_stop_reason_type_0 = (
                    WaitResponseSessionStopReasonType0.from_dict(data)
                )

                return session_stop_reason_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | WaitResponseSessionStopReasonType0, data)

        session_stop_reason = _parse_session_stop_reason(d.pop("session_stop_reason"))

        next_after = d.pop("next_after")

        _session_awaiting = d.pop("session_awaiting", UNSET)
        session_awaiting: list[AwaitingToolCall] | Unset = UNSET
        if _session_awaiting is not UNSET:
            session_awaiting = []
            for session_awaiting_item_data in _session_awaiting:
                session_awaiting_item = AwaitingToolCall.from_dict(
                    session_awaiting_item_data
                )

                session_awaiting.append(session_awaiting_item)

        wait_response = cls(
            events=events,
            session_status=session_status,
            session_stop_reason=session_stop_reason,
            next_after=next_after,
            session_awaiting=session_awaiting,
        )

        wait_response.additional_properties = d
        return wait_response

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
