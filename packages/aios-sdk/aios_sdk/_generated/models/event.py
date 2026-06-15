from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.event_kind import EventKind
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.event_data import EventData


T = TypeVar("T", bound="Event")


@_attrs_define
class Event:
    """Read view of a single event from the session log.

    Schema (#1140): a session event is ``{kind, data, seq}`` — a DIFFERENT
    shape from a *run* event (``{type, payload, seq}``, see
    ``aios.models.workflows.WfRunEvent``). See module docstring and
    ``docs/reference/run-observability.md`` for the split.

        Attributes:
            id (str):
            session_id (str):
            seq (int):
            kind (EventKind):
            data (EventData):
            created_at (datetime.datetime):
            cumulative_tokens (int | None | Unset):
            orig_channel (None | str | Unset):
            focal_channel_at_arrival (None | str | Unset):
            channel (None | str | Unset):
    """

    id: str
    session_id: str
    seq: int
    kind: EventKind
    data: EventData
    created_at: datetime.datetime
    cumulative_tokens: int | None | Unset = UNSET
    orig_channel: None | str | Unset = UNSET
    focal_channel_at_arrival: None | str | Unset = UNSET
    channel: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        session_id = self.session_id

        seq = self.seq

        kind = self.kind.value

        data = self.data.to_dict()

        created_at = self.created_at.isoformat()

        cumulative_tokens: int | None | Unset
        if isinstance(self.cumulative_tokens, Unset):
            cumulative_tokens = UNSET
        else:
            cumulative_tokens = self.cumulative_tokens

        orig_channel: None | str | Unset
        if isinstance(self.orig_channel, Unset):
            orig_channel = UNSET
        else:
            orig_channel = self.orig_channel

        focal_channel_at_arrival: None | str | Unset
        if isinstance(self.focal_channel_at_arrival, Unset):
            focal_channel_at_arrival = UNSET
        else:
            focal_channel_at_arrival = self.focal_channel_at_arrival

        channel: None | str | Unset
        if isinstance(self.channel, Unset):
            channel = UNSET
        else:
            channel = self.channel

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "session_id": session_id,
                "seq": seq,
                "kind": kind,
                "data": data,
                "created_at": created_at,
            }
        )
        if cumulative_tokens is not UNSET:
            field_dict["cumulative_tokens"] = cumulative_tokens
        if orig_channel is not UNSET:
            field_dict["orig_channel"] = orig_channel
        if focal_channel_at_arrival is not UNSET:
            field_dict["focal_channel_at_arrival"] = focal_channel_at_arrival
        if channel is not UNSET:
            field_dict["channel"] = channel

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.event_data import EventData

        d = dict(src_dict)
        id = d.pop("id")

        session_id = d.pop("session_id")

        seq = d.pop("seq")

        kind = EventKind(d.pop("kind"))

        data = EventData.from_dict(d.pop("data"))

        created_at = isoparse(d.pop("created_at"))

        def _parse_cumulative_tokens(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        cumulative_tokens = _parse_cumulative_tokens(d.pop("cumulative_tokens", UNSET))

        def _parse_orig_channel(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        orig_channel = _parse_orig_channel(d.pop("orig_channel", UNSET))

        def _parse_focal_channel_at_arrival(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        focal_channel_at_arrival = _parse_focal_channel_at_arrival(
            d.pop("focal_channel_at_arrival", UNSET)
        )

        def _parse_channel(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        channel = _parse_channel(d.pop("channel", UNSET))

        event = cls(
            id=id,
            session_id=session_id,
            seq=seq,
            kind=kind,
            data=data,
            created_at=created_at,
            cumulative_tokens=cumulative_tokens,
            orig_channel=orig_channel,
            focal_channel_at_arrival=focal_channel_at_arrival,
            channel=channel,
        )

        event.additional_properties = d
        return event

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
