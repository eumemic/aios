from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.input_event import InputEvent


T = TypeVar("T", bound="InputBatch")


@_attrs_define
class InputBatch:
    """One coalesced batch of input events, epoch-stamped.

    The API pre-checks the epoch against the grant record; the DRIVER is the
    enforcement authority and drops stale-epoch spool lines regardless.

        Attributes:
            epoch (int):
            seq (int):
            events (list[InputEvent]):
    """

    epoch: int
    seq: int
    events: list[InputEvent]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        epoch = self.epoch

        seq = self.seq

        events = []
        for events_item_data in self.events:
            events_item = events_item_data.to_dict()
            events.append(events_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "epoch": epoch,
                "seq": seq,
                "events": events,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.input_event import InputEvent

        d = dict(src_dict)
        epoch = d.pop("epoch")

        seq = d.pop("seq")

        events = []
        _events = d.pop("events")
        for events_item_data in _events:
            events_item = InputEvent.from_dict(events_item_data)

            events.append(events_item)

        input_batch = cls(
            epoch=epoch,
            seq=seq,
            events=events,
        )

        input_batch.additional_properties = d
        return input_batch

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
