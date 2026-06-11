from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.wf_run_event_type import WfRunEventType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.wf_run_event_payload import WfRunEventPayload


T = TypeVar("T", bound="WfRunEvent")


@_attrs_define
class WfRunEvent:
    """One row of a run's append-only journal (the replay-with-memo source).

    ``call_key`` is set for ``call_started``/``call_result`` (the memo key) and for
    ``annotation`` (the branch-local dedup key that makes ``log()``/``phase()``
    emit-once across replays); it is ``None`` for the ``run_started``/``run_completed``
    bookends. An ``annotation`` is a journaled progress marker (``payload`` =
    ``{"kind": "log" | "phase", "text": ...}``), not a capability call.

        Attributes:
            id (str):
            run_id (str):
            seq (int):
            type_ (WfRunEventType):
            payload (WfRunEventPayload):
            created_at (datetime.datetime):
            call_key (None | str | Unset):
    """

    id: str
    run_id: str
    seq: int
    type_: WfRunEventType
    payload: WfRunEventPayload
    created_at: datetime.datetime
    call_key: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        run_id = self.run_id

        seq = self.seq

        type_ = self.type_.value

        payload = self.payload.to_dict()

        created_at = self.created_at.isoformat()

        call_key: None | str | Unset
        if isinstance(self.call_key, Unset):
            call_key = UNSET
        else:
            call_key = self.call_key

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "run_id": run_id,
                "seq": seq,
                "type": type_,
                "payload": payload,
                "created_at": created_at,
            }
        )
        if call_key is not UNSET:
            field_dict["call_key"] = call_key

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.wf_run_event_payload import WfRunEventPayload

        d = dict(src_dict)
        id = d.pop("id")

        run_id = d.pop("run_id")

        seq = d.pop("seq")

        type_ = WfRunEventType(d.pop("type"))

        payload = WfRunEventPayload.from_dict(d.pop("payload"))

        created_at = isoparse(d.pop("created_at"))

        def _parse_call_key(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        call_key = _parse_call_key(d.pop("call_key", UNSET))

        wf_run_event = cls(
            id=id,
            run_id=run_id,
            seq=seq,
            type_=type_,
            payload=payload,
            created_at=created_at,
            call_key=call_key,
        )

        wf_run_event.additional_properties = d
        return wf_run_event

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
