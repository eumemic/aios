from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="OwedRequest")


@_attrs_define
class OwedRequest:
    """Read-model projection of an open obligation on the :class:`Session` view (#1413).

    Parallel to ``Session.awaiting`` (tool-call-only), but for the request edges
    the session owes a response to. Projected from the same ``get_open_obligations``
    rows. Load-bearing for the operator-cancel path (#1414) — it cannot enumerate
    a session's open goal_ids otherwise.

        Attributes:
            request_id (str):
            caller_kind (str):
            opened_at (datetime.datetime):
            caller_id (None | str | Unset):
            summary (None | str | Unset):
    """

    request_id: str
    caller_kind: str
    opened_at: datetime.datetime
    caller_id: None | str | Unset = UNSET
    summary: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        request_id = self.request_id

        caller_kind = self.caller_kind

        opened_at = self.opened_at.isoformat()

        caller_id: None | str | Unset
        if isinstance(self.caller_id, Unset):
            caller_id = UNSET
        else:
            caller_id = self.caller_id

        summary: None | str | Unset
        if isinstance(self.summary, Unset):
            summary = UNSET
        else:
            summary = self.summary

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "request_id": request_id,
                "caller_kind": caller_kind,
                "opened_at": opened_at,
            }
        )
        if caller_id is not UNSET:
            field_dict["caller_id"] = caller_id
        if summary is not UNSET:
            field_dict["summary"] = summary

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        request_id = d.pop("request_id")

        caller_kind = d.pop("caller_kind")

        opened_at = isoparse(d.pop("opened_at"))

        def _parse_caller_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        caller_id = _parse_caller_id(d.pop("caller_id", UNSET))

        def _parse_summary(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        summary = _parse_summary(d.pop("summary", UNSET))

        owed_request = cls(
            request_id=request_id,
            caller_kind=caller_kind,
            opened_at=opened_at,
            caller_id=caller_id,
            summary=summary,
        )

        owed_request.additional_properties = d
        return owed_request

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
