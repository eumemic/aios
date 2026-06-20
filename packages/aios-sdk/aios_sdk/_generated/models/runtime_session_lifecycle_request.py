from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.runtime_session_lifecycle_request_data_type_0 import (
        RuntimeSessionLifecycleRequestDataType0,
    )


T = TypeVar("T", bound="RuntimeSessionLifecycleRequest")


@_attrs_define
class RuntimeSessionLifecycleRequest:
    """Body for ``POST /v1/connectors/runtime/session-lifecycle`` (#1261).

    The per-session-targeted sibling of :class:`RuntimeLifecycleRequest`.
    Where the broadcast ``/runtime/lifecycle`` route fans a transport-down
    notice across *every* session bound to the connection, this appends a
    single ``kind=lifecycle`` event onto **one** named session — the gap
    called out by the SMS design (§3.5 req 1): a delivery failure must reach
    the *originating* session, not be broadcast.

    ``wake`` optionally pairs the append with a ``defer_wake`` so the failure
    isn't merely visible-on-next-turn but actually wakes the session (the
    "give it stimulus" half of the design's option (a)). Defaults ``False``
    so the primitive stays a plain visible-on-next-wake append unless the
    caller opts into the wake.

        Attributes:
            connection_id (str):
            session_id (str):
            event (str):
            reason (None | str | Unset):
            data (None | RuntimeSessionLifecycleRequestDataType0 | Unset):
            wake (bool | Unset):  Default: False.
    """

    connection_id: str
    session_id: str
    event: str
    reason: None | str | Unset = UNSET
    data: None | RuntimeSessionLifecycleRequestDataType0 | Unset = UNSET
    wake: bool | Unset = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.runtime_session_lifecycle_request_data_type_0 import (
            RuntimeSessionLifecycleRequestDataType0,
        )

        connection_id = self.connection_id

        session_id = self.session_id

        event = self.event

        reason: None | str | Unset
        if isinstance(self.reason, Unset):
            reason = UNSET
        else:
            reason = self.reason

        data: dict[str, Any] | None | Unset
        if isinstance(self.data, Unset):
            data = UNSET
        elif isinstance(self.data, RuntimeSessionLifecycleRequestDataType0):
            data = self.data.to_dict()
        else:
            data = self.data

        wake = self.wake

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "connection_id": connection_id,
                "session_id": session_id,
                "event": event,
            }
        )
        if reason is not UNSET:
            field_dict["reason"] = reason
        if data is not UNSET:
            field_dict["data"] = data
        if wake is not UNSET:
            field_dict["wake"] = wake

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.runtime_session_lifecycle_request_data_type_0 import (
            RuntimeSessionLifecycleRequestDataType0,
        )

        d = dict(src_dict)
        connection_id = d.pop("connection_id")

        session_id = d.pop("session_id")

        event = d.pop("event")

        def _parse_reason(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        reason = _parse_reason(d.pop("reason", UNSET))

        def _parse_data(
            data: object,
        ) -> None | RuntimeSessionLifecycleRequestDataType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                data_type_0 = RuntimeSessionLifecycleRequestDataType0.from_dict(data)

                return data_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | RuntimeSessionLifecycleRequestDataType0 | Unset, data)

        data = _parse_data(d.pop("data", UNSET))

        wake = d.pop("wake", UNSET)

        runtime_session_lifecycle_request = cls(
            connection_id=connection_id,
            session_id=session_id,
            event=event,
            reason=reason,
            data=data,
            wake=wake,
        )

        runtime_session_lifecycle_request.additional_properties = d
        return runtime_session_lifecycle_request

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
