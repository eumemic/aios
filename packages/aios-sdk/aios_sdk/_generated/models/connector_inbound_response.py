from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="ConnectorInboundResponse")


@_attrs_define
class ConnectorInboundResponse:
    """Response for ``POST /v1/connectors/inbound``.

    Attributes:
        appended_event_id (None | str):
        session_id (None | str):
        deduped (bool):
    """

    appended_event_id: None | str
    session_id: None | str
    deduped: bool
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        appended_event_id: None | str
        appended_event_id = self.appended_event_id

        session_id: None | str
        session_id = self.session_id

        deduped = self.deduped

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "appended_event_id": appended_event_id,
                "session_id": session_id,
                "deduped": deduped,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)

        def _parse_appended_event_id(data: object) -> None | str:
            if data is None:
                return data
            return cast(None | str, data)

        appended_event_id = _parse_appended_event_id(d.pop("appended_event_id"))

        def _parse_session_id(data: object) -> None | str:
            if data is None:
                return data
            return cast(None | str, data)

        session_id = _parse_session_id(d.pop("session_id"))

        deduped = d.pop("deduped")

        connector_inbound_response = cls(
            appended_event_id=appended_event_id,
            session_id=session_id,
            deduped=deduped,
        )

        connector_inbound_response.additional_properties = d
        return connector_inbound_response

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
