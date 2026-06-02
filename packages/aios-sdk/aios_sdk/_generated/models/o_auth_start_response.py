from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="OAuthStartResponse")


@_attrs_define
class OAuthStartResponse:
    """The authorization URL to redirect the user to, plus the flow's CSRF state.

    Attributes:
        flow_id (str):
        state (str):
        authorization_url (str):
    """

    flow_id: str
    state: str
    authorization_url: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        flow_id = self.flow_id

        state = self.state

        authorization_url = self.authorization_url

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "flow_id": flow_id,
                "state": state,
                "authorization_url": authorization_url,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        flow_id = d.pop("flow_id")

        state = d.pop("state")

        authorization_url = d.pop("authorization_url")

        o_auth_start_response = cls(
            flow_id=flow_id,
            state=state,
            authorization_url=authorization_url,
        )

        o_auth_start_response.additional_properties = d
        return o_auth_start_response

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
