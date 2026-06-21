from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="SessionAwaitResponse")


@_attrs_define
class SessionAwaitResponse:
    """Response for GET /v1/sessions/{id}/await — the session **quiescence
    drive-and-join** alias. Poll until `done` (`last_reacted_seq >= watermark`).
    Request correlation is the unified awaiter's job (`AwaitResponse`).

        Attributes:
            done (bool):
            last_reacted_seq (int):
    """

    done: bool
    last_reacted_seq: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        done = self.done

        last_reacted_seq = self.last_reacted_seq

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "done": done,
                "last_reacted_seq": last_reacted_seq,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        done = d.pop("done")

        last_reacted_seq = d.pop("last_reacted_seq")

        session_await_response = cls(
            done=done,
            last_reacted_seq=last_reacted_seq,
        )

        session_await_response.additional_properties = d
        return session_await_response

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
