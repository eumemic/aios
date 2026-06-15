from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.invocation_handle_servicer_kind import InvocationHandleServicerKind

T = TypeVar("T", bound="InvocationHandle")


@_attrs_define
class InvocationHandle:
    """Structured handle returned by ``POST /v1/invocations``.

    Plain JSON fields, no opaque encoding: the handle is **not** an auth boundary
    (``await`` re-authorizes by ``account_id``). The caller picks the matching
    awaiter off ``servicer_kind`` — ``session`` → ``GET /sessions/{servicer_id}/await``
    with ``request_id``, ``run`` → ``GET /runs/{servicer_id}/wait``.

        Attributes:
            servicer_kind (InvocationHandleServicerKind):
            servicer_id (str):
            request_id (str):
    """

    servicer_kind: InvocationHandleServicerKind
    servicer_id: str
    request_id: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        servicer_kind = self.servicer_kind.value

        servicer_id = self.servicer_id

        request_id = self.request_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "servicer_kind": servicer_kind,
                "servicer_id": servicer_id,
                "request_id": request_id,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        servicer_kind = InvocationHandleServicerKind(d.pop("servicer_kind"))

        servicer_id = d.pop("servicer_id")

        request_id = d.pop("request_id")

        invocation_handle = cls(
            servicer_kind=servicer_kind,
            servicer_id=servicer_id,
            request_id=request_id,
        )

        invocation_handle.additional_properties = d
        return invocation_handle

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
