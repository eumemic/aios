from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.task_handle_servicer_kind import TaskHandleServicerKind

T = TypeVar("T", bound="TaskHandle")


@_attrs_define
class TaskHandle:
    """Structured handle returned by ``POST /v1/tasks``.

    Plain JSON fields, no opaque encoding: the handle is **not** an auth boundary
    (``await`` re-authorizes by ``account_id``). Await the task at the one
    unified awaiter ``GET /v1/tasks/{servicer_id}/await`` — the ``task_id``
    path segment is the ``servicer_id`` and its kind is read off the id prefix; a
    ``session`` servicer additionally needs ``?request_id=`` to correlate the
    response, a ``run`` servicer resolves off its terminal row.

        Attributes:
            servicer_kind (TaskHandleServicerKind):
            servicer_id (str):
            request_id (str):
    """

    servicer_kind: TaskHandleServicerKind
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
        servicer_kind = TaskHandleServicerKind(d.pop("servicer_kind"))

        servicer_id = d.pop("servicer_id")

        request_id = d.pop("request_id")

        task_handle = cls(
            servicer_kind=servicer_kind,
            servicer_id=servicer_id,
            request_id=request_id,
        )

        task_handle.additional_properties = d
        return task_handle

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
