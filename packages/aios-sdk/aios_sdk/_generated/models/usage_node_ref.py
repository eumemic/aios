from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.usage_node_ref_kind import UsageNodeRefKind

T = TypeVar("T", bound="UsageNodeRef")


@_attrs_define
class UsageNodeRef:
    """One immutable parent in the creation-accounting tree.

    Attributes:
        kind (UsageNodeRefKind):
        id (str):
    """

    kind: UsageNodeRefKind
    id: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        kind = self.kind.value

        id = self.id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "kind": kind,
                "id": id,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        kind = UsageNodeRefKind(d.pop("kind"))

        id = d.pop("id")

        usage_node_ref = cls(
            kind=kind,
            id=id,
        )

        usage_node_ref.additional_properties = d
        return usage_node_ref

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
