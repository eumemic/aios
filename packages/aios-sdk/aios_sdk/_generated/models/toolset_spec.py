from __future__ import annotations

from collections.abc import Mapping
from typing import (
    Any,
    Literal,
    TypeVar,
    cast,
)

from attrs import define as _attrs_define

from ..models.toolset_spec_name import ToolsetSpecName

T = TypeVar("T", bound="ToolsetSpec")


@_attrs_define
class ToolsetSpec:
    """Ingress-only reference to an immutable, curated capability set.

    Toolsets are expanded while validating an agent create/update request. They
    are never persisted or returned: the resolved agent surface contains only
    ordinary :class:`ToolSpec` entries, making every granted capability explicit.

        Attributes:
            type_ (Literal['toolset']):
            name (ToolsetSpecName):
            version (Literal[1]):
    """

    type_: Literal["toolset"]
    name: ToolsetSpecName
    version: Literal[1]

    def to_dict(self) -> dict[str, Any]:
        type_ = self.type_

        name = self.name.value

        version = self.version

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "type": type_,
                "name": name,
                "version": version,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        type_ = cast(Literal["toolset"], d.pop("type"))
        if type_ != "toolset":
            raise ValueError(f"type must match const 'toolset', got '{type_}'")

        name = ToolsetSpecName(d.pop("name"))

        version = cast(Literal[1], d.pop("version"))
        if version != 1:
            raise ValueError(f"version must match const 1, got '{version}'")

        toolset_spec = cls(
            type_=type_,
            name=name,
            version=version,
        )

        return toolset_spec
