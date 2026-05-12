from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define

if TYPE_CHECKING:
    from ..models.tools_schema_update_tools_item import ToolsSchemaUpdateToolsItem


T = TypeVar("T", bound="ToolsSchemaUpdate")


@_attrs_define
class ToolsSchemaUpdate:
    """Body for ``PUT /v1/connectors/{connector}/tools_schema``.

    Attributes:
        tools (list[ToolsSchemaUpdateToolsItem]):
    """

    tools: list[ToolsSchemaUpdateToolsItem]

    def to_dict(self) -> dict[str, Any]:
        tools = []
        for tools_item_data in self.tools:
            tools_item = tools_item_data.to_dict()
            tools.append(tools_item)

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "tools": tools,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.tools_schema_update_tools_item import ToolsSchemaUpdateToolsItem

        d = dict(src_dict)
        tools = []
        _tools = d.pop("tools")
        for tools_item_data in _tools:
            tools_item = ToolsSchemaUpdateToolsItem.from_dict(tools_item_data)

            tools.append(tools_item)

        tools_schema_update = cls(
            tools=tools,
        )

        return tools_schema_update
