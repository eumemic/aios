from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.tool_spec import ToolSpec


T = TypeVar("T", bound="ConnectionSetTools")


@_attrs_define
class ConnectionSetTools:
    """Request body for ``PUT /v1/connections/{id}/tools`` (#301).

    Replaces the connection's tools array wholesale.  Each entry must
    be ``type="custom"`` — see :func:`_validate_connection_tools`.

        Attributes:
            tools (list[ToolSpec] | Unset):
    """

    tools: list[ToolSpec] | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        tools: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.tools, Unset):
            tools = []
            for tools_item_data in self.tools:
                tools_item = tools_item_data.to_dict()
                tools.append(tools_item)

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if tools is not UNSET:
            field_dict["tools"] = tools

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.tool_spec import ToolSpec

        d = dict(src_dict)
        _tools = d.pop("tools", UNSET)
        tools: list[ToolSpec] | Unset = UNSET
        if _tools is not UNSET:
            tools = []
            for tools_item_data in _tools:
                tools_item = ToolSpec.from_dict(tools_item_data)

                tools.append(tools_item)

        connection_set_tools = cls(
            tools=tools,
        )

        return connection_set_tools
