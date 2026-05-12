from __future__ import annotations

from collections.abc import Mapping
from typing import (
    Any,
    Literal,
    TypeVar,
    cast,
)

from attrs import define as _attrs_define

from ..types import UNSET, Unset

T = TypeVar("T", bound="McpServerSpec")


@_attrs_define
class McpServerSpec:
    """One entry in an agent's ``mcp_servers`` list.

    Declares a remote MCP server reachable via streamable HTTP transport.
    The ``name`` is used to cross-reference from ``mcp_toolset`` tool entries
    and to namespace discovered tools as ``mcp__<name>__<tool_name>``.

    ``include_instructions`` controls whether the server's
    ``InitializeResult.instructions`` (per MCP spec) is rendered into the
    system prompt.  Defaults true so connector-mounted servers — and any
    third-party server that ships useful affordance prose — light up
    automatically.  Set false to opt out per agent (unfamiliar prose,
    noisy servers).

        Attributes:
            name (str):
            url (str):
            type_ (Literal['url'] | Unset):  Default: 'url'.
            include_instructions (bool | Unset):  Default: True.
    """

    name: str
    url: str
    type_: Literal["url"] | Unset = "url"
    include_instructions: bool | Unset = True

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        url = self.url

        type_ = self.type_

        include_instructions = self.include_instructions

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "name": name,
                "url": url,
            }
        )
        if type_ is not UNSET:
            field_dict["type"] = type_
        if include_instructions is not UNSET:
            field_dict["include_instructions"] = include_instructions

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        name = d.pop("name")

        url = d.pop("url")

        type_ = cast(Literal["url"] | Unset, d.pop("type", UNSET))
        if type_ != "url" and not isinstance(type_, Unset):
            raise ValueError(f"type must match const 'url', got '{type_}'")

        include_instructions = d.pop("include_instructions", UNSET)

        mcp_server_spec = cls(
            name=name,
            url=url,
            type_=type_,
            include_instructions=include_instructions,
        )

        return mcp_server_spec
