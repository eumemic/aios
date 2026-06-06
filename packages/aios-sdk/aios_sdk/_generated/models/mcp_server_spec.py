from __future__ import annotations

from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
    cast,
)

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.mcp_server_spec_headers_type_0 import McpServerSpecHeadersType0


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

    ``headers`` are extra NON-SECRET HTTP headers sent on every request to
    this server — toolset selectors (e.g. GitHub's
    ``X-MCP-Toolsets: discussions,issues``), format hints, API-version
    pins.  Do NOT put secrets here: this dict is stored in plaintext agent
    JSON.  Real credentials belong in the vault path; a vault-derived auth
    header overrides a same-named entry here (auth headers win on
    collision).  Names must be valid HTTP tokens and values printable ASCII
    (validated below) so they can't fail only at connection time; headers
    the MCP transport authors itself (Accept, Content-Type, Mcp-Session-Id,
    Mcp-Protocol-Version) are rejected — setting them here is a silent no-op.

        Attributes:
            name (str):
            url (str):
            type_ (Literal['url'] | Unset):  Default: 'url'.
            include_instructions (bool | Unset):  Default: True.
            headers (McpServerSpecHeadersType0 | None | Unset):
    """

    name: str
    url: str
    type_: Literal["url"] | Unset = "url"
    include_instructions: bool | Unset = True
    headers: McpServerSpecHeadersType0 | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.mcp_server_spec_headers_type_0 import McpServerSpecHeadersType0

        name = self.name

        url = self.url

        type_ = self.type_

        include_instructions = self.include_instructions

        headers: dict[str, Any] | None | Unset
        if isinstance(self.headers, Unset):
            headers = UNSET
        elif isinstance(self.headers, McpServerSpecHeadersType0):
            headers = self.headers.to_dict()
        else:
            headers = self.headers

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
        if headers is not UNSET:
            field_dict["headers"] = headers

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.mcp_server_spec_headers_type_0 import McpServerSpecHeadersType0

        d = dict(src_dict)
        name = d.pop("name")

        url = d.pop("url")

        type_ = cast(Literal["url"] | Unset, d.pop("type", UNSET))
        if type_ != "url" and not isinstance(type_, Unset):
            raise ValueError(f"type must match const 'url', got '{type_}'")

        include_instructions = d.pop("include_instructions", UNSET)

        def _parse_headers(data: object) -> McpServerSpecHeadersType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                headers_type_0 = McpServerSpecHeadersType0.from_dict(data)

                return headers_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(McpServerSpecHeadersType0 | None | Unset, data)

        headers = _parse_headers(d.pop("headers", UNSET))

        mcp_server_spec = cls(
            name=name,
            url=url,
            type_=type_,
            include_instructions=include_instructions,
            headers=headers,
        )

        return mcp_server_spec
