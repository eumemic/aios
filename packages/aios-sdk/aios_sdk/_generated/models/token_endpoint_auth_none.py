from __future__ import annotations

from collections.abc import Mapping
from typing import (
    Any,
    Literal,
    TypeVar,
    cast,
)

from attrs import define as _attrs_define

T = TypeVar("T", bound="TokenEndpointAuthNone")


@_attrs_define
class TokenEndpointAuthNone:
    """Public OAuth client — no credentials sent on the refresh call.

    Attributes:
        method (Literal['none']):
    """

    method: Literal["none"]

    def to_dict(self) -> dict[str, Any]:
        method = self.method

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "method": method,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        method = cast(Literal["none"], d.pop("method"))
        if method != "none":
            raise ValueError(f"method must match const 'none', got '{method}'")

        token_endpoint_auth_none = cls(
            method=method,
        )

        return token_endpoint_auth_none
