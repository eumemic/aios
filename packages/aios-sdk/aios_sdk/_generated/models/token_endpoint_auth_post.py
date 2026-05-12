from __future__ import annotations

from collections.abc import Mapping
from typing import (
    Any,
    Literal,
    TypeVar,
    cast,
)

from attrs import define as _attrs_define

T = TypeVar("T", bound="TokenEndpointAuthPost")


@_attrs_define
class TokenEndpointAuthPost:
    """OAuth client_secret_post — client_secret in the form body.

    Attributes:
        method (Literal['client_secret_post']):
        client_secret (str):
    """

    method: Literal["client_secret_post"]
    client_secret: str

    def to_dict(self) -> dict[str, Any]:
        method = self.method

        client_secret = self.client_secret

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "method": method,
                "client_secret": client_secret,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        method = cast(Literal["client_secret_post"], d.pop("method"))
        if method != "client_secret_post":
            raise ValueError(
                f"method must match const 'client_secret_post', got '{method}'"
            )

        client_secret = d.pop("client_secret")

        token_endpoint_auth_post = cls(
            method=method,
            client_secret=client_secret,
        )

        return token_endpoint_auth_post
