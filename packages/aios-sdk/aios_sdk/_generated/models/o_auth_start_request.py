from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define

from ..models.o_auth_start_request_token_endpoint_auth_method_type_0 import (
    OAuthStartRequestTokenEndpointAuthMethodType0,
)
from ..types import UNSET, Unset

T = TypeVar("T", bound="OAuthStartRequest")


@_attrs_define
class OAuthStartRequest:
    """Begin an interactive OAuth authorization-code flow for an MCP server.

    With the token fields left blank, the server discovers the target's OAuth
    metadata, registers a client (RFC 7591 Dynamic Client Registration) or uses
    the supplied ``client_id``/``client_secret``, and returns an
    ``authorization_url`` to redirect the user to. The ``redirect_uri`` is the
    console's callback and is reused verbatim on completion.

        Attributes:
            target_url (str):
            redirect_uri (str):
            display_name (None | str | Unset):
            scope (None | str | Unset):
            client_id (None | str | Unset):
            client_secret (None | str | Unset):
            token_endpoint_auth_method (None | OAuthStartRequestTokenEndpointAuthMethodType0 | Unset):
    """

    target_url: str
    redirect_uri: str
    display_name: None | str | Unset = UNSET
    scope: None | str | Unset = UNSET
    client_id: None | str | Unset = UNSET
    client_secret: None | str | Unset = UNSET
    token_endpoint_auth_method: (
        None | OAuthStartRequestTokenEndpointAuthMethodType0 | Unset
    ) = UNSET

    def to_dict(self) -> dict[str, Any]:
        target_url = self.target_url

        redirect_uri = self.redirect_uri

        display_name: None | str | Unset
        if isinstance(self.display_name, Unset):
            display_name = UNSET
        else:
            display_name = self.display_name

        scope: None | str | Unset
        if isinstance(self.scope, Unset):
            scope = UNSET
        else:
            scope = self.scope

        client_id: None | str | Unset
        if isinstance(self.client_id, Unset):
            client_id = UNSET
        else:
            client_id = self.client_id

        client_secret: None | str | Unset
        if isinstance(self.client_secret, Unset):
            client_secret = UNSET
        else:
            client_secret = self.client_secret

        token_endpoint_auth_method: None | str | Unset
        if isinstance(self.token_endpoint_auth_method, Unset):
            token_endpoint_auth_method = UNSET
        elif isinstance(
            self.token_endpoint_auth_method,
            OAuthStartRequestTokenEndpointAuthMethodType0,
        ):
            token_endpoint_auth_method = self.token_endpoint_auth_method.value
        else:
            token_endpoint_auth_method = self.token_endpoint_auth_method

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "target_url": target_url,
                "redirect_uri": redirect_uri,
            }
        )
        if display_name is not UNSET:
            field_dict["display_name"] = display_name
        if scope is not UNSET:
            field_dict["scope"] = scope
        if client_id is not UNSET:
            field_dict["client_id"] = client_id
        if client_secret is not UNSET:
            field_dict["client_secret"] = client_secret
        if token_endpoint_auth_method is not UNSET:
            field_dict["token_endpoint_auth_method"] = token_endpoint_auth_method

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        target_url = d.pop("target_url")

        redirect_uri = d.pop("redirect_uri")

        def _parse_display_name(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        display_name = _parse_display_name(d.pop("display_name", UNSET))

        def _parse_scope(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        scope = _parse_scope(d.pop("scope", UNSET))

        def _parse_client_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        client_id = _parse_client_id(d.pop("client_id", UNSET))

        def _parse_client_secret(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        client_secret = _parse_client_secret(d.pop("client_secret", UNSET))

        def _parse_token_endpoint_auth_method(
            data: object,
        ) -> None | OAuthStartRequestTokenEndpointAuthMethodType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                token_endpoint_auth_method_type_0 = (
                    OAuthStartRequestTokenEndpointAuthMethodType0(data)
                )

                return token_endpoint_auth_method_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(
                None | OAuthStartRequestTokenEndpointAuthMethodType0 | Unset, data
            )

        token_endpoint_auth_method = _parse_token_endpoint_auth_method(
            d.pop("token_endpoint_auth_method", UNSET)
        )

        o_auth_start_request = cls(
            target_url=target_url,
            redirect_uri=redirect_uri,
            display_name=display_name,
            scope=scope,
            client_id=client_id,
            client_secret=client_secret,
            token_endpoint_auth_method=token_endpoint_auth_method,
        )

        return o_auth_start_request
