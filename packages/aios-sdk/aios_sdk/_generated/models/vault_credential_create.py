from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from dateutil.parser import isoparse

from ..models.vault_credential_create_auth_type import VaultCredentialCreateAuthType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.token_endpoint_auth_basic import TokenEndpointAuthBasic
    from ..models.token_endpoint_auth_none import TokenEndpointAuthNone
    from ..models.token_endpoint_auth_post import TokenEndpointAuthPost
    from ..models.vault_credential_create_metadata import VaultCredentialCreateMetadata


T = TypeVar("T", bound="VaultCredentialCreate")


@_attrs_define
class VaultCredentialCreate:
    """Request body for ``POST /v1/vaults/{vault_id}/credentials``.

    All secret fields are write-only. The structural fields — ``target_url``,
    ``secret_name``, ``allowed_hosts``, and ``auth_type`` — are immutable after
    creation; only the secret (and ``display_name``/``metadata``) can be
    rotated via PUT, so changing a credential's egress scope means archiving
    and recreating it. The service layer validates required secret fields per
    ``auth_type``; this model validates the structural shape (which kind
    carries ``target_url`` vs ``secret_name``/``allowed_hosts``).

        Attributes:
            auth_type (VaultCredentialCreateAuthType):
            access_token (None | str | Unset):
            expires_at (datetime.datetime | None | Unset):
            client_id (None | str | Unset):
            refresh_token (None | str | Unset):
            token_endpoint (None | str | Unset):
            token_endpoint_auth (None | TokenEndpointAuthBasic | TokenEndpointAuthNone | TokenEndpointAuthPost | Unset):
            scope (None | str | Unset):
            resource (None | str | Unset):
            token (None | str | Unset):
            username (None | str | Unset):
            password (None | str | Unset):
            header_name (None | str | Unset):
            header_value (None | str | Unset):
            secret_value (None | str | Unset):
            display_name (None | str | Unset):
            metadata (VaultCredentialCreateMetadata | Unset):
            target_url (None | str | Unset):
            secret_name (None | str | Unset):
            allowed_hosts (list[str] | None | Unset):
    """

    auth_type: VaultCredentialCreateAuthType
    access_token: None | str | Unset = UNSET
    expires_at: datetime.datetime | None | Unset = UNSET
    client_id: None | str | Unset = UNSET
    refresh_token: None | str | Unset = UNSET
    token_endpoint: None | str | Unset = UNSET
    token_endpoint_auth: (
        None
        | TokenEndpointAuthBasic
        | TokenEndpointAuthNone
        | TokenEndpointAuthPost
        | Unset
    ) = UNSET
    scope: None | str | Unset = UNSET
    resource: None | str | Unset = UNSET
    token: None | str | Unset = UNSET
    username: None | str | Unset = UNSET
    password: None | str | Unset = UNSET
    header_name: None | str | Unset = UNSET
    header_value: None | str | Unset = UNSET
    secret_value: None | str | Unset = UNSET
    display_name: None | str | Unset = UNSET
    metadata: VaultCredentialCreateMetadata | Unset = UNSET
    target_url: None | str | Unset = UNSET
    secret_name: None | str | Unset = UNSET
    allowed_hosts: list[str] | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.token_endpoint_auth_basic import TokenEndpointAuthBasic
        from ..models.token_endpoint_auth_none import TokenEndpointAuthNone
        from ..models.token_endpoint_auth_post import TokenEndpointAuthPost

        auth_type = self.auth_type.value

        access_token: None | str | Unset
        if isinstance(self.access_token, Unset):
            access_token = UNSET
        else:
            access_token = self.access_token

        expires_at: None | str | Unset
        if isinstance(self.expires_at, Unset):
            expires_at = UNSET
        elif isinstance(self.expires_at, datetime.datetime):
            expires_at = self.expires_at.isoformat()
        else:
            expires_at = self.expires_at

        client_id: None | str | Unset
        if isinstance(self.client_id, Unset):
            client_id = UNSET
        else:
            client_id = self.client_id

        refresh_token: None | str | Unset
        if isinstance(self.refresh_token, Unset):
            refresh_token = UNSET
        else:
            refresh_token = self.refresh_token

        token_endpoint: None | str | Unset
        if isinstance(self.token_endpoint, Unset):
            token_endpoint = UNSET
        else:
            token_endpoint = self.token_endpoint

        token_endpoint_auth: dict[str, Any] | None | Unset
        if isinstance(self.token_endpoint_auth, Unset):
            token_endpoint_auth = UNSET
        elif isinstance(self.token_endpoint_auth, TokenEndpointAuthNone):
            token_endpoint_auth = self.token_endpoint_auth.to_dict()
        elif isinstance(self.token_endpoint_auth, TokenEndpointAuthBasic):
            token_endpoint_auth = self.token_endpoint_auth.to_dict()
        elif isinstance(self.token_endpoint_auth, TokenEndpointAuthPost):
            token_endpoint_auth = self.token_endpoint_auth.to_dict()
        else:
            token_endpoint_auth = self.token_endpoint_auth

        scope: None | str | Unset
        if isinstance(self.scope, Unset):
            scope = UNSET
        else:
            scope = self.scope

        resource: None | str | Unset
        if isinstance(self.resource, Unset):
            resource = UNSET
        else:
            resource = self.resource

        token: None | str | Unset
        if isinstance(self.token, Unset):
            token = UNSET
        else:
            token = self.token

        username: None | str | Unset
        if isinstance(self.username, Unset):
            username = UNSET
        else:
            username = self.username

        password: None | str | Unset
        if isinstance(self.password, Unset):
            password = UNSET
        else:
            password = self.password

        header_name: None | str | Unset
        if isinstance(self.header_name, Unset):
            header_name = UNSET
        else:
            header_name = self.header_name

        header_value: None | str | Unset
        if isinstance(self.header_value, Unset):
            header_value = UNSET
        else:
            header_value = self.header_value

        secret_value: None | str | Unset
        if isinstance(self.secret_value, Unset):
            secret_value = UNSET
        else:
            secret_value = self.secret_value

        display_name: None | str | Unset
        if isinstance(self.display_name, Unset):
            display_name = UNSET
        else:
            display_name = self.display_name

        metadata: dict[str, Any] | Unset = UNSET
        if not isinstance(self.metadata, Unset):
            metadata = self.metadata.to_dict()

        target_url: None | str | Unset
        if isinstance(self.target_url, Unset):
            target_url = UNSET
        else:
            target_url = self.target_url

        secret_name: None | str | Unset
        if isinstance(self.secret_name, Unset):
            secret_name = UNSET
        else:
            secret_name = self.secret_name

        allowed_hosts: list[str] | None | Unset
        if isinstance(self.allowed_hosts, Unset):
            allowed_hosts = UNSET
        elif isinstance(self.allowed_hosts, list):
            allowed_hosts = self.allowed_hosts

        else:
            allowed_hosts = self.allowed_hosts

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "auth_type": auth_type,
            }
        )
        if access_token is not UNSET:
            field_dict["access_token"] = access_token
        if expires_at is not UNSET:
            field_dict["expires_at"] = expires_at
        if client_id is not UNSET:
            field_dict["client_id"] = client_id
        if refresh_token is not UNSET:
            field_dict["refresh_token"] = refresh_token
        if token_endpoint is not UNSET:
            field_dict["token_endpoint"] = token_endpoint
        if token_endpoint_auth is not UNSET:
            field_dict["token_endpoint_auth"] = token_endpoint_auth
        if scope is not UNSET:
            field_dict["scope"] = scope
        if resource is not UNSET:
            field_dict["resource"] = resource
        if token is not UNSET:
            field_dict["token"] = token
        if username is not UNSET:
            field_dict["username"] = username
        if password is not UNSET:
            field_dict["password"] = password
        if header_name is not UNSET:
            field_dict["header_name"] = header_name
        if header_value is not UNSET:
            field_dict["header_value"] = header_value
        if secret_value is not UNSET:
            field_dict["secret_value"] = secret_value
        if display_name is not UNSET:
            field_dict["display_name"] = display_name
        if metadata is not UNSET:
            field_dict["metadata"] = metadata
        if target_url is not UNSET:
            field_dict["target_url"] = target_url
        if secret_name is not UNSET:
            field_dict["secret_name"] = secret_name
        if allowed_hosts is not UNSET:
            field_dict["allowed_hosts"] = allowed_hosts

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.token_endpoint_auth_basic import TokenEndpointAuthBasic
        from ..models.token_endpoint_auth_none import TokenEndpointAuthNone
        from ..models.token_endpoint_auth_post import TokenEndpointAuthPost
        from ..models.vault_credential_create_metadata import (
            VaultCredentialCreateMetadata,
        )

        d = dict(src_dict)
        auth_type = VaultCredentialCreateAuthType(d.pop("auth_type"))

        def _parse_access_token(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        access_token = _parse_access_token(d.pop("access_token", UNSET))

        def _parse_expires_at(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                expires_at_type_0 = isoparse(data)

                return expires_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        expires_at = _parse_expires_at(d.pop("expires_at", UNSET))

        def _parse_client_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        client_id = _parse_client_id(d.pop("client_id", UNSET))

        def _parse_refresh_token(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        refresh_token = _parse_refresh_token(d.pop("refresh_token", UNSET))

        def _parse_token_endpoint(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        token_endpoint = _parse_token_endpoint(d.pop("token_endpoint", UNSET))

        def _parse_token_endpoint_auth(
            data: object,
        ) -> (
            None
            | TokenEndpointAuthBasic
            | TokenEndpointAuthNone
            | TokenEndpointAuthPost
            | Unset
        ):
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                token_endpoint_auth_type_0_type_0 = TokenEndpointAuthNone.from_dict(
                    data
                )

                return token_endpoint_auth_type_0_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                token_endpoint_auth_type_0_type_1 = TokenEndpointAuthBasic.from_dict(
                    data
                )

                return token_endpoint_auth_type_0_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                token_endpoint_auth_type_0_type_2 = TokenEndpointAuthPost.from_dict(
                    data
                )

                return token_endpoint_auth_type_0_type_2
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(
                None
                | TokenEndpointAuthBasic
                | TokenEndpointAuthNone
                | TokenEndpointAuthPost
                | Unset,
                data,
            )

        token_endpoint_auth = _parse_token_endpoint_auth(
            d.pop("token_endpoint_auth", UNSET)
        )

        def _parse_scope(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        scope = _parse_scope(d.pop("scope", UNSET))

        def _parse_resource(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        resource = _parse_resource(d.pop("resource", UNSET))

        def _parse_token(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        token = _parse_token(d.pop("token", UNSET))

        def _parse_username(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        username = _parse_username(d.pop("username", UNSET))

        def _parse_password(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        password = _parse_password(d.pop("password", UNSET))

        def _parse_header_name(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        header_name = _parse_header_name(d.pop("header_name", UNSET))

        def _parse_header_value(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        header_value = _parse_header_value(d.pop("header_value", UNSET))

        def _parse_secret_value(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        secret_value = _parse_secret_value(d.pop("secret_value", UNSET))

        def _parse_display_name(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        display_name = _parse_display_name(d.pop("display_name", UNSET))

        _metadata = d.pop("metadata", UNSET)
        metadata: VaultCredentialCreateMetadata | Unset
        if isinstance(_metadata, Unset):
            metadata = UNSET
        else:
            metadata = VaultCredentialCreateMetadata.from_dict(_metadata)

        def _parse_target_url(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        target_url = _parse_target_url(d.pop("target_url", UNSET))

        def _parse_secret_name(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        secret_name = _parse_secret_name(d.pop("secret_name", UNSET))

        def _parse_allowed_hosts(data: object) -> list[str] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                allowed_hosts_type_0 = cast(list[str], data)

                return allowed_hosts_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None | Unset, data)

        allowed_hosts = _parse_allowed_hosts(d.pop("allowed_hosts", UNSET))

        vault_credential_create = cls(
            auth_type=auth_type,
            access_token=access_token,
            expires_at=expires_at,
            client_id=client_id,
            refresh_token=refresh_token,
            token_endpoint=token_endpoint,
            token_endpoint_auth=token_endpoint_auth,
            scope=scope,
            resource=resource,
            token=token,
            username=username,
            password=password,
            header_name=header_name,
            header_value=header_value,
            secret_value=secret_value,
            display_name=display_name,
            metadata=metadata,
            target_url=target_url,
            secret_name=secret_name,
            allowed_hosts=allowed_hosts,
        )

        return vault_credential_create
