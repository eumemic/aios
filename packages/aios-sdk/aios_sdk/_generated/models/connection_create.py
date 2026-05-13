from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.connection_create_metadata import ConnectionCreateMetadata
    from ..models.connection_create_secrets_type_0 import ConnectionCreateSecretsType0


T = TypeVar("T", bound="ConnectionCreate")


@_attrs_define
class ConnectionCreate:
    """Request body for ``POST /v1/connections``.

    Created in detached mode — neither ``session_id`` nor
    ``session_template_id`` is set.  Use ``POST .../attach`` or
    ``POST .../configure-per-chat`` afterward to bind a routing mode.

    ``connector`` and ``account`` may not contain ``/`` — they're used
    in the focal-channel address scheme ``{connector}/{account}/{chat_id}``
    and a ``/`` would create ambiguous segment boundaries.

        Attributes:
            connector (str):
            account (str):
            metadata (ConnectionCreateMetadata | Unset):
            secrets (ConnectionCreateSecretsType0 | None | Unset): Platform credentials (e.g. ``bot_token``).  Encrypted at
                rest via the server's ``AIOS_VAULT_KEY``; only ever read back via the connector-scoped ``GET
                /v1/connectors/secrets``.  Operator-facing reads return ``secrets_set: bool`` instead of values.
    """

    connector: str
    account: str
    metadata: ConnectionCreateMetadata | Unset = UNSET
    secrets: ConnectionCreateSecretsType0 | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.connection_create_secrets_type_0 import (
            ConnectionCreateSecretsType0,
        )

        connector = self.connector

        account = self.account

        metadata: dict[str, Any] | Unset = UNSET
        if not isinstance(self.metadata, Unset):
            metadata = self.metadata.to_dict()

        secrets: dict[str, Any] | None | Unset
        if isinstance(self.secrets, Unset):
            secrets = UNSET
        elif isinstance(self.secrets, ConnectionCreateSecretsType0):
            secrets = self.secrets.to_dict()
        else:
            secrets = self.secrets

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "connector": connector,
                "account": account,
            }
        )
        if metadata is not UNSET:
            field_dict["metadata"] = metadata
        if secrets is not UNSET:
            field_dict["secrets"] = secrets

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.connection_create_metadata import ConnectionCreateMetadata
        from ..models.connection_create_secrets_type_0 import (
            ConnectionCreateSecretsType0,
        )

        d = dict(src_dict)
        connector = d.pop("connector")

        account = d.pop("account")

        _metadata = d.pop("metadata", UNSET)
        metadata: ConnectionCreateMetadata | Unset
        if isinstance(_metadata, Unset):
            metadata = UNSET
        else:
            metadata = ConnectionCreateMetadata.from_dict(_metadata)

        def _parse_secrets(data: object) -> ConnectionCreateSecretsType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                secrets_type_0 = ConnectionCreateSecretsType0.from_dict(data)

                return secrets_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(ConnectionCreateSecretsType0 | None | Unset, data)

        secrets = _parse_secrets(d.pop("secrets", UNSET))

        connection_create = cls(
            connector=connector,
            account=account,
            metadata=metadata,
            secrets=secrets,
        )

        return connection_create
