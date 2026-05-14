from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.connection_set_secrets_secrets import ConnectionSetSecretsSecrets


T = TypeVar("T", bound="ConnectionSetSecrets")


@_attrs_define
class ConnectionSetSecrets:
    """Request body for ``PUT /v1/connections/{id}/secrets``.

    Replaces the connection's secrets dict wholesale.  Encrypted at
    rest server-side via ``AIOS_VAULT_KEY``; the operator never reads
    them back.

    Pass an empty dict to clear secrets.

        Attributes:
            secrets (ConnectionSetSecretsSecrets | Unset):
    """

    secrets: ConnectionSetSecretsSecrets | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        secrets: dict[str, Any] | Unset = UNSET
        if not isinstance(self.secrets, Unset):
            secrets = self.secrets.to_dict()

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if secrets is not UNSET:
            field_dict["secrets"] = secrets

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.connection_set_secrets_secrets import ConnectionSetSecretsSecrets

        d = dict(src_dict)
        _secrets = d.pop("secrets", UNSET)
        secrets: ConnectionSetSecretsSecrets | Unset
        if isinstance(_secrets, Unset):
            secrets = UNSET
        else:
            secrets = ConnectionSetSecretsSecrets.from_dict(_secrets)

        connection_set_secrets = cls(
            secrets=secrets,
        )

        return connection_set_secrets
