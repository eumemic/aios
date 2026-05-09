from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.connector_secrets_secrets import ConnectorSecretsSecrets


T = TypeVar("T", bound="ConnectorSecrets")


@_attrs_define
class ConnectorSecrets:
    """Response shape for ``GET /v1/connectors/secrets``.

    Only the connector container's bearer token (which scopes to one
    ``connection_id``) can hit this route.  Returns the decrypted dict
    the operator stored at create / set-secrets time.  Empty dict when
    the connection has no secrets configured.

        Attributes:
            secrets (ConnectorSecretsSecrets | Unset):
    """

    secrets: ConnectorSecretsSecrets | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        secrets: dict[str, Any] | Unset = UNSET
        if not isinstance(self.secrets, Unset):
            secrets = self.secrets.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if secrets is not UNSET:
            field_dict["secrets"] = secrets

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.connector_secrets_secrets import ConnectorSecretsSecrets

        d = dict(src_dict)
        _secrets = d.pop("secrets", UNSET)
        secrets: ConnectorSecretsSecrets | Unset
        if isinstance(_secrets, Unset):
            secrets = UNSET
        else:
            secrets = ConnectorSecretsSecrets.from_dict(_secrets)

        connector_secrets = cls(
            secrets=secrets,
        )

        connector_secrets.additional_properties = d
        return connector_secrets

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
