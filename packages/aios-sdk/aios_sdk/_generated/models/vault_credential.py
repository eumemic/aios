from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.vault_credential_auth_type import VaultCredentialAuthType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.vault_credential_metadata import VaultCredentialMetadata


T = TypeVar("T", bound="VaultCredential")


@_attrs_define
class VaultCredential:
    """Read view of a vault credential. Secrets are never returned.

    Attributes:
        id (str):
        vault_id (str):
        display_name (None | str):
        mcp_server_url (str):
        auth_type (VaultCredentialAuthType):
        metadata (VaultCredentialMetadata):
        created_at (datetime.datetime):
        updated_at (datetime.datetime):
        archived_at (datetime.datetime | None | Unset):
    """

    id: str
    vault_id: str
    display_name: None | str
    mcp_server_url: str
    auth_type: VaultCredentialAuthType
    metadata: VaultCredentialMetadata
    created_at: datetime.datetime
    updated_at: datetime.datetime
    archived_at: datetime.datetime | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        vault_id = self.vault_id

        display_name: None | str
        display_name = self.display_name

        mcp_server_url = self.mcp_server_url

        auth_type = self.auth_type.value

        metadata = self.metadata.to_dict()

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        archived_at: None | str | Unset
        if isinstance(self.archived_at, Unset):
            archived_at = UNSET
        elif isinstance(self.archived_at, datetime.datetime):
            archived_at = self.archived_at.isoformat()
        else:
            archived_at = self.archived_at

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "vault_id": vault_id,
                "display_name": display_name,
                "mcp_server_url": mcp_server_url,
                "auth_type": auth_type,
                "metadata": metadata,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        if archived_at is not UNSET:
            field_dict["archived_at"] = archived_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.vault_credential_metadata import VaultCredentialMetadata

        d = dict(src_dict)
        id = d.pop("id")

        vault_id = d.pop("vault_id")

        def _parse_display_name(data: object) -> None | str:
            if data is None:
                return data
            return cast(None | str, data)

        display_name = _parse_display_name(d.pop("display_name"))

        mcp_server_url = d.pop("mcp_server_url")

        auth_type = VaultCredentialAuthType(d.pop("auth_type"))

        metadata = VaultCredentialMetadata.from_dict(d.pop("metadata"))

        created_at = isoparse(d.pop("created_at"))

        updated_at = isoparse(d.pop("updated_at"))

        def _parse_archived_at(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                archived_at_type_0 = isoparse(data)

                return archived_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        archived_at = _parse_archived_at(d.pop("archived_at", UNSET))

        vault_credential = cls(
            id=id,
            vault_id=vault_id,
            display_name=display_name,
            mcp_server_url=mcp_server_url,
            auth_type=auth_type,
            metadata=metadata,
            created_at=created_at,
            updated_at=updated_at,
            archived_at=archived_at,
        )

        vault_credential.additional_properties = d
        return vault_credential

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
