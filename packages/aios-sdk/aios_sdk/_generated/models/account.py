from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.account_metadata import AccountMetadata


T = TypeVar("T", bound="Account")


@_attrs_define
class Account:
    """
    Attributes:
        id (str):
        parent_account_id (None | str):
        can_mint_children (bool):
        display_name (str):
        metadata (AccountMetadata):
        created_at (datetime.datetime):
        archived_at (datetime.datetime | None | Unset):
    """

    id: str
    parent_account_id: None | str
    can_mint_children: bool
    display_name: str
    metadata: AccountMetadata
    created_at: datetime.datetime
    archived_at: datetime.datetime | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        parent_account_id: None | str
        parent_account_id = self.parent_account_id

        can_mint_children = self.can_mint_children

        display_name = self.display_name

        metadata = self.metadata.to_dict()

        created_at = self.created_at.isoformat()

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
                "parent_account_id": parent_account_id,
                "can_mint_children": can_mint_children,
                "display_name": display_name,
                "metadata": metadata,
                "created_at": created_at,
            }
        )
        if archived_at is not UNSET:
            field_dict["archived_at"] = archived_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.account_metadata import AccountMetadata

        d = dict(src_dict)
        id = d.pop("id")

        def _parse_parent_account_id(data: object) -> None | str:
            if data is None:
                return data
            return cast(None | str, data)

        parent_account_id = _parse_parent_account_id(d.pop("parent_account_id"))

        can_mint_children = d.pop("can_mint_children")

        display_name = d.pop("display_name")

        metadata = AccountMetadata.from_dict(d.pop("metadata"))

        created_at = isoparse(d.pop("created_at"))

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

        account = cls(
            id=id,
            parent_account_id=parent_account_id,
            can_mint_children=can_mint_children,
            display_name=display_name,
            metadata=metadata,
            created_at=created_at,
            archived_at=archived_at,
        )

        account.additional_properties = d
        return account

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
