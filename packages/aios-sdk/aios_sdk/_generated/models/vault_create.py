from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.vault_create_metadata import VaultCreateMetadata


T = TypeVar("T", bound="VaultCreate")


@_attrs_define
class VaultCreate:
    """Request body for ``POST /v1/vaults``.

    Attributes:
        display_name (str):
        metadata (VaultCreateMetadata | Unset):
    """

    display_name: str
    metadata: VaultCreateMetadata | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        display_name = self.display_name

        metadata: dict[str, Any] | Unset = UNSET
        if not isinstance(self.metadata, Unset):
            metadata = self.metadata.to_dict()

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "display_name": display_name,
            }
        )
        if metadata is not UNSET:
            field_dict["metadata"] = metadata

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.vault_create_metadata import VaultCreateMetadata

        d = dict(src_dict)
        display_name = d.pop("display_name")

        _metadata = d.pop("metadata", UNSET)
        metadata: VaultCreateMetadata | Unset
        if isinstance(_metadata, Unset):
            metadata = UNSET
        else:
            metadata = VaultCreateMetadata.from_dict(_metadata)

        vault_create = cls(
            display_name=display_name,
            metadata=metadata,
        )

        return vault_create
