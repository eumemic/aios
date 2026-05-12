from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.vault_update_metadata_type_0 import VaultUpdateMetadataType0


T = TypeVar("T", bound="VaultUpdate")


@_attrs_define
class VaultUpdate:
    """Request body for ``PUT /v1/vaults/{vault_id}``.

    Attributes:
        display_name (None | str | Unset):
        metadata (None | Unset | VaultUpdateMetadataType0):
    """

    display_name: None | str | Unset = UNSET
    metadata: None | Unset | VaultUpdateMetadataType0 = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.vault_update_metadata_type_0 import VaultUpdateMetadataType0

        display_name: None | str | Unset
        if isinstance(self.display_name, Unset):
            display_name = UNSET
        else:
            display_name = self.display_name

        metadata: dict[str, Any] | None | Unset
        if isinstance(self.metadata, Unset):
            metadata = UNSET
        elif isinstance(self.metadata, VaultUpdateMetadataType0):
            metadata = self.metadata.to_dict()
        else:
            metadata = self.metadata

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if display_name is not UNSET:
            field_dict["display_name"] = display_name
        if metadata is not UNSET:
            field_dict["metadata"] = metadata

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.vault_update_metadata_type_0 import VaultUpdateMetadataType0

        d = dict(src_dict)

        def _parse_display_name(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        display_name = _parse_display_name(d.pop("display_name", UNSET))

        def _parse_metadata(data: object) -> None | Unset | VaultUpdateMetadataType0:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                metadata_type_0 = VaultUpdateMetadataType0.from_dict(data)

                return metadata_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | VaultUpdateMetadataType0, data)

        metadata = _parse_metadata(d.pop("metadata", UNSET))

        vault_update = cls(
            display_name=display_name,
            metadata=metadata,
        )

        return vault_update
