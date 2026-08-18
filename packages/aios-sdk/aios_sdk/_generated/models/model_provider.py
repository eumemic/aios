from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.model_provider_litellm_defaults import ModelProviderLitellmDefaults


T = TypeVar("T", bound="ModelProvider")


@_attrs_define
class ModelProvider:
    """Read view of a model-provider config. ``api_key`` is never returned.

    Attributes:
        id (str):
        provider (str):
        api_key_set (bool):
        created_at (datetime.datetime):
        updated_at (datetime.datetime):
        api_base (None | str | Unset):
        credentials_set (bool | Unset):  Default: False.
        litellm_defaults (ModelProviderLitellmDefaults | Unset):
        version (int | Unset):  Default: 1.
        archived_at (datetime.datetime | None | Unset):
    """

    id: str
    provider: str
    api_key_set: bool
    created_at: datetime.datetime
    updated_at: datetime.datetime
    api_base: None | str | Unset = UNSET
    credentials_set: bool | Unset = False
    litellm_defaults: ModelProviderLitellmDefaults | Unset = UNSET
    version: int | Unset = 1
    archived_at: datetime.datetime | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        provider = self.provider

        api_key_set = self.api_key_set

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        api_base: None | str | Unset
        if isinstance(self.api_base, Unset):
            api_base = UNSET
        else:
            api_base = self.api_base

        credentials_set = self.credentials_set

        litellm_defaults: dict[str, Any] | Unset = UNSET
        if not isinstance(self.litellm_defaults, Unset):
            litellm_defaults = self.litellm_defaults.to_dict()

        version = self.version

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
                "provider": provider,
                "api_key_set": api_key_set,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        if api_base is not UNSET:
            field_dict["api_base"] = api_base
        if credentials_set is not UNSET:
            field_dict["credentials_set"] = credentials_set
        if litellm_defaults is not UNSET:
            field_dict["litellm_defaults"] = litellm_defaults
        if version is not UNSET:
            field_dict["version"] = version
        if archived_at is not UNSET:
            field_dict["archived_at"] = archived_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.model_provider_litellm_defaults import (
            ModelProviderLitellmDefaults,
        )

        d = dict(src_dict)
        id = d.pop("id")

        provider = d.pop("provider")

        api_key_set = d.pop("api_key_set")

        created_at = isoparse(d.pop("created_at"))

        updated_at = isoparse(d.pop("updated_at"))

        def _parse_api_base(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        api_base = _parse_api_base(d.pop("api_base", UNSET))

        credentials_set = d.pop("credentials_set", UNSET)

        _litellm_defaults = d.pop("litellm_defaults", UNSET)
        litellm_defaults: ModelProviderLitellmDefaults | Unset
        if isinstance(_litellm_defaults, Unset):
            litellm_defaults = UNSET
        else:
            litellm_defaults = ModelProviderLitellmDefaults.from_dict(_litellm_defaults)

        version = d.pop("version", UNSET)

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

        model_provider = cls(
            id=id,
            provider=provider,
            api_key_set=api_key_set,
            created_at=created_at,
            updated_at=updated_at,
            api_base=api_base,
            credentials_set=credentials_set,
            litellm_defaults=litellm_defaults,
            version=version,
            archived_at=archived_at,
        )

        model_provider.additional_properties = d
        return model_provider

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
