from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.model_provider_update_litellm_defaults import (
        ModelProviderUpdateLitellmDefaults,
    )


T = TypeVar("T", bound="ModelProviderUpdate")


@_attrs_define
class ModelProviderUpdate:
    """Request body for ``PUT /v1/model-providers/{id}``.

    ``api_key`` omitted → keep the existing key (rotation is opt-in via an
    explicit value; there is no way to clear it back to unset in v1 — archive
    and recreate instead). ``api_base`` omitted → keep; explicit ``null`` →
    clear (checked via ``model_fields_set``, not a sentinel default, since
    ``None`` is itself a valid target value).

        Attributes:
            api_key (None | str | Unset):
            api_base (None | str | Unset):
            litellm_defaults (ModelProviderUpdateLitellmDefaults | Unset):
    """

    api_key: None | str | Unset = UNSET
    api_base: None | str | Unset = UNSET
    litellm_defaults: ModelProviderUpdateLitellmDefaults | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        api_key: None | str | Unset
        if isinstance(self.api_key, Unset):
            api_key = UNSET
        else:
            api_key = self.api_key

        api_base: None | str | Unset
        if isinstance(self.api_base, Unset):
            api_base = UNSET
        else:
            api_base = self.api_base

        litellm_defaults: dict[str, Any] | Unset = UNSET
        if not isinstance(self.litellm_defaults, Unset):
            litellm_defaults = self.litellm_defaults.to_dict()

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if api_key is not UNSET:
            field_dict["api_key"] = api_key
        if api_base is not UNSET:
            field_dict["api_base"] = api_base
        if litellm_defaults is not UNSET:
            field_dict["litellm_defaults"] = litellm_defaults

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.model_provider_update_litellm_defaults import (
            ModelProviderUpdateLitellmDefaults,
        )

        d = dict(src_dict)

        def _parse_api_key(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        api_key = _parse_api_key(d.pop("api_key", UNSET))

        def _parse_api_base(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        api_base = _parse_api_base(d.pop("api_base", UNSET))

        _litellm_defaults = d.pop("litellm_defaults", UNSET)
        litellm_defaults: ModelProviderUpdateLitellmDefaults | Unset
        if isinstance(_litellm_defaults, Unset):
            litellm_defaults = UNSET
        else:
            litellm_defaults = ModelProviderUpdateLitellmDefaults.from_dict(
                _litellm_defaults
            )

        model_provider_update = cls(
            api_key=api_key,
            api_base=api_base,
            litellm_defaults=litellm_defaults,
        )

        return model_provider_update
