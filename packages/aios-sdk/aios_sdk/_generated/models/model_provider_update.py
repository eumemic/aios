from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.model_provider_update_credentials_type_0 import (
        ModelProviderUpdateCredentialsType0,
    )
    from ..models.model_provider_update_litellm_defaults_type_0 import (
        ModelProviderUpdateLitellmDefaultsType0,
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
            credentials (ModelProviderUpdateCredentialsType0 | None | Unset):
            litellm_defaults (ModelProviderUpdateLitellmDefaultsType0 | None | Unset):
    """

    api_key: None | str | Unset = UNSET
    api_base: None | str | Unset = UNSET
    credentials: ModelProviderUpdateCredentialsType0 | None | Unset = UNSET
    litellm_defaults: ModelProviderUpdateLitellmDefaultsType0 | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.model_provider_update_credentials_type_0 import (
            ModelProviderUpdateCredentialsType0,
        )
        from ..models.model_provider_update_litellm_defaults_type_0 import (
            ModelProviderUpdateLitellmDefaultsType0,
        )

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

        credentials: dict[str, Any] | None | Unset
        if isinstance(self.credentials, Unset):
            credentials = UNSET
        elif isinstance(self.credentials, ModelProviderUpdateCredentialsType0):
            credentials = self.credentials.to_dict()
        else:
            credentials = self.credentials

        litellm_defaults: dict[str, Any] | None | Unset
        if isinstance(self.litellm_defaults, Unset):
            litellm_defaults = UNSET
        elif isinstance(self.litellm_defaults, ModelProviderUpdateLitellmDefaultsType0):
            litellm_defaults = self.litellm_defaults.to_dict()
        else:
            litellm_defaults = self.litellm_defaults

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if api_key is not UNSET:
            field_dict["api_key"] = api_key
        if api_base is not UNSET:
            field_dict["api_base"] = api_base
        if credentials is not UNSET:
            field_dict["credentials"] = credentials
        if litellm_defaults is not UNSET:
            field_dict["litellm_defaults"] = litellm_defaults

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.model_provider_update_credentials_type_0 import (
            ModelProviderUpdateCredentialsType0,
        )
        from ..models.model_provider_update_litellm_defaults_type_0 import (
            ModelProviderUpdateLitellmDefaultsType0,
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

        def _parse_credentials(
            data: object,
        ) -> ModelProviderUpdateCredentialsType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                credentials_type_0 = ModelProviderUpdateCredentialsType0.from_dict(data)

                return credentials_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(ModelProviderUpdateCredentialsType0 | None | Unset, data)

        credentials = _parse_credentials(d.pop("credentials", UNSET))

        def _parse_litellm_defaults(
            data: object,
        ) -> ModelProviderUpdateLitellmDefaultsType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                litellm_defaults_type_0 = (
                    ModelProviderUpdateLitellmDefaultsType0.from_dict(data)
                )

                return litellm_defaults_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(ModelProviderUpdateLitellmDefaultsType0 | None | Unset, data)

        litellm_defaults = _parse_litellm_defaults(d.pop("litellm_defaults", UNSET))

        model_provider_update = cls(
            api_key=api_key,
            api_base=api_base,
            credentials=credentials,
            litellm_defaults=litellm_defaults,
        )

        return model_provider_update
