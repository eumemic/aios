from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.model_provider_create_credentials_type_0 import (
        ModelProviderCreateCredentialsType0,
    )
    from ..models.model_provider_create_litellm_defaults import (
        ModelProviderCreateLitellmDefaults,
    )


T = TypeVar("T", bound="ModelProviderCreate")


@_attrs_define
class ModelProviderCreate:
    """Request body for ``POST /v1/model-providers``.

    ``provider`` is a LiteLLM provider name (e.g. ``anthropic``, ``openai``,
    ``openrouter``) — lower-cased and stripped so it matches what
    ``litellm.get_llm_provider`` returns at resolve time regardless of the
    caller's casing. ``api_key`` is write-only and required in v1 (a keyless
    arm for unauthenticated self-hosted endpoints is a documented future
    extension, not yet supported).

        Attributes:
            provider (str):
            api_key (None | str | Unset):
            api_base (None | str | Unset):
            credentials (ModelProviderCreateCredentialsType0 | None | Unset):
            litellm_defaults (ModelProviderCreateLitellmDefaults | Unset):
    """

    provider: str
    api_key: None | str | Unset = UNSET
    api_base: None | str | Unset = UNSET
    credentials: ModelProviderCreateCredentialsType0 | None | Unset = UNSET
    litellm_defaults: ModelProviderCreateLitellmDefaults | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.model_provider_create_credentials_type_0 import (
            ModelProviderCreateCredentialsType0,
        )

        provider = self.provider

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
        elif isinstance(self.credentials, ModelProviderCreateCredentialsType0):
            credentials = self.credentials.to_dict()
        else:
            credentials = self.credentials

        litellm_defaults: dict[str, Any] | Unset = UNSET
        if not isinstance(self.litellm_defaults, Unset):
            litellm_defaults = self.litellm_defaults.to_dict()

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "provider": provider,
            }
        )
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
        from ..models.model_provider_create_credentials_type_0 import (
            ModelProviderCreateCredentialsType0,
        )
        from ..models.model_provider_create_litellm_defaults import (
            ModelProviderCreateLitellmDefaults,
        )

        d = dict(src_dict)
        provider = d.pop("provider")

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
        ) -> ModelProviderCreateCredentialsType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                credentials_type_0 = ModelProviderCreateCredentialsType0.from_dict(data)

                return credentials_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(ModelProviderCreateCredentialsType0 | None | Unset, data)

        credentials = _parse_credentials(d.pop("credentials", UNSET))

        _litellm_defaults = d.pop("litellm_defaults", UNSET)
        litellm_defaults: ModelProviderCreateLitellmDefaults | Unset
        if isinstance(_litellm_defaults, Unset):
            litellm_defaults = UNSET
        else:
            litellm_defaults = ModelProviderCreateLitellmDefaults.from_dict(
                _litellm_defaults
            )

        model_provider_create = cls(
            provider=provider,
            api_key=api_key,
            api_base=api_base,
            credentials=credentials,
            litellm_defaults=litellm_defaults,
        )

        return model_provider_create
