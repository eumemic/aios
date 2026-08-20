from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.model_provider_create_model_routes import (
        ModelProviderCreateModelRoutes,
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
            api_key (str):
            api_base (None | str | Unset):
            model_routes (ModelProviderCreateModelRoutes | Unset): Bare model aliases mapped to canonical, fully-qualified
                LiteLLM routes.
    """

    provider: str
    api_key: str
    api_base: None | str | Unset = UNSET
    model_routes: ModelProviderCreateModelRoutes | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        provider = self.provider

        api_key = self.api_key

        api_base: None | str | Unset
        if isinstance(self.api_base, Unset):
            api_base = UNSET
        else:
            api_base = self.api_base

        model_routes: dict[str, Any] | Unset = UNSET
        if not isinstance(self.model_routes, Unset):
            model_routes = self.model_routes.to_dict()

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "provider": provider,
                "api_key": api_key,
            }
        )
        if api_base is not UNSET:
            field_dict["api_base"] = api_base
        if model_routes is not UNSET:
            field_dict["model_routes"] = model_routes

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.model_provider_create_model_routes import (
            ModelProviderCreateModelRoutes,
        )

        d = dict(src_dict)
        provider = d.pop("provider")

        api_key = d.pop("api_key")

        def _parse_api_base(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        api_base = _parse_api_base(d.pop("api_base", UNSET))

        _model_routes = d.pop("model_routes", UNSET)
        model_routes: ModelProviderCreateModelRoutes | Unset
        if isinstance(_model_routes, Unset):
            model_routes = UNSET
        else:
            model_routes = ModelProviderCreateModelRoutes.from_dict(_model_routes)

        model_provider_create = cls(
            provider=provider,
            api_key=api_key,
            api_base=api_base,
            model_routes=model_routes,
        )

        return model_provider_create
