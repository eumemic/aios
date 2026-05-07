from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.environment_config import EnvironmentConfig


T = TypeVar("T", bound="EnvironmentUpdate")


@_attrs_define
class EnvironmentUpdate:
    """Request body for ``PUT /v1/environments/{id}``.

    All fields are optional; omitted fields are preserved.

        Attributes:
            name (None | str | Unset):
            config (EnvironmentConfig | None | Unset):
    """

    name: None | str | Unset = UNSET
    config: EnvironmentConfig | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.environment_config import EnvironmentConfig

        name: None | str | Unset
        if isinstance(self.name, Unset):
            name = UNSET
        else:
            name = self.name

        config: dict[str, Any] | None | Unset
        if isinstance(self.config, Unset):
            config = UNSET
        elif isinstance(self.config, EnvironmentConfig):
            config = self.config.to_dict()
        else:
            config = self.config

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if name is not UNSET:
            field_dict["name"] = name
        if config is not UNSET:
            field_dict["config"] = config

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.environment_config import EnvironmentConfig

        d = dict(src_dict)

        def _parse_name(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        name = _parse_name(d.pop("name", UNSET))

        def _parse_config(data: object) -> EnvironmentConfig | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                config_type_0 = EnvironmentConfig.from_dict(data)

                return config_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(EnvironmentConfig | None | Unset, data)

        config = _parse_config(d.pop("config", UNSET))

        environment_update = cls(
            name=name,
            config=config,
        )

        return environment_update
