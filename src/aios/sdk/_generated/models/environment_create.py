from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.environment_config import EnvironmentConfig


T = TypeVar("T", bound="EnvironmentCreate")


@_attrs_define
class EnvironmentCreate:
    """Request body for `POST /v1/environments`.

    Attributes:
        name (str):
        config (EnvironmentConfig | Unset): Container configuration for an environment.
    """

    name: str
    config: EnvironmentConfig | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        config: dict[str, Any] | Unset = UNSET
        if not isinstance(self.config, Unset):
            config = self.config.to_dict()

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "name": name,
            }
        )
        if config is not UNSET:
            field_dict["config"] = config

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.environment_config import EnvironmentConfig

        d = dict(src_dict)
        name = d.pop("name")

        _config = d.pop("config", UNSET)
        config: EnvironmentConfig | Unset
        if isinstance(_config, Unset):
            config = UNSET
        else:
            config = EnvironmentConfig.from_dict(_config)

        environment_create = cls(
            name=name,
            config=config,
        )

        return environment_create
