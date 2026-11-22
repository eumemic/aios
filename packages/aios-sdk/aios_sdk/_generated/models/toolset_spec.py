from __future__ import annotations

from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
    cast,
)

from attrs import define as _attrs_define

from ..models.toolset_spec_name import ToolsetSpecName
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.toolset_member_config import ToolsetMemberConfig
    from ..models.toolset_member_defaults import ToolsetMemberDefaults


T = TypeVar("T", bound="ToolsetSpec")


@_attrs_define
class ToolsetSpec:
    """Ingress-only reference to an immutable, curated capability set.

    Toolsets are expanded while validating an agent create/update request. They
    are never persisted or returned: the resolved agent surface contains only
    ordinary :class:`ToolSpec` entries, making every granted capability explicit.

        Attributes:
            type_ (Literal['toolset']):
            name (ToolsetSpecName):
            version (Literal[1]):
            default_config (None | ToolsetMemberDefaults | Unset):
            configs (list[ToolsetMemberConfig] | None | Unset):
    """

    type_: Literal["toolset"]
    name: ToolsetSpecName
    version: Literal[1]
    default_config: None | ToolsetMemberDefaults | Unset = UNSET
    configs: list[ToolsetMemberConfig] | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.toolset_member_defaults import ToolsetMemberDefaults

        type_ = self.type_

        name = self.name.value

        version = self.version

        default_config: dict[str, Any] | None | Unset
        if isinstance(self.default_config, Unset):
            default_config = UNSET
        elif isinstance(self.default_config, ToolsetMemberDefaults):
            default_config = self.default_config.to_dict()
        else:
            default_config = self.default_config

        configs: list[dict[str, Any]] | None | Unset
        if isinstance(self.configs, Unset):
            configs = UNSET
        elif isinstance(self.configs, list):
            configs = []
            for configs_type_0_item_data in self.configs:
                configs_type_0_item = configs_type_0_item_data.to_dict()
                configs.append(configs_type_0_item)

        else:
            configs = self.configs

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "type": type_,
                "name": name,
                "version": version,
            }
        )
        if default_config is not UNSET:
            field_dict["default_config"] = default_config
        if configs is not UNSET:
            field_dict["configs"] = configs

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.toolset_member_config import ToolsetMemberConfig
        from ..models.toolset_member_defaults import ToolsetMemberDefaults

        d = dict(src_dict)
        type_ = cast(Literal["toolset"], d.pop("type"))
        if type_ != "toolset":
            raise ValueError(f"type must match const 'toolset', got '{type_}'")

        name = ToolsetSpecName(d.pop("name"))

        version = cast(Literal[1], d.pop("version"))
        if version != 1:
            raise ValueError(f"version must match const 1, got '{version}'")

        def _parse_default_config(data: object) -> None | ToolsetMemberDefaults | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                default_config_type_0 = ToolsetMemberDefaults.from_dict(data)

                return default_config_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | ToolsetMemberDefaults | Unset, data)

        default_config = _parse_default_config(d.pop("default_config", UNSET))

        def _parse_configs(data: object) -> list[ToolsetMemberConfig] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                configs_type_0 = []
                _configs_type_0 = data
                for configs_type_0_item_data in _configs_type_0:
                    configs_type_0_item = ToolsetMemberConfig.from_dict(
                        configs_type_0_item_data
                    )

                    configs_type_0.append(configs_type_0_item)

                return configs_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[ToolsetMemberConfig] | None | Unset, data)

        configs = _parse_configs(d.pop("configs", UNSET))

        toolset_spec = cls(
            type_=type_,
            name=name,
            version=version,
            default_config=default_config,
            configs=configs,
        )

        return toolset_spec
