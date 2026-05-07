from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.environment_config_env_type_0 import EnvironmentConfigEnvType0
    from ..models.environment_config_packages_type_0 import (
        EnvironmentConfigPackagesType0,
    )
    from ..models.limited_networking import LimitedNetworking
    from ..models.unrestricted_networking import UnrestrictedNetworking


T = TypeVar("T", bound="EnvironmentConfig")


@_attrs_define
class EnvironmentConfig:
    """Container configuration for an environment.

    Attributes:
        packages (EnvironmentConfigPackagesType0 | None | Unset): Package manager → package list, e.g. {"pip":
            ["pandas"], "npm": ["express"]}.
        networking (LimitedNetworking | None | UnrestrictedNetworking | Unset): Network access rules.  None or {"type":
            "unrestricted"} for full access; {"type": "limited", "allowed_hosts": [...]} to restrict.
        env (EnvironmentConfigEnvType0 | None | Unset): Environment variables injected into every session container
            using this environment.  Per-session env overrides these.
    """

    packages: EnvironmentConfigPackagesType0 | None | Unset = UNSET
    networking: LimitedNetworking | None | UnrestrictedNetworking | Unset = UNSET
    env: EnvironmentConfigEnvType0 | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.environment_config_env_type_0 import EnvironmentConfigEnvType0
        from ..models.environment_config_packages_type_0 import (
            EnvironmentConfigPackagesType0,
        )
        from ..models.limited_networking import LimitedNetworking
        from ..models.unrestricted_networking import UnrestrictedNetworking

        packages: dict[str, Any] | None | Unset
        if isinstance(self.packages, Unset):
            packages = UNSET
        elif isinstance(self.packages, EnvironmentConfigPackagesType0):
            packages = self.packages.to_dict()
        else:
            packages = self.packages

        networking: dict[str, Any] | None | Unset
        if isinstance(self.networking, Unset):
            networking = UNSET
        elif isinstance(self.networking, UnrestrictedNetworking):
            networking = self.networking.to_dict()
        elif isinstance(self.networking, LimitedNetworking):
            networking = self.networking.to_dict()
        else:
            networking = self.networking

        env: dict[str, Any] | None | Unset
        if isinstance(self.env, Unset):
            env = UNSET
        elif isinstance(self.env, EnvironmentConfigEnvType0):
            env = self.env.to_dict()
        else:
            env = self.env

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if packages is not UNSET:
            field_dict["packages"] = packages
        if networking is not UNSET:
            field_dict["networking"] = networking
        if env is not UNSET:
            field_dict["env"] = env

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.environment_config_env_type_0 import EnvironmentConfigEnvType0
        from ..models.environment_config_packages_type_0 import (
            EnvironmentConfigPackagesType0,
        )
        from ..models.limited_networking import LimitedNetworking
        from ..models.unrestricted_networking import UnrestrictedNetworking

        d = dict(src_dict)

        def _parse_packages(
            data: object,
        ) -> EnvironmentConfigPackagesType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                packages_type_0 = EnvironmentConfigPackagesType0.from_dict(data)

                return packages_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(EnvironmentConfigPackagesType0 | None | Unset, data)

        packages = _parse_packages(d.pop("packages", UNSET))

        def _parse_networking(
            data: object,
        ) -> LimitedNetworking | None | UnrestrictedNetworking | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                networking_type_0_type_0 = UnrestrictedNetworking.from_dict(data)

                return networking_type_0_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                networking_type_0_type_1 = LimitedNetworking.from_dict(data)

                return networking_type_0_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(LimitedNetworking | None | UnrestrictedNetworking | Unset, data)

        networking = _parse_networking(d.pop("networking", UNSET))

        def _parse_env(data: object) -> EnvironmentConfigEnvType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                env_type_0 = EnvironmentConfigEnvType0.from_dict(data)

                return env_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(EnvironmentConfigEnvType0 | None | Unset, data)

        env = _parse_env(d.pop("env", UNSET))

        environment_config = cls(
            packages=packages,
            networking=networking,
            env=env,
        )

        return environment_config
