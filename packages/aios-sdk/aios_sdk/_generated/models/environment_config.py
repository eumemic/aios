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
        image (None | str | Unset): Container image for sessions bound to this environment. When unset, sessions
            provision from the worker's global ``settings.docker_image``. Lets a purpose-built environment (e.g. an autodev
            dev image with toolchains baked in) pin its own image without changing the image every other session on the
            shared worker uses (issue #724). Accepts any reference the worker's Docker daemon can resolve: a registry image
            (``ghcr.io/eumemic/aios-sandbox:pinned``) or a bare local tag for development.
        packages (EnvironmentConfigPackagesType0 | None | Unset): Package manager → package list, e.g. {"pip":
            ["pandas"], "npm": ["express"]}.
        networking (LimitedNetworking | None | UnrestrictedNetworking | Unset): Network access rules.  None or {"type":
            "unrestricted"} for full access; {"type": "limited", "allowed_hosts": [...]} to restrict.
        env (EnvironmentConfigEnvType0 | None | Unset): Environment variables injected into every session container
            using this environment.  Per-session env overrides these. A vaulted environment_variable credential whose
            secret_name matches a key — here or in the per-session env — outranks both: that key resolves to the
            credential's opaque placeholder, not the value set here.
        snapshot_budget_bytes (int | None | Unset): Per-session snapshot budget in **unique** bytes for sessions bound
            to this environment (durable session sandboxes). When unset, falls back to the worker's global
            ``settings.sandbox_snapshot_budget_bytes`` (4 GiB). When a session's unique snapshot bytes would exceed this at
            teardown, the snapshot flattens (collapse + whiteout) instead of growing another layer — commit-and-flag, never
            a refusal. Replaces the former ``disk_bytes`` writable-layer cap, which required overlay2+pquota and never
            worked on prod ext4. Minimum 10 MiB.
        bash_timeout_seconds (int | None | Unset): Ceiling, in seconds, for a single bash tool call in sessions bound to
            this environment. When unset, falls back to the worker's global ``settings.bash_default_timeout_seconds``
            (120s). Lets heavy dev workloads run >120s commands without raising the global default for every session on the
            worker. The agent can still request a shorter per-call timeout; this is the maximum it is capped to (issue
            #725).
    """

    image: None | str | Unset = UNSET
    packages: EnvironmentConfigPackagesType0 | None | Unset = UNSET
    networking: LimitedNetworking | None | UnrestrictedNetworking | Unset = UNSET
    env: EnvironmentConfigEnvType0 | None | Unset = UNSET
    snapshot_budget_bytes: int | None | Unset = UNSET
    bash_timeout_seconds: int | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.environment_config_env_type_0 import EnvironmentConfigEnvType0
        from ..models.environment_config_packages_type_0 import (
            EnvironmentConfigPackagesType0,
        )
        from ..models.limited_networking import LimitedNetworking
        from ..models.unrestricted_networking import UnrestrictedNetworking

        image: None | str | Unset
        if isinstance(self.image, Unset):
            image = UNSET
        else:
            image = self.image

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

        snapshot_budget_bytes: int | None | Unset
        if isinstance(self.snapshot_budget_bytes, Unset):
            snapshot_budget_bytes = UNSET
        else:
            snapshot_budget_bytes = self.snapshot_budget_bytes

        bash_timeout_seconds: int | None | Unset
        if isinstance(self.bash_timeout_seconds, Unset):
            bash_timeout_seconds = UNSET
        else:
            bash_timeout_seconds = self.bash_timeout_seconds

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if image is not UNSET:
            field_dict["image"] = image
        if packages is not UNSET:
            field_dict["packages"] = packages
        if networking is not UNSET:
            field_dict["networking"] = networking
        if env is not UNSET:
            field_dict["env"] = env
        if snapshot_budget_bytes is not UNSET:
            field_dict["snapshot_budget_bytes"] = snapshot_budget_bytes
        if bash_timeout_seconds is not UNSET:
            field_dict["bash_timeout_seconds"] = bash_timeout_seconds

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

        def _parse_image(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        image = _parse_image(d.pop("image", UNSET))

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

        def _parse_snapshot_budget_bytes(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        snapshot_budget_bytes = _parse_snapshot_budget_bytes(
            d.pop("snapshot_budget_bytes", UNSET)
        )

        def _parse_bash_timeout_seconds(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        bash_timeout_seconds = _parse_bash_timeout_seconds(
            d.pop("bash_timeout_seconds", UNSET)
        )

        environment_config = cls(
            image=image,
            packages=packages,
            networking=networking,
            env=env,
            snapshot_budget_bytes=snapshot_budget_bytes,
            bash_timeout_seconds=bash_timeout_seconds,
        )

        return environment_config
