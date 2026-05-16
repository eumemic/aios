from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.http_permission_policy import HttpPermissionPolicy


T = TypeVar("T", bound="HttpRouteSpec")


@_attrs_define
class HttpRouteSpec:
    """One entry in an ``HttpServerSpec.routes`` allowlist.

    ``path_pattern`` is a glob against the request path (``*`` matches one
    segment, ``**`` matches any number of segments).  ``description`` is
    operator-authored prose rendered into the system prompt so the agent
    knows what the route does and how to call it.  ``permission_policy``
    gates *invocation*: ``always_ask`` parks the call in
    ``requires_action`` until the client confirms.

        Attributes:
            path_pattern (str):
            description (None | str | Unset):
            enabled (bool | Unset):  Default: True.
            permission_policy (HttpPermissionPolicy | None | Unset):
    """

    path_pattern: str
    description: None | str | Unset = UNSET
    enabled: bool | Unset = True
    permission_policy: HttpPermissionPolicy | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.http_permission_policy import HttpPermissionPolicy

        path_pattern = self.path_pattern

        description: None | str | Unset
        if isinstance(self.description, Unset):
            description = UNSET
        else:
            description = self.description

        enabled = self.enabled

        permission_policy: dict[str, Any] | None | Unset
        if isinstance(self.permission_policy, Unset):
            permission_policy = UNSET
        elif isinstance(self.permission_policy, HttpPermissionPolicy):
            permission_policy = self.permission_policy.to_dict()
        else:
            permission_policy = self.permission_policy

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "path_pattern": path_pattern,
            }
        )
        if description is not UNSET:
            field_dict["description"] = description
        if enabled is not UNSET:
            field_dict["enabled"] = enabled
        if permission_policy is not UNSET:
            field_dict["permission_policy"] = permission_policy

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.http_permission_policy import HttpPermissionPolicy

        d = dict(src_dict)
        path_pattern = d.pop("path_pattern")

        def _parse_description(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        description = _parse_description(d.pop("description", UNSET))

        enabled = d.pop("enabled", UNSET)

        def _parse_permission_policy(
            data: object,
        ) -> HttpPermissionPolicy | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                permission_policy_type_0 = HttpPermissionPolicy.from_dict(data)

                return permission_policy_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(HttpPermissionPolicy | None | Unset, data)

        permission_policy = _parse_permission_policy(d.pop("permission_policy", UNSET))

        http_route_spec = cls(
            path_pattern=path_pattern,
            description=description,
            enabled=enabled,
            permission_policy=permission_policy,
        )

        return http_route_spec
