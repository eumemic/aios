from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.http_route_spec import HttpRouteSpec


T = TypeVar("T", bound="HttpServerSpec")


@_attrs_define
class HttpServerSpec:
    """One entry in an agent's ``http_servers`` list.

    Declares an authenticated HTTP endpoint the agent can reach via the
    ``http_request`` built-in tool.  ``base_url`` is the common URL
    prefix the agent's ``path`` argument is appended to; ``routes`` is
    the allowlist of path patterns the broker permits.  Credentials are
    resolved at request time from the session's bound vaults, keyed on
    ``base_url``.  Secret never enters the sandbox — the worker authors
    the ``Authorization`` header from the vault credential.

        Attributes:
            name (str):
            base_url (str):
            description (None | str | Unset):
            routes (list[HttpRouteSpec] | Unset):
            suppressed_response_status (int | Unset):  Default: 200.
    """

    name: str
    base_url: str
    description: None | str | Unset = UNSET
    routes: list[HttpRouteSpec] | Unset = UNSET
    suppressed_response_status: int | Unset = 200

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        base_url = self.base_url

        description: None | str | Unset
        if isinstance(self.description, Unset):
            description = UNSET
        else:
            description = self.description

        routes: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.routes, Unset):
            routes = []
            for routes_item_data in self.routes:
                routes_item = routes_item_data.to_dict()
                routes.append(routes_item)

        suppressed_response_status = self.suppressed_response_status

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "name": name,
                "base_url": base_url,
            }
        )
        if description is not UNSET:
            field_dict["description"] = description
        if routes is not UNSET:
            field_dict["routes"] = routes
        if suppressed_response_status is not UNSET:
            field_dict["suppressed_response_status"] = suppressed_response_status

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.http_route_spec import HttpRouteSpec

        d = dict(src_dict)
        name = d.pop("name")

        base_url = d.pop("base_url")

        def _parse_description(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        description = _parse_description(d.pop("description", UNSET))

        _routes = d.pop("routes", UNSET)
        routes: list[HttpRouteSpec] | Unset = UNSET
        if _routes is not UNSET:
            routes = []
            for routes_item_data in _routes:
                routes_item = HttpRouteSpec.from_dict(routes_item_data)

                routes.append(routes_item)

        suppressed_response_status = d.pop("suppressed_response_status", UNSET)

        http_server_spec = cls(
            name=name,
            base_url=base_url,
            description=description,
            routes=routes,
            suppressed_response_status=suppressed_response_status,
        )

        return http_server_spec
