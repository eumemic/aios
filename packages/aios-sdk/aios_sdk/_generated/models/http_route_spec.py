from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define

from ..models.http_route_spec_methods_type_0_item import HttpRouteSpecMethodsType0Item
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
    gates *invocation*: ``always_ask`` leaves the call unresolved in the
    event log until the client confirms via
    ``POST /sessions/:id/tool-confirmations``.

    ``methods`` scopes the route to a set of HTTP verbs so a surface can
    express read/write attenuation structurally — e.g. ``GET`` everywhere
    but ``POST`` only on a sandbox path (#828).  ``None`` (the default)
    means *all* methods are allowed (the method-dimension lattice top;
    backward-compatible with routes authored before method scoping).  A
    non-empty list restricts the route to exactly those verbs.  An empty
    list (``[]``) is *deny-all* — the method-dimension lattice bottom, and
    the natural result of intersecting two disjoint method sets during
    attenuation; it matches nothing.  The capability meet
    (:mod:`aios.models.attenuation`) intersects ``methods`` per route, so a
    child surface can narrow a parent route's verbs but never widen them.

    GraphQL caveat — **REST-only discipline for attenuated surfaces.**
    Method scoping confines REST read/write because the verb encodes the
    semantics.  A GraphQL endpoint serves both queries (reads) and
    mutations (writes) over a single ``POST`` path, so method scoping
    *cannot* separate read from write there, and the broker does not
    inspect request bodies.  An operator who needs to confine writes on a
    GraphQL surface must place reads and writes behind distinct
    ``base_url`` servers (each with its own credential/route allowlist) or
    accept that granting ``POST`` grants both.

    ``allow_query`` opts the route into permitting a query string on the
    request ``path``.  The default is ``False``: a ``?...`` is rejected at
    the route gate (#485) because ``httpx`` parses the query off the URL
    and an unanticipating allowlist — e.g. a read-only ``/lights/*`` — is
    bypassed when the upstream interprets ``?action=delete`` as a write.
    An operator sets ``allow_query=True`` only on routes where the query
    string cannot escalate beyond what the route already grants and where
    it is functionally required — e.g. a GitHub ``/repos/**`` route that
    already permits every verb and must follow cursor/``page`` pagination
    to read a full comment thread (#1156).  The path portion is still
    glob-matched against ``path_pattern`` (the query is stripped before
    the match), and ``.``/``..`` dot-segment rejection still applies, so a
    query allowance never widens the path-dimension grant.  It is
    launcher-verbatim under attenuation: a child surface cannot turn it on
    where the parent left it off.

        Attributes:
            path_pattern (str):
            description (None | str | Unset):
            enabled (bool | Unset):  Default: True.
            permission_policy (HttpPermissionPolicy | None | Unset):
            methods (list[HttpRouteSpecMethodsType0Item] | None | Unset):
            allow_query (bool | Unset):  Default: False.
    """

    path_pattern: str
    description: None | str | Unset = UNSET
    enabled: bool | Unset = True
    permission_policy: HttpPermissionPolicy | None | Unset = UNSET
    methods: list[HttpRouteSpecMethodsType0Item] | None | Unset = UNSET
    allow_query: bool | Unset = False

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

        methods: list[str] | None | Unset
        if isinstance(self.methods, Unset):
            methods = UNSET
        elif isinstance(self.methods, list):
            methods = []
            for methods_type_0_item_data in self.methods:
                methods_type_0_item = methods_type_0_item_data.value
                methods.append(methods_type_0_item)

        else:
            methods = self.methods

        allow_query = self.allow_query

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
        if methods is not UNSET:
            field_dict["methods"] = methods
        if allow_query is not UNSET:
            field_dict["allow_query"] = allow_query

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

        def _parse_methods(
            data: object,
        ) -> list[HttpRouteSpecMethodsType0Item] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                methods_type_0 = []
                _methods_type_0 = data
                for methods_type_0_item_data in _methods_type_0:
                    methods_type_0_item = HttpRouteSpecMethodsType0Item(
                        methods_type_0_item_data
                    )

                    methods_type_0.append(methods_type_0_item)

                return methods_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[HttpRouteSpecMethodsType0Item] | None | Unset, data)

        methods = _parse_methods(d.pop("methods", UNSET))

        allow_query = d.pop("allow_query", UNSET)

        http_route_spec = cls(
            path_pattern=path_pattern,
            description=description,
            enabled=enabled,
            permission_policy=permission_policy,
            methods=methods,
            allow_query=allow_query,
        )

        return http_route_spec
