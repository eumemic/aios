"""Route-aware http-server domination used by the agent-update delta gate (#1961)."""

from __future__ import annotations

from aios.models.agents import HttpPermissionPolicy, HttpRouteSpec, HttpServerSpec
from aios.models.attenuation import http_server_dominated

_BASE = "https://api.github.com"


def _srv(*routes: HttpRouteSpec, name: str = "github") -> HttpServerSpec:
    return HttpServerSpec(name=name, base_url=_BASE, routes=list(routes))


_PRIOR = _srv(
    HttpRouteSpec(path_pattern="/repos/**", methods=["GET", "POST"]),
    HttpRouteSpec(
        path_pattern="/graphql",
        methods=["POST"],
        permission_policy=HttpPermissionPolicy(type="always_ask"),
    ),
)


def test_identical_server_is_dominated() -> None:
    assert http_server_dominated(_PRIOR.model_copy(deep=True), _PRIOR)


def test_reordered_methods_are_semantically_equal() -> None:
    declared = _srv(
        HttpRouteSpec(path_pattern="/repos/**", methods=["POST", "GET", "GET"]),
        _PRIOR.routes[1],
    )
    assert http_server_dominated(declared, _PRIOR)


def test_narrowing_methods_is_dominated() -> None:
    declared = _srv(HttpRouteSpec(path_pattern="/repos/**", methods=["GET"]), _PRIOR.routes[1])
    assert http_server_dominated(declared, _PRIOR)


def test_added_route_is_not_dominated() -> None:
    declared = _srv(*_PRIOR.routes, HttpRouteSpec(path_pattern="/admin/**"))
    assert not http_server_dominated(declared, _PRIOR)


def test_widened_methods_are_not_dominated() -> None:
    declared = _srv(HttpRouteSpec(path_pattern="/repos/**"), _PRIOR.routes[1])
    assert not http_server_dominated(declared, _PRIOR)


def test_relaxed_permission_gate_is_not_dominated() -> None:
    declared = _srv(
        _PRIOR.routes[0],
        HttpRouteSpec(
            path_pattern="/graphql",
            methods=["POST"],
            permission_policy=HttpPermissionPolicy(type="always_allow"),
        ),
    )
    assert not http_server_dominated(declared, _PRIOR)


def test_renamed_server_is_not_dominated() -> None:
    assert not http_server_dominated(_srv(*_PRIOR.routes, name="other"), _PRIOR)


def test_route_description_change_is_dominated() -> None:
    """Route ``description`` is prose, not authority: editing it is not a widening."""
    declared = _srv(
        _PRIOR.routes[0].model_copy(update={"description": "Repos: issues, PRs, labels."}),
        _PRIOR.routes[1].model_copy(update={"description": "GraphQL mutations."}),
    )
    assert http_server_dominated(declared, _PRIOR)


def test_route_description_change_does_not_mask_widening() -> None:
    declared = _srv(
        HttpRouteSpec(path_pattern="/repos/**", description="docs only"),
        _PRIOR.routes[1],
    )
    assert not http_server_dominated(declared, _PRIOR)
