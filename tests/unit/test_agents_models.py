"""Pydantic validation for agent request models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aios.models.agents import AgentCreate, AgentUpdate, HttpRouteSpec


def _http_server(name: str, base_url: str) -> dict[str, str]:
    return {"name": name, "base_url": base_url}


class TestHttpRouteSpecMethods:
    def test_methods_default_none_means_all(self) -> None:
        route = HttpRouteSpec(path_pattern="/x")
        assert route.methods is None

    def test_methods_accepts_explicit_list(self) -> None:
        route = HttpRouteSpec(path_pattern="/x", methods=["GET", "POST"])
        assert route.methods == ["GET", "POST"]

    def test_methods_empty_list_is_deny_all(self) -> None:
        # [] = lattice bottom (deny-all); a legitimate, if pointless, author choice
        # and the natural empty-intersection result of the method meet.
        route = HttpRouteSpec(path_pattern="/x", methods=[])
        assert route.methods == []

    def test_methods_rejects_unknown_verb(self) -> None:
        with pytest.raises(ValidationError):
            HttpRouteSpec.model_validate({"path_pattern": "/x", "methods": ["FETCH"]})


class TestAgentCreateHttpServers:
    def test_rejects_duplicate_base_url_with_different_names(self) -> None:
        with pytest.raises(
            ValidationError, match=r"duplicate base_url 'https://api\.example\.com'"
        ):
            AgentCreate.model_validate(
                {
                    "name": "agent",
                    "model": "gpt-4",
                    "http_servers": [
                        _http_server("primary", "https://api.example.com"),
                        _http_server("secondary", "https://api.example.com"),
                    ],
                }
            )

    def test_accepts_duplicate_names_with_distinct_base_urls(self) -> None:
        agent = AgentCreate.model_validate(
            {
                "name": "agent",
                "model": "gpt-4",
                "http_servers": [
                    _http_server("api", "https://one.example.com"),
                    _http_server("api", "https://two.example.com"),
                ],
            }
        )

        assert [server.base_url for server in agent.http_servers] == [
            "https://one.example.com",
            "https://two.example.com",
        ]


class TestAgentUpdateHttpServers:
    def test_rejects_duplicate_base_url_with_different_names(self) -> None:
        with pytest.raises(
            ValidationError, match=r"duplicate base_url 'https://api\.example\.com'"
        ):
            AgentUpdate.model_validate(
                {
                    "version": 1,
                    "http_servers": [
                        _http_server("primary", "https://api.example.com"),
                        _http_server("secondary", "https://api.example.com"),
                    ],
                }
            )

    def test_accepts_duplicate_names_with_distinct_base_urls(self) -> None:
        update = AgentUpdate.model_validate(
            {
                "version": 1,
                "http_servers": [
                    _http_server("api", "https://one.example.com"),
                    _http_server("api", "https://two.example.com"),
                ],
            }
        )

        assert update.http_servers is not None
        assert [server.base_url for server in update.http_servers] == [
            "https://one.example.com",
            "https://two.example.com",
        ]

    def test_accepts_omitted_http_servers(self) -> None:
        update = AgentUpdate.model_validate({"version": 1})

        assert update.http_servers is None


class TestResolveHttpServerRefs:
    """#953: names-only ``http_servers`` resolution against an acting agent's servers.

    A bare name resolves to an empty-routes identity spec at the agent's ``base_url``
    (the existing #949 identity-match path then inherits the agent's frozen routes); an
    unknown name raises; full ``HttpServerSpec`` entries pass through verbatim.
    """

    def test_bare_name_resolves_to_agent_base_url_empty_routes(self) -> None:
        from aios.models.agents import HttpServerSpec, resolve_http_server_refs

        agent = [
            HttpServerSpec(
                name="davenant",
                base_url="https://davenant.example",
                routes=[HttpRouteSpec(path_pattern="/v1/**")],
            )
        ]
        out = resolve_http_server_refs(["davenant"], agent)
        assert out == [
            HttpServerSpec(name="davenant", base_url="https://davenant.example", routes=[])
        ]

    def test_unknown_name_raises(self) -> None:
        from aios.models.agents import HttpServerSpec, resolve_http_server_refs

        agent = [HttpServerSpec(name="davenant", base_url="https://davenant.example")]
        with pytest.raises(ValueError, match=r"references 'ghost', which the acting agent"):
            resolve_http_server_refs(["ghost"], agent)

    def test_full_spec_passes_through_verbatim(self) -> None:
        from aios.models.agents import HttpServerSpec, resolve_http_server_refs

        spec = HttpServerSpec(name="api", base_url="https://api", routes=[])
        out = resolve_http_server_refs([spec], [spec])
        assert out == [spec]

    def test_mixed_names_and_specs(self) -> None:
        from aios.models.agents import HttpServerSpec, resolve_http_server_refs

        agent = [
            HttpServerSpec(name="davenant", base_url="https://davenant.example"),
            HttpServerSpec(name="api", base_url="https://api"),
        ]
        spec = HttpServerSpec(name="api", base_url="https://api", routes=[])
        out = resolve_http_server_refs(["davenant", spec], agent)
        assert out == [
            HttpServerSpec(name="davenant", base_url="https://davenant.example", routes=[]),
            spec,
        ]
