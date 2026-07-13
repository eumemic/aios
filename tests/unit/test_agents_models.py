"""Pydantic validation for agent request models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aios.models.agents import AgentCreate, AgentUpdate, HttpRouteSpec, McpServerSpec


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


class TestAgentCreateDuplicateIngressInvariants:
    """#1758 scope (1): duplicate ``_tool_key`` / mcp-server-name / configs[]-name
    entries are API-reachable and break the attenuation module's own normal-form
    contract (``attenuate(x, x) == canonicalize(x)``) — see
    ``models.attenuation._tool_key`` and the two soft spots documented on issue
    #1758. Reject at the ingress boundary (mirroring the existing ``base_url``
    dedup) rather than let the un-enforced invariant erode the meet's laws.
    """

    def test_rejects_duplicate_builtin_tool_type(self) -> None:
        with pytest.raises(ValidationError, match=r"duplicate tool entry 'bash'"):
            AgentCreate.model_validate(
                {
                    "name": "agent",
                    "model": "gpt-4",
                    "tools": [
                        {"type": "bash", "permission": "always_allow"},
                        {"type": "bash", "permission": "always_ask"},
                    ],
                }
            )

    def test_rejects_duplicate_custom_tool_name(self) -> None:
        custom = {
            "type": "custom",
            "name": "fetch",
            "description": "d",
            "input_schema": {"type": "object"},
        }
        with pytest.raises(ValidationError, match=r"duplicate tool entry 'fetch'"):
            AgentCreate.model_validate(
                {"name": "agent", "model": "gpt-4", "tools": [custom, custom]}
            )

    def test_rejects_duplicate_mcp_toolset_server(self) -> None:
        toolset = {"type": "mcp_toolset", "mcp_server_name": "gh"}
        with pytest.raises(ValidationError, match=r"duplicate tool entry 'gh'"):
            AgentCreate.model_validate(
                {"name": "agent", "model": "gpt-4", "tools": [toolset, toolset]}
            )

    def test_mcp_server_rejects_loopback_url(self) -> None:
        with pytest.raises(ValidationError, match="private or runtime-local host"):
            McpServerSpec(name="local", url="http://127.0.0.1:8080/mcp")

    def test_mcp_server_rejects_runtime_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AIOS_URL", "https://runtime.example/v1")
        with pytest.raises(ValidationError, match="private or runtime-local host"):
            McpServerSpec(name="self", url="https://runtime.example/mcp")

    def test_mcp_server_private_host_can_be_explicitly_allowlisted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIOS_TARGET_URL_ALLOW_HOSTS", "127.0.0.1")
        assert McpServerSpec(name="dev", url="http://127.0.0.1:9000/mcp").url.endswith("/mcp")

    def test_rejects_duplicate_mcp_server_name(self) -> None:
        with pytest.raises(ValidationError, match=r"duplicate mcp server name 'gh'"):
            AgentCreate.model_validate(
                {
                    "name": "agent",
                    "model": "gpt-4",
                    "mcp_servers": [
                        {"name": "gh", "url": "https://gh1"},
                        {"name": "gh", "url": "https://gh2"},
                    ],
                }
            )

    def test_rejects_duplicate_configs_name_within_toolset(self) -> None:
        with pytest.raises(ValidationError, match=r"duplicate configs\[\] entry 'create_issue'"):
            AgentCreate.model_validate(
                {
                    "name": "agent",
                    "model": "gpt-4",
                    "tools": [
                        {
                            "type": "mcp_toolset",
                            "mcp_server_name": "gh",
                            "configs": [
                                {"name": "create_issue", "enabled": True},
                                {"name": "create_issue", "enabled": False},
                            ],
                        }
                    ],
                }
            )

    def test_accepts_distinct_keys_on_every_dimension(self) -> None:
        agent = AgentCreate.model_validate(
            {
                "name": "agent",
                "model": "gpt-4",
                "tools": [
                    {"type": "bash"},
                    {"type": "read"},
                    {
                        "type": "custom",
                        "name": "fetch",
                        "description": "d",
                        "input_schema": {"type": "object"},
                    },
                    {"type": "mcp_toolset", "mcp_server_name": "gh"},
                ],
                "mcp_servers": [{"name": "gh", "url": "https://gh"}],
            }
        )
        assert len(agent.tools) == 4
        assert len(agent.mcp_servers) == 1

    def test_agent_update_rejects_duplicate_mcp_server_name(self) -> None:
        with pytest.raises(ValidationError, match=r"duplicate mcp server name 'gh'"):
            AgentUpdate.model_validate(
                {
                    "version": 1,
                    "mcp_servers": [
                        {"name": "gh", "url": "https://gh1"},
                        {"name": "gh", "url": "https://gh2"},
                    ],
                }
            )

    def test_agent_update_rejects_duplicate_tool_key(self) -> None:
        with pytest.raises(ValidationError, match=r"duplicate tool entry 'bash'"):
            AgentUpdate.model_validate(
                {"version": 1, "tools": [{"type": "bash"}, {"type": "bash"}]}
            )


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
