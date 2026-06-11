"""The capability-attenuation operator — pure, table-driven unit tests.

Pure in-memory: no Postgres, no Docker. Covers the meet per dimension, the MCP
toolset normal form (incl. the resolver short-circuits and the fall-through pin),
the joint ``(name, url)`` MCP key, the algebraic contract that makes the author-edge
predicate exact (``attenuate(x, x) == canonicalize(x)`` + idempotence), and the
``litellm_extra`` tripwire guarding the model-identity axis (#823).
"""

from __future__ import annotations

import json

import pytest

from aios.models.agents import (
    HttpPermissionPolicy,
    HttpRouteSpec,
    HttpServerSpec,
    McpPermissionPolicy,
    McpServerSpec,
    McpToolConfig,
    McpToolsetConfig,
    PermissionPolicy,
    ToolSpec,
    ToolTransport,
    resolve_mcp_enabled,
    resolve_mcp_permission,
)
from aios.models.attenuation import Surface, attenuate, canonicalize, surface_diff

# Controlled defaults so the meet-logic tests are independent of the live registry.
DMP: PermissionPolicy = "always_ask"
BT: dict[str, ToolTransport] = {
    "bash": "both",
    "read": "both",
    "write": "agent_tool",
    "edit": "cli",
    "web_fetch": "both",
}


def att(declared: Surface, launcher: Surface, *, dmp: PermissionPolicy = DMP) -> Surface:
    return attenuate(declared, launcher, default_mcp_permission=dmp, builtin_transports=BT)


def canon(s: Surface, *, dmp: PermissionPolicy = DMP) -> Surface:
    return canonicalize(s, default_mcp_permission=dmp, builtin_transports=BT)


def _toolset(server: str, **kw: object) -> ToolSpec:
    return ToolSpec(type="mcp_toolset", mcp_server_name=server, **kw)


def _mcp_cfg(name: str, *, perm: PermissionPolicy | None = None, **kw: object) -> McpToolConfig:
    pp = McpPermissionPolicy(type=perm) if perm is not None else None
    return McpToolConfig(name=name, permission_policy=pp, **kw)


# ── builtin / custom tools ────────────────────────────────────────────────────


class TestBuiltinMeet:
    def test_permission_meet_takes_stricter(self) -> None:
        declared = Surface([ToolSpec(type="bash", permission="always_ask")], [], [])
        launcher = Surface([ToolSpec(type="bash", permission="always_allow")], [], [])
        out = att(declared, launcher)
        assert [t.permission for t in out.tools] == ["always_ask"]
        # declared narrows → it is a fixpoint → predicate admits it.
        assert out == canon(declared)

    def test_widening_permission_is_rejected_by_predicate(self) -> None:
        declared = Surface([ToolSpec(type="bash", permission="always_allow")], [], [])
        launcher = Surface([ToolSpec(type="bash", permission="always_ask")], [], [])
        out = att(declared, launcher)
        assert [t.permission for t in out.tools] == ["always_ask"]  # clamped down
        assert out != canon(declared)  # not a fixpoint → ForbiddenError at the author edge
        assert surface_diff(canon(declared), out) == {"tools": ["bash"]}

    def test_none_permission_resolves_to_always_allow(self) -> None:
        declared = Surface([ToolSpec(type="bash")], [], [])
        launcher = Surface([ToolSpec(type="bash")], [], [])
        assert canon(declared).tools[0].permission == "always_allow"
        assert att(declared, launcher) == canon(declared)

    def test_tool_absent_from_launcher_is_dropped(self) -> None:
        declared = Surface([ToolSpec(type="bash"), ToolSpec(type="read")], [], [])
        launcher = Surface([ToolSpec(type="bash")], [], [])
        out = att(declared, launcher)
        assert [t.type for t in out.tools] == ["bash"]

    def test_transport_glb_both_narrows_to_launcher(self) -> None:
        declared = Surface([ToolSpec(type="bash", transport="both")], [], [])
        launcher = Surface([ToolSpec(type="bash", transport="agent_tool")], [], [])
        assert att(declared, launcher).tools[0].transport == "agent_tool"

    def test_disjoint_transport_drops_the_tool(self) -> None:
        declared = Surface([ToolSpec(type="bash", transport="cli")], [], [])
        launcher = Surface([ToolSpec(type="bash", transport="agent_tool")], [], [])
        assert att(declared, launcher).tools == []

    def test_disabled_tool_is_pruned_both_sides(self) -> None:
        declared = Surface([ToolSpec(type="bash", enabled=False)], [], [])
        assert canon(declared).tools == []
        # A launcher-disabled tool grants nothing — declared can't enable it.
        out = att(Surface([ToolSpec(type="bash")], [], []), declared)
        assert out.tools == []

    def test_custom_tool_keyed_by_name(self) -> None:
        spec = dict(name="foo", description="d", input_schema={"type": "object"})
        declared = Surface([ToolSpec(type="custom", **spec)], [], [])
        launcher = Surface([ToolSpec(type="custom", **spec)], [], [])
        out = att(declared, launcher)
        assert [t.name for t in out.tools] == ["foo"]
        assert out == canon(declared)


# ── MCP + HTTP servers ────────────────────────────────────────────────────────


class TestServerSurvival:
    def test_mcp_server_survives_on_joint_name_url(self) -> None:
        srv = McpServerSpec(name="gh", url="https://gh/mcp")
        declared = Surface([_toolset("gh")], [srv], [])
        launcher = Surface([_toolset("gh")], [srv], [])
        out = att(declared, launcher)
        assert [(s.name, s.url) for s in out.mcp_servers] == [("gh", "https://gh/mcp")]
        assert any(t.type == "mcp_toolset" for t in out.tools)

    def test_mcp_server_repointed_url_is_dropped(self) -> None:
        # Launcher holds (gh, U1) and (other, U2); child re-points gh → U2.
        u1, u2 = "https://gh/mcp", "https://other/mcp"
        launcher = Surface(
            [_toolset("gh"), _toolset("other")],
            [McpServerSpec(name="gh", url=u1), McpServerSpec(name="other", url=u2)],
            [],
        )
        declared = Surface([_toolset("gh")], [McpServerSpec(name="gh", url=u2)], [])
        out = att(declared, launcher)
        assert out.mcp_servers == []  # (gh, U2) is not a launcher key
        assert out.tools == []  # the toolset's server didn't survive
        assert out != canon(declared)  # author edge rejects

    def test_mcp_server_emitted_launcher_verbatim(self) -> None:
        # Launcher carries headers; child declares the same (name, url) without them.
        l_srv = McpServerSpec(name="gh", url="https://gh/mcp", headers={"X-Toolsets": "issues"})
        d_srv = McpServerSpec(name="gh", url="https://gh/mcp")
        out = att(Surface([], [d_srv], []), Surface([], [l_srv], []))
        assert out.mcp_servers[0].headers == {"X-Toolsets": "issues"}  # launcher wins
        assert out != canon(Surface([], [d_srv], []))  # child must declare it identically

    def test_http_routes_parent_wins_frozen(self) -> None:
        # First-match-wins ordering: a child re-declaring the broad route must NOT be
        # able to escape the launcher's earlier always_ask gate on /reports/admin.
        l_routes = [
            HttpRouteSpec(
                path_pattern="/reports/admin",
                permission_policy=HttpPermissionPolicy(type="always_ask"),
            ),
            HttpRouteSpec(
                path_pattern="/reports/**",
                permission_policy=HttpPermissionPolicy(type="always_allow"),
            ),
        ]
        l_srv = HttpServerSpec(name="api", base_url="https://api", routes=l_routes)
        d_srv = HttpServerSpec(
            name="api",
            base_url="https://api",
            routes=[
                HttpRouteSpec(
                    path_pattern="/reports/**",
                    permission_policy=HttpPermissionPolicy(type="always_allow"),
                )
            ],
        )
        out = att(Surface([], [], [d_srv]), Surface([], [], [l_srv]))
        assert out.http_servers[0].routes == l_routes  # launcher-verbatim, both routes
        assert out != canon(Surface([], [], [d_srv]))  # narrowing is rejected (R1: verbatim only)

    def test_http_server_absent_is_dropped(self) -> None:
        d_srv = HttpServerSpec(name="api", base_url="https://api")
        out = att(Surface([], [], [d_srv]), Surface([], [], []))
        assert out.http_servers == []


class TestSurfaceDiffHttpIdentity:
    """#939: ``surface_diff``'s http section is keyed on identity ``(name, base_url)``,
    not full-spec equality — so a route/field divergence on an identically-identified
    server is NOT flagged (the authoring gate inherits the launcher's frozen routes).
    Contrast tools/mcp, which stay full-equality keyed checks.
    """

    def test_surface_diff_http_ignores_route_divergence_when_identity_matches(self) -> None:
        expected = canon(
            Surface(
                [],
                [],
                [
                    HttpServerSpec(
                        name="api",
                        base_url="https://api",
                        routes=[
                            HttpRouteSpec(
                                path_pattern="/v1/**",
                                permission_policy=HttpPermissionPolicy(type="always_ask"),
                            )
                        ],
                    )
                ],
            )
        )
        # Same (name, base_url) identity, but routes diverge (here: empty).
        actual = canon(
            Surface([], [], [HttpServerSpec(name="api", base_url="https://api", routes=[])])
        )
        assert surface_diff(expected, actual) == {}

    def test_surface_diff_http_flags_absent_base_url(self) -> None:
        expected = canon(Surface([], [], [HttpServerSpec(name="api", base_url="https://api")]))
        actual = canon(Surface([], [], []))
        assert surface_diff(expected, actual) == {"http_servers": ["api"]}

    def test_surface_diff_http_flags_renamed_server_same_base_url(self) -> None:
        expected = canon(Surface([], [], [HttpServerSpec(name="api", base_url="https://api")]))
        actual = canon(Surface([], [], [HttpServerSpec(name="api2", base_url="https://api")]))
        assert surface_diff(expected, actual) == {"http_servers": ["api"]}


# ── MCP toolset normal form ───────────────────────────────────────────────────


class TestToolsetNormalForm:
    def test_dangling_toolset_is_pruned(self) -> None:
        # No matching McpServerSpec → no discovery → drop.
        declared = Surface([_toolset("ghost")], [], [])
        assert canon(declared).tools == []

    def test_default_config_frozen_concrete(self) -> None:
        srv = McpServerSpec(name="s", url="https://s")
        out = canon(Surface([_toolset("s")], [srv], []), dmp="always_allow")
        ts = out.tools[0]
        assert ts.default_config is not None
        assert ts.default_config.permission_policy == McpPermissionPolicy(type="always_allow")
        assert ts.default_config.transport == "both"
        assert ts.configs is None  # no per-tool overrides → minimal form

    def test_config_none_policy_short_circuits_to_operator_default(self) -> None:
        # A configs[] entry with permission_policy=None resolves to the OPERATOR default,
        # NOT default_config — even when default_config says always_allow.
        srv = McpServerSpec(name="s", url="https://s")
        ts = _toolset(
            "s",
            default_config=McpToolsetConfig(
                permission_policy=McpPermissionPolicy(type="always_allow")
            ),
            configs=[_mcp_cfg("danger")],  # permission_policy=None
        )
        out = canon(Surface([ts], [srv], []), dmp="always_ask")
        assert (
            resolve_mcp_permission("mcp__s__danger", out.tools) == "always_ask"
        )  # operator default
        assert (
            resolve_mcp_permission("mcp__s__other", out.tools) == "always_allow"
        )  # default_config

    def test_fall_through_pins_launcher_only_name(self) -> None:
        # Regression (b): launcher pins delete_repo=always_ask; operator default
        # always_allow; child declares the toolset bare → the meet must PIN delete_repo.
        srv = McpServerSpec(name="s", url="https://s")
        launcher = Surface(
            [_toolset("s", configs=[_mcp_cfg("delete_repo", perm="always_ask")])], [srv], []
        )
        declared = Surface([_toolset("s")], [srv], [])
        out = att(declared, launcher, dmp="always_allow")
        assert resolve_mcp_permission("mcp__s__delete_repo", out.tools) == "always_ask"
        assert resolve_mcp_permission("mcp__s__read_repo", out.tools) == "always_allow"

    def test_per_name_transport_conflict_disables(self) -> None:
        srv = McpServerSpec(name="s", url="https://s")
        launcher = Surface([_toolset("s", configs=[_mcp_cfg("x", transport="cli")])], [srv], [])
        declared = Surface(
            [_toolset("s", configs=[_mcp_cfg("x", transport="agent_tool")])], [srv], []
        )
        out = att(declared, launcher)
        assert resolve_mcp_enabled("mcp__s__x", out.tools) is False  # disjoint transport → disabled

    def test_disabling_a_tool_is_not_a_widening(self) -> None:
        # A declaration that merely *disables* a tool can never exceed the launcher,
        # even when the launcher holds it at a narrower transport. Disabled entries
        # collapse to one bottom, so their dead fields don't fail the predicate.
        srv = McpServerSpec(name="s", url="https://s")
        launcher = Surface(
            [_toolset("s", default_config=McpToolsetConfig(transport="cli"))], [srv], []
        )
        declared = Surface(
            [
                _toolset(
                    "s",
                    default_config=McpToolsetConfig(transport="cli"),
                    configs=[_mcp_cfg("t2", enabled=False, perm="always_allow", transport="both")],
                )
            ],
            [srv],
            [],
        )
        assert att(declared, launcher) == canon(declared)  # fixpoint → author edge admits

    def test_toolset_permission_default_widening_rejected(self) -> None:
        # Child default always_allow vs launcher default always_ask → child widens → reject.
        srv = McpServerSpec(name="s", url="https://s")
        launcher = Surface(
            [
                _toolset(
                    "s",
                    default_config=McpToolsetConfig(
                        permission_policy=McpPermissionPolicy(type="always_ask")
                    ),
                )
            ],
            [srv],
            [],
        )
        declared = Surface(
            [
                _toolset(
                    "s",
                    default_config=McpToolsetConfig(
                        permission_policy=McpPermissionPolicy(type="always_allow")
                    ),
                )
            ],
            [srv],
            [],
        )
        out = att(declared, launcher)
        assert out != canon(declared)


# ── the algebraic contract (the predicate's exactness) ────────────────────────


def _gnarly_surfaces() -> list[Surface]:
    s1 = McpServerSpec(name="s1", url="https://s1", headers={"H": "1"})
    s2 = McpServerSpec(name="s2", url="https://s2")
    http = HttpServerSpec(
        name="api",
        base_url="https://api",
        routes=[
            HttpRouteSpec(
                path_pattern="/a/**", permission_policy=HttpPermissionPolicy(type="always_ask")
            )
        ],
    )
    return [
        Surface([], [], []),
        Surface([ToolSpec(type="bash"), ToolSpec(type="read", permission="always_ask")], [], []),
        Surface(
            [
                _toolset(
                    "s1",
                    default_config=McpToolsetConfig(transport="both"),
                    configs=[_mcp_cfg("write", perm="always_ask")],
                ),
                _toolset("s2"),
                ToolSpec(type="custom", name="c", description="d", input_schema={}),
            ],
            [s1, s2],
            [http],
        ),
        # Disabled entries with *divergent* dead fields — the case that broke the
        # normal-form contract before disabled triples were collapsed to one bottom.
        Surface(
            [
                _toolset(
                    "s1",
                    default_config=McpToolsetConfig(enabled=False, transport="both"),
                    configs=[
                        _mcp_cfg("a", enabled=False, perm="always_allow", transport="both"),
                        _mcp_cfg("b", perm="always_ask", transport="cli"),
                    ],
                ),
                _toolset("s2", configs=[_mcp_cfg("x", enabled=False, transport="agent_tool")]),
            ],
            [s1, s2],
            [],
        ),
    ]


@pytest.mark.parametrize("s", _gnarly_surfaces())
def test_self_meet_equals_canonicalize(s: Surface) -> None:
    # The contract that makes the author-edge predicate exact.
    assert att(s, s) == canon(s)


@pytest.mark.parametrize("s", _gnarly_surfaces())
def test_canonicalize_is_idempotent(s: Surface) -> None:
    assert canon(canon(s)) == canon(s)


@pytest.mark.parametrize("d", _gnarly_surfaces())
@pytest.mark.parametrize("ln", _gnarly_surfaces())
def test_attenuate_is_idempotent_against_launcher(d: Surface, ln: Surface) -> None:
    once = att(d, ln)
    assert att(once, ln) == once  # already clamped → no further narrowing


@pytest.mark.parametrize("d", _gnarly_surfaces())
@pytest.mark.parametrize("ln", _gnarly_surfaces())
def test_meet_never_exceeds_either_operand(d: Surface, ln: Surface) -> None:
    # Absorption: the meet is a fixpoint against each operand (≤ both).
    out = att(d, ln)
    assert att(out, ln) == out
    assert att(out, d) == out


# ── the model-identity tripwire (g) ───────────────────────────────────────────


def test_no_registered_tool_exposes_litellm_extra() -> None:
    """No agent_tool may set ``litellm_extra`` (hence ``api_base``) — #823's precondition.

    The surface+credential meet does NOT clamp model routing; it is sound only while no
    runtime principal can mint/edit an agent's ``litellm_extra``. The day a
    create_agent-style builtin is added, its schema exposes the agent field and this
    test fails — forcing the model-identity clamp (#823) to land with it.
    """
    import aios.tools  # noqa: F401  — populate the registry
    from aios.tools.registry import registry

    for name in registry.names():
        schema = json.dumps(registry.get(name).parameters_schema)
        assert "litellm_extra" not in schema, f"tool {name!r} exposes litellm_extra (see #823)"
