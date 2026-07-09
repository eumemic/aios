"""Hand-built ``hypothesis`` strategies over :class:`aios.models.attenuation.Surface`
(#1758).

Pydantic v2 has no first-party hypothesis plugin, so these strategies are built by
hand from small, SHARED pools of identity keys (``_tool_key`` components, MCP
``(name, url)``, http ``base_url``, route ``path_pattern``) — the meet is a *join* on
these keys, so drawing them from a small shared pool (rather than unique-per-draw
strings) is what makes generated pairs actually collide often enough to exercise the
meet's key-match paths instead of only its "absent from launcher → drop" path.

Every generated ``ToolSpec``/``McpServerSpec``/``HttpServerSpec`` list satisfies the
live ingress validators (:func:`aios.models.agents.validate_tools`,
:func:`validate_mcp_servers`, :func:`validate_http_servers`, plus the model-level
``configs[]``/custom-tool-name/``mcp__`` checks) — generating input the API would
itself reject would test an invariant the operator does not need to hold under.

Path patterns and URLs are opaque identity keys to the attenuation operator (it never
glob-matches or parses them) — no glob/URL semantics are generated, just small pools
of distinct strings.
"""

from __future__ import annotations

from hypothesis import strategies as st

from aios.models.agents import (
    HttpMethod,
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
)

# ── small shared pools (identity keys) ─────────────────────────────────────────
#
# Small on purpose: the meet is dimension-wise and keyed, so a pool of 2-3 values
# per dimension is enough for generated pairs to collide on a key often (making the
# launcher-verbatim-survival / narrowing paths non-vacuous) while still leaving
# "absent from the other side" as a frequent, cheaply-drawn case.

BUILTIN_TYPES = ["bash", "read", "write"]  # BT below gives each a distinct transport
CUSTOM_NAMES = ["cust_a", "cust_b"]
MCP_SERVER_NAMES = ["srv_a", "srv_b"]
MCP_URLS = ["https://mcp-a", "https://mcp-b"]
MCP_TOOL_NAMES = ["t1", "t2"]  # discovered-tool names for configs[]
HTTP_SERVER_NAMES = ["api_a", "api_b"]
HTTP_BASE_URLS = ["https://http-a", "https://http-b"]
PATH_PATTERNS = ["/x", "/y"]  # opaque keys — no glob semantics generated

PERMS: list[PermissionPolicy] = ["always_allow", "always_ask"]
TRANSPORTS: list[ToolTransport] = ["cli", "agent_tool", "both"]
METHOD_POOL: list[HttpMethod] = ["GET", "POST", "DELETE"]

# The operator-default / builtin-transport-registry inputs used by every test in this
# module — controlled constants, matching test_attenuation.py's DMP/BT convention, so
# the law tests are independent of the live settings/registry.
DMP: PermissionPolicy = "always_ask"
BT: dict[str, ToolTransport] = {"bash": "both", "read": "both", "write": "agent_tool"}


def _perm() -> st.SearchStrategy[PermissionPolicy]:
    return st.sampled_from(PERMS)


def _mpp() -> st.SearchStrategy[McpPermissionPolicy | None]:
    return st.none() | st.builds(McpPermissionPolicy, type=_perm())


def _hpp() -> st.SearchStrategy[HttpPermissionPolicy | None]:
    return st.none() | st.builds(HttpPermissionPolicy, type=_perm())


def _transport() -> st.SearchStrategy[ToolTransport | None]:
    return st.none() | st.sampled_from(TRANSPORTS)


# ── tools ────────────────────────────────────────────────────────────────────


def _builtin_tool() -> st.SearchStrategy[ToolSpec]:
    return st.builds(
        ToolSpec,
        type=st.sampled_from(BUILTIN_TYPES),
        enabled=st.booleans(),
        permission=st.none() | _perm(),
        transport=_transport(),
    )


def _custom_tool() -> st.SearchStrategy[ToolSpec]:
    return st.builds(
        ToolSpec,
        type=st.just("custom"),
        name=st.sampled_from(CUSTOM_NAMES),
        description=st.just("d"),
        input_schema=st.just({"type": "object"}),
        enabled=st.booleans(),
        permission=st.none() | _perm(),
        transport=_transport(),
    )


def _mcp_tool_config() -> st.SearchStrategy[McpToolConfig]:
    return st.builds(
        McpToolConfig,
        name=st.sampled_from(MCP_TOOL_NAMES),
        enabled=st.booleans(),
        permission_policy=_mpp(),
        transport=_transport(),
        read_allow=st.booleans(),
    )


def _mcp_configs_list() -> st.SearchStrategy[list[McpToolConfig] | None]:
    """A ``configs[]`` list with UNIQUE names (the ingress invariant) — dedup by
    name after drawing rather than filter, so hypothesis doesn't have to search
    for valid draws (``st.lists(..., unique_by=...)`` would work too, but a
    post-hoc dedup keeps this readable and is stable across hypothesis versions).
    """
    return st.none() | st.lists(_mcp_tool_config(), max_size=len(MCP_TOOL_NAMES)).map(
        lambda cfgs: list({c.name: c for c in cfgs}.values()) or None
    )


def _mcp_toolset_tool() -> st.SearchStrategy[ToolSpec]:
    return st.builds(
        ToolSpec,
        type=st.just("mcp_toolset"),
        mcp_server_name=st.sampled_from(MCP_SERVER_NAMES),
        enabled=st.booleans(),
        default_config=st.none()
        | st.builds(
            McpToolsetConfig,
            enabled=st.booleans(),
            permission_policy=_mpp(),
            transport=_transport(),
        ),
        configs=_mcp_configs_list(),
    )


def tools_list() -> st.SearchStrategy[list[ToolSpec]]:
    """A ``tools`` list satisfying :func:`aios.models.agents.validate_tools`
    (unique attenuation identity key: builtin by type, custom by name, toolset by
    server) — dedup by key after drawing, same approach as ``_mcp_configs_list``.
    """

    def _key(t: ToolSpec) -> tuple[str, str | None]:
        if t.type == "mcp_toolset":
            return ("mcp_toolset", t.mcp_server_name)
        if t.type == "custom":
            return ("custom", t.name)
        return ("builtin", t.type)

    one_tool = _builtin_tool() | _custom_tool() | _mcp_toolset_tool()
    return st.lists(one_tool, max_size=4).map(lambda ts: list({_key(t): t for t in ts}.values()))


# ── mcp servers ──────────────────────────────────────────────────────────────


def _mcp_headers() -> st.SearchStrategy[dict[str, str] | None]:
    return st.none() | st.just({"X-Tag": "a"}) | st.just({"X-Tag": "b"})


def _mcp_server() -> st.SearchStrategy[McpServerSpec]:
    return st.builds(
        McpServerSpec,
        name=st.sampled_from(MCP_SERVER_NAMES),
        url=st.sampled_from(MCP_URLS),
        include_instructions=st.booleans(),
        headers=_mcp_headers(),
    )


def mcp_servers_list() -> st.SearchStrategy[list[McpServerSpec]]:
    """An ``mcp_servers`` list satisfying :func:`validate_mcp_servers` (unique
    ``name``) — dedup by name after drawing.
    """
    return st.lists(_mcp_server(), max_size=len(MCP_SERVER_NAMES)).map(
        lambda ss: list({s.name: s for s in ss}.values())
    )


# ── http servers (prioritized dimension — #1487 shipped here) ─────────────────


def _methods() -> st.SearchStrategy[list[HttpMethod] | None]:
    # None (top/all-verbs), [] (bottom/deny-all), or a proper subset — every
    # method-lattice element #828 introduced, including duplicates-before-norm
    # to exercise `_norm_methods`' dedup.
    return st.none() | st.just([]) | st.lists(st.sampled_from(METHOD_POOL), min_size=1, max_size=4)


def _http_route(*, path_pattern: str | None = None) -> st.SearchStrategy[HttpRouteSpec]:
    pattern = st.just(path_pattern) if path_pattern is not None else st.sampled_from(PATH_PATTERNS)
    return st.builds(
        HttpRouteSpec,
        path_pattern=pattern,
        enabled=st.booleans(),
        permission_policy=_hpp(),
        methods=_methods(),
        allow_query=st.booleans(),
        suppress=st.none() | st.booleans(),
    )


def _http_routes_list() -> st.SearchStrategy[list[HttpRouteSpec]]:
    """A server's ``routes``: 0-3 routes, path_pattern drawn from the SAME small
    pool so duplicate-``path_pattern`` routes with disjoint methods (#1487's exact
    shape) show up often — each route independently drawn (not deduped), since a
    server legitimately carries >1 route sharing a pattern (test_attenuation.py's
    ``test_http_route_meet_duplicate_path_pattern_preserves_each_verb``).
    """
    return st.lists(_http_route(), max_size=3)


def _http_server() -> st.SearchStrategy[HttpServerSpec]:
    return st.builds(
        HttpServerSpec,
        name=st.sampled_from(HTTP_SERVER_NAMES),
        base_url=st.sampled_from(HTTP_BASE_URLS),
        description=st.none() | st.just("d"),
        routes=_http_routes_list(),
        suppressed_response_status=st.just(200),
    )


def http_servers_list() -> st.SearchStrategy[list[HttpServerSpec]]:
    """An ``http_servers`` list satisfying :func:`validate_http_servers` (unique
    ``base_url``) — dedup by ``base_url`` after drawing.
    """
    return st.lists(_http_server(), max_size=len(HTTP_BASE_URLS)).map(
        lambda ss: list({s.base_url: s for s in ss}.values())
    )
