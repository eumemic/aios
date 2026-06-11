"""The capability-attenuation operator — one lattice meet over a principal's surface.

A *surface* is the triple ``(tools, mcp_servers, http_servers)``. Every principal
in the system (agent, workflow, run, child session) carries one. The single law of
the authority story is **materialize ⇒ clamp, frozen-once**: whenever a principal
materializes its surface it is set to ``attenuate(its declared, its launcher's
already-frozen effective)`` and never recomputed. Composition is free by
associativity — each launcher was itself born clamped, so a single-step meet against
the launcher's frozen effective equals the whole-chain fold; no ancestor walk.

This module is the pure operator behind that law: spec-in, spec-out, principal-
agnostic, no I/O. It lives at the bottom of the import graph (it imports only
``models.agents``) so the query layer, the services, and the workflow runtime can all
reach it without a cycle.

The operator has two uses:

* a **clamp** (run / child birth — store ``attenuate(declared, launcher)``), and
* a **predicate** (workflow author / edit — admit ``declared`` iff
  ``attenuate(declared, actor) == canonicalize(declared)``, i.e. ``declared`` is a
  fixpoint of the meet against the actor: it grants nothing the actor lacks).

**The normal-form contract.** ``canonicalize`` and ``attenuate`` must emit
*byte-identical* normal forms for the predicate's equality to be exact:
``attenuate(x, x) == canonicalize(x)`` and ``attenuate(attenuate(d, l), l) ==
attenuate(d, l)``. Every ``None``-inherit sentinel is resolved to its concrete
default *before* meeting (resolve-then-meet); ``configs[]`` are sorted by name and
default-equal entries dropped; empty collections use a single representation. The
meet only ever narrows, so the predicate's equality fails exactly when ``declared``
tried to widen.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

from aios.models.agents import (
    HttpServerSpec,
    McpPermissionPolicy,
    McpServerSpec,
    McpToolConfig,
    McpToolsetConfig,
    PermissionPolicy,
    ToolSpec,
    ToolTransport,
)


class Surface(NamedTuple):
    """A principal's capability surface — the meet operates on this triple.

    Constructed via :func:`surface_of` at the materialize edges (the author edge,
    ``create_run``, ``create_child_session``, the frozen-overlay read) from whatever
    carries the three lists; not a persisted type. The lists are treated as immutable
    by the operator — it never mutates an input, always builds fresh output.
    """

    tools: list[ToolSpec]
    mcp_servers: list[McpServerSpec]
    http_servers: list[HttpServerSpec]


class HasSurface(Protocol):
    """Anything carrying the surface triple — ``Agent``/``AgentVersion``/``Workflow``/``WfRun``."""

    tools: list[ToolSpec]
    mcp_servers: list[McpServerSpec]
    http_servers: list[HttpServerSpec]


def surface_of(principal: HasSurface) -> Surface:
    """Project a principal's ``(tools, mcp_servers, http_servers)`` into a ``Surface``."""
    return Surface(principal.tools, principal.mcp_servers, principal.http_servers)


def _tool_key(t: ToolSpec) -> tuple[str, str | None]:
    """A tool's attenuation identity: builtins by type, custom by name, toolsets by server.

    The join carrier for the tools dimension — two entries are comparable iff their
    keys are equal. (Module-private; the old ``services.workflows`` copy is gone.)
    """
    if t.type == "mcp_toolset":
        return ("mcp_toolset", t.mcp_server_name)
    if t.type == "custom":
        return ("custom", t.name)
    return ("builtin", t.type)


# ── per-discovered-tool authority triple (the meet's atom for MCP) ────────────


class _Triple(NamedTuple):
    """The resolved authority of one (discovered) MCP tool name: the meet's atom."""

    enabled: bool
    permission: PermissionPolicy
    transport: ToolTransport


# The single canonical representation of an authority-bottom (disabled) tool name.
# A disabled tool grants nothing, so its permission/transport are dead fields — but the
# normal-form contract requires two disabled entries to compare *equal*, so they are
# pinned to fixed values via ``_norm_triple`` wherever a triple is produced (both in
# ``canonicalize`` and in the meet). Without this, ``canonicalize`` would keep a disabled
# entry's authored transport while the meet narrowed it, falsely failing the author-edge
# predicate for a declaration that merely disables a tool. ``transport="cli"`` also hides
# the tool from the model-discovery path (``loop.discover_session_mcp_tools`` filters on
# transport only), complementing ``enabled=False`` (honoured by the CLI broker's
# ``resolve_mcp_enabled``) so a disabled tool is invisible to *both* readers.
_DISABLED = _Triple(enabled=False, permission="always_ask", transport="cli")


def _norm_triple(t: _Triple) -> _Triple:
    """Collapse any disabled triple to the single canonical bottom (dead fields fixed)."""
    return _DISABLED if not t.enabled else t


def _transport_glb(a: ToolTransport, b: ToolTransport) -> ToolTransport | None:
    """Greatest lower bound over ``{cli, agent_tool}`` with ``both`` as the top element.

    Returns ``None`` when the meet is empty (``cli`` ⊓ ``agent_tool``) — the tool is
    reachable from neither side and must drop.
    """
    if a == "both":
        return b
    if b == "both":
        return a
    if a == b:
        return a
    return None  # {cli} ⊓ {agent_tool} = ∅


def _permission_meet(a: PermissionPolicy, b: PermissionPolicy) -> PermissionPolicy:
    """``always_ask`` ⊏ ``always_allow``: the meet is the stricter (ask wins)."""
    return "always_ask" if "always_ask" in (a, b) else "always_allow"


def _triple_meet(a: _Triple, b: _Triple) -> _Triple:
    """Dimension-wise meet, normalized so any disabled result is the single bottom.

    An empty transport GLB (``cli`` ⊓ ``agent_tool``) means the tool is reachable from
    neither side — also a disabled bottom.
    """
    transport = _transport_glb(a.transport, b.transport)
    if transport is None:
        return _DISABLED
    permission = _permission_meet(a.permission, b.permission)
    return _norm_triple(
        _Triple(enabled=a.enabled and b.enabled, permission=permission, transport=transport)
    )


# ── MCP toolset normal form ───────────────────────────────────────────────────


def _toolset_default_triple(t: ToolSpec, default_mcp_permission: PermissionPolicy) -> _Triple:
    """Resolve a toolset's default (no-config) authority, reproducing the resolvers.

    Mirrors ``resolve_mcp_permission``/``resolve_mcp_transport``/``resolve_mcp_enabled``
    for the *fall-through* case (no matching ``configs[]`` entry): ``default_config``
    fields → ``spec.permission`` → the operator default; transport → ``"both"``.
    """
    dc = t.default_config
    if dc is not None and dc.permission_policy is not None:
        permission = dc.permission_policy.type
    elif t.permission is not None:
        permission = t.permission
    else:
        permission = default_mcp_permission
    enabled = dc.enabled if dc is not None else True
    transport = (dc.transport if dc is not None else None) or "both"
    return _norm_triple(_Triple(enabled=enabled, permission=permission, transport=transport))


def _config_triple(cfg: McpToolConfig, default_mcp_permission: PermissionPolicy) -> _Triple:
    """Resolve one ``configs[]`` entry, reproducing the resolver short-circuits exactly.

    A ``configs[]`` entry with ``permission_policy=None`` resolves to the *operator
    default*, NOT ``default_config`` (``resolve_mcp_permission`` returns ``None`` from
    the matched entry, blocking fall-through, then the caller substitutes the default);
    ``transport=None`` resolves to ``"both"``, NOT ``default_config.transport``.
    """
    permission = cfg.permission_policy.type if cfg.permission_policy else default_mcp_permission
    return _norm_triple(
        _Triple(enabled=cfg.enabled, permission=permission, transport=cfg.transport or "both")
    )


def _build_toolset(
    server_name: str, default: _Triple, configs: list[tuple[str, _Triple]]
) -> ToolSpec:
    """The single constructor of a normal-form ``mcp_toolset`` spec.

    Used by both ``canonicalize`` and the meet so their outputs are byte-identical.
    ``permission``/``transport`` on the ``ToolSpec`` itself are ``None`` — all
    authority lives in the concrete ``default_config`` + sorted ``configs``. An empty
    ``configs`` is represented as ``None`` (the field default), never ``[]``.
    """
    return ToolSpec(
        type="mcp_toolset",
        mcp_server_name=server_name,
        enabled=True,
        permission=None,
        transport=None,
        default_config=McpToolsetConfig(
            enabled=default.enabled,
            permission_policy=McpPermissionPolicy(type=default.permission),
            transport=default.transport,
        ),
        configs=[
            McpToolConfig(
                name=name,
                enabled=tr.enabled,
                permission_policy=McpPermissionPolicy(type=tr.permission),
                transport=tr.transport,
            )
            for name, tr in configs
        ]
        or None,
    )


def _canon_toolset(t: ToolSpec, default_mcp_permission: PermissionPolicy) -> ToolSpec:
    """Normalize a toolset to its normal form: concrete default + minimal sorted configs."""
    default = _toolset_default_triple(t, default_mcp_permission)
    configs: list[tuple[str, _Triple]] = []
    for cfg in sorted(t.configs or [], key=lambda c: c.name):
        triple = _config_triple(cfg, default_mcp_permission)
        if triple != default:  # drop entries equal to the default — they carry no info
            configs.append((cfg.name, triple))
    return _build_toolset(t.mcp_server_name or "", default, configs)


def _canon_default_triple(canon: ToolSpec) -> _Triple:
    """The default triple of a canonical toolset — a plain read of its concrete fields."""
    dc = canon.default_config
    assert dc is not None and dc.permission_policy is not None and dc.transport is not None
    return _Triple(dc.enabled, dc.permission_policy.type, dc.transport)


def _canon_toolset_name_triple(canon: ToolSpec, name: str) -> _Triple:
    """Read a canonical toolset's resolved triple for ``name`` (config override or default).

    ``canon`` is a normal-form toolset (every field concrete), so this is a plain read,
    not a re-resolution.
    """
    for cfg in canon.configs or []:
        if cfg.name == name:
            assert cfg.permission_policy is not None and cfg.transport is not None
            return _Triple(cfg.enabled, cfg.permission_policy.type, cfg.transport)
    return _canon_default_triple(canon)


def _meet_toolset(declared: ToolSpec, launcher: ToolSpec) -> ToolSpec:
    """Meet two *canonical* toolsets for the same server, on the resolved-per-name policy.

    ``out_default`` is the meet of the two defaults; each name in the union of both
    config sets gets the meet of its resolved triples, emitted explicitly iff it
    differs from ``out_default``. This proves the fall-through case: a child omitting a
    launcher-pinned name still gets an explicit pinned entry whenever the meet is
    stricter than the child's default.
    """
    out_default = _triple_meet(_canon_default_triple(declared), _canon_default_triple(launcher))
    names = sorted(
        {c.name for c in (declared.configs or [])} | {c.name for c in (launcher.configs or [])}
    )
    configs: list[tuple[str, _Triple]] = []
    for name in names:
        meet = _triple_meet(
            _canon_toolset_name_triple(declared, name),
            _canon_toolset_name_triple(launcher, name),
        )
        if meet != out_default:
            configs.append((name, meet))
    return _build_toolset(declared.mcp_server_name or "", out_default, configs)


# ── builtin / custom tool normal form ─────────────────────────────────────────


def _canon_builtin(t: ToolSpec, *, transport_default: ToolTransport) -> ToolSpec:
    """Resolve a builtin/custom tool's ``None`` sentinels to concrete defaults.

    ``permission=None`` → ``always_allow`` (the dispatch gate treats ``None`` as
    immediate); ``transport=None`` → the registry default (builtins) / ``"both"``
    (custom). Other fields (name/description/input_schema) ride along unchanged.
    """
    return t.model_copy(
        update={
            "permission": t.permission or "always_allow",
            "transport": t.transport or transport_default,
        }
    )


def _meet_builtin(declared: ToolSpec, launcher: ToolSpec) -> ToolSpec | None:
    """Meet two *canonical* builtin/custom tools (same key). ``None`` if transport ⊓ is empty.

    Emits the declared entry's definition (name/description/input_schema) with the
    meet of permission and transport.
    """
    assert declared.transport is not None and launcher.transport is not None
    assert declared.permission is not None and launcher.permission is not None
    transport = _transport_glb(declared.transport, launcher.transport)
    if transport is None:
        return None
    return declared.model_copy(
        update={
            "permission": _permission_meet(declared.permission, launcher.permission),
            "transport": transport,
        }
    )


# ── the operator ──────────────────────────────────────────────────────────────


def canonicalize(
    s: Surface,
    *,
    default_mcp_permission: PermissionPolicy,
    builtin_transports: dict[str, ToolTransport],
) -> Surface:
    """Resolve a surface to its normal form (the identity element of the meet).

    Prunes disabled tools and dangling toolsets, resolves every ``None`` sentinel to
    its concrete default, and emits toolsets in minimal-config normal form. Servers
    are kept verbatim (they are compared and copied wholesale; normalizing e.g.
    ``headers None→{}`` would create needless drift). ``canonicalize(x) ==
    attenuate(x, x)``.

    ``default_mcp_permission`` and ``builtin_transports`` are passed in (read from
    settings / the tool registry by the caller) to keep the operator pure.
    """
    server_names = {srv.name for srv in s.mcp_servers}
    tools: list[ToolSpec] = []
    for t in s.tools:
        if not t.enabled:
            continue  # disabled tools are invisible to every reader → drop
        if t.type == "mcp_toolset":
            if t.mcp_server_name not in server_names:
                continue  # dangling toolset — no server → no discovery → drop
            tools.append(_canon_toolset(t, default_mcp_permission))
        elif t.type == "custom":
            tools.append(_canon_builtin(t, transport_default="both"))
        else:
            tools.append(
                _canon_builtin(t, transport_default=builtin_transports.get(t.type, "both"))
            )
    return Surface(tools, list(s.mcp_servers), list(s.http_servers))


def attenuate(
    declared: Surface,
    launcher: Surface,
    *,
    default_mcp_permission: PermissionPolicy,
    builtin_transports: dict[str, ToolTransport],
) -> Surface:
    """The lattice meet: ``declared`` clamped to never exceed ``launcher``.

    Per identity key in ``declared``: absent from ``launcher`` → drop; present → the
    per-dimension meet, dropping if it bottoms out. Servers survive only on a key match
    and are emitted **launcher-verbatim** (parent-wins-frozen: routes and headers come
    from the launcher; the child narrows only by dropping whole servers). http servers
    survive on a ``base_url`` key and are emitted launcher-verbatim — the child's declared
    routes are inert here, consistent with the authoring gate, which now admits http
    servers by identity (``(name, base_url)``) and inherits the launcher's frozen routes.
    MCP servers key on the joint ``(name, url)`` — a re-pointed name is a different key
    and drops. Output is canonical (``attenuate(d, l)`` is a fixpoint of ``canonicalize``).
    """
    d = canonicalize(
        declared,
        default_mcp_permission=default_mcp_permission,
        builtin_transports=builtin_transports,
    )
    lau = canonicalize(
        launcher,
        default_mcp_permission=default_mcp_permission,
        builtin_transports=builtin_transports,
    )

    # http_servers — key base_url, launcher-verbatim survival.
    l_http = {srv.base_url: srv for srv in lau.http_servers}
    out_http = [l_http[srv.base_url] for srv in d.http_servers if srv.base_url in l_http]

    # mcp_servers — joint (name, url) key, launcher-verbatim survival.
    l_mcp = {(srv.name, srv.url): srv for srv in lau.mcp_servers}
    out_mcp = [l_mcp[(srv.name, srv.url)] for srv in d.mcp_servers if (srv.name, srv.url) in l_mcp]
    surviving_servers = {srv.name for srv in out_mcp}

    # tools — _tool_key, per-dimension meet (toolsets gated on surviving server).
    l_tools = {_tool_key(t): t for t in lau.tools}
    out_tools: list[ToolSpec] = []
    for t in d.tools:
        match = l_tools.get(_tool_key(t))
        if match is None:
            continue
        if t.type == "mcp_toolset":
            if t.mcp_server_name not in surviving_servers:
                continue  # the toolset's server was re-pointed or dropped
            out_tools.append(_meet_toolset(t, match))
        else:
            met = _meet_builtin(t, match)
            if met is not None:
                out_tools.append(met)
    return Surface(out_tools, out_mcp, out_http)


def surface_diff(expected: Surface, actual: Surface) -> dict[str, list[str]]:
    """Per-section identity-keys present in ``expected`` but dropped/narrowed in ``actual``.

    ``expected`` is ``canonicalize(declared)``; ``actual`` is ``attenuate(declared,
    launcher)``. Used to build the author-edge ``ForbiddenError`` detail — names the
    exact tools/servers the declared surface exceeded the actor on.

    Tools and mcp_servers are full-equality keyed checks (the value must match, not just
    the key) — a widened per-tool policy or a re-pointed server is flagged. http_servers
    are identity-keyed on ``(name, base_url)`` (membership only): a server is flagged iff
    its identity is absent from ``actual``, never for a route/field divergence — the
    authoring gate inherits the launcher's frozen routes, so identity is the whole test.
    """
    out: dict[str, list[str]] = {}

    actual_tools = {_tool_key(t): t for t in actual.tools}
    bad_tools = [
        t.name or t.mcp_server_name or t.type
        for t in expected.tools
        if actual_tools.get(_tool_key(t)) != t
    ]
    if bad_tools:
        out["tools"] = bad_tools

    actual_mcp = {(s.name, s.url): s for s in actual.mcp_servers}
    bad_mcp = [s.name for s in expected.mcp_servers if actual_mcp.get((s.name, s.url)) != s]
    if bad_mcp:
        out["mcp_servers"] = bad_mcp

    actual_ids = {(s.name, s.base_url) for s in actual.http_servers}
    bad_http = [s.name for s in expected.http_servers if (s.name, s.base_url) not in actual_ids]
    if bad_http:
        out["http_servers"] = bad_http

    return out
