"""Property/exhaustive agreement between the attenuation meet and the live MCP resolvers.

The MCP precedence ladder (per-tool ``configs[]`` entry → ``default_config`` → bare
``ToolSpec.permission`` → operator default, with transport/enabled fall-throughs and the
non-obvious matched-entry-``None`` short-circuit) is implemented **twice**:

* the **runtime authority path** — ``resolve_mcp_permission`` / ``resolve_mcp_transport``
  / ``resolve_mcp_enabled`` (:mod:`aios.models.agents`), consumed by the tool broker, the
  registry, and the agents service; and
* the **attenuation meet path** — ``_toolset_default_triple`` + ``_config_triple``
  (:mod:`aios.models.attenuation`), which re-derive the *same* ladder to normalize a
  toolset for the lattice ``meet``.

Until now the agreement between the two was asserted only **indirectly and by example**
(a handful of curated specs in ``test_attenuation.py``) and otherwise held only by four
"Mirrors ``resolve_mcp_*``" / "reproducing the resolver short-circuits exactly"
docstrings. This module converts that vigilance promise into a **test-enforced** property
over an exhaustive enumeration of the ladder's branch points: for every toolset spec and
every tool name, the live resolver's verdict on ``canonicalize(spec).tools`` — and on
``attenuate(a, b).tools`` — equals the verdict derived from the pre-canonical input. A
future edit to the ladder in ``models/agents.py`` (e.g. changing the matched-entry-``None``
short-circuit) that desyncs the meet from the runtime becomes a RED test here, not a
shipped authority bug.

Security-bearing (capability/authority resolution); the precedence ladder is the
highest-stakes path in the capability system, so its two implementations must provably
agree. This is a pure test addition — no production-code change. ``hypothesis`` is not a
repo dependency, so "property/exhaustive" here means a parametrized exhaustive enumeration
of the ladder's branch points, in the established ``@pytest.mark.parametrize`` idiom of
``test_attenuation.py``.

The agreement is stated at the *contract* level the runtime actually consumes:

* **enabled** — agrees **exactly** (the broker's enabled gate reads this verbatim).
* **permission / transport** — a disabled tool's permission/transport are dead fields
  (``canonicalize`` collapses every disabled triple to one ``_DISABLED`` bottom, so the
  resolvers read back fixed sentinels), so they are only compared **when the tool is
  enabled on both forms**. On the pre-canonical side a resolver ``None`` means "fall back
  to the operator default" — exactly what every live caller does
  (``services/agents.py``, ``tools/registry.py``) — so the pre-canonical verdict is
  normalized through that same fall-back (``None`` permission → operator default;
  ``None`` transport → ``"both"``) before comparison. The meet verdict is derived from the
  per-operand resolver verdicts via the lattice law (permission meet = stricter,
  transport = GLB, enabled = AND), then compared to the live resolver on the meet output.
"""

from __future__ import annotations

import itertools

import pytest

from aios.models.agents import (
    McpPermissionPolicy,
    McpServerSpec,
    McpToolConfig,
    McpToolsetConfig,
    PermissionPolicy,
    ToolSpec,
    ToolTransport,
    resolve_mcp_enabled,
    resolve_mcp_permission,
    resolve_mcp_transport,
)
from aios.models.attenuation import Surface, attenuate, canonicalize

# The two operator defaults to exercise — the matched-entry-``None`` short-circuit must
# resolve to *this* value (NOT ``default_config``), so we sweep both so the test pins the
# short-circuit's direction rather than coincidentally agreeing under one default.
DMPS: list[PermissionPolicy] = ["always_allow", "always_ask"]
# Builtin transports are irrelevant to MCP toolsets; an empty map keeps the MCP ladder
# the sole variable under test.
BT: dict[str, ToolTransport] = {}

SERVER = "s"
SRV = McpServerSpec(name=SERVER, url="https://s")

# The tool the configs[] entry targets, plus a name with *no* matching entry so the
# fall-through (default_config / bare spec / operator default) branch is exercised too.
TARGET = f"mcp__{SERVER}__t"
FALLTHROUGH = f"mcp__{SERVER}__other"
NAMES = [TARGET, FALLTHROUGH]

# The ladder's per-field branch points.
PERMS: list[PermissionPolicy | None] = [None, "always_allow", "always_ask"]
TRANSPORTS: list[ToolTransport | None] = [None, "cli", "agent_tool", "both"]
ENABLEDS = [True, False]


def _mpp(p: PermissionPolicy | None) -> McpPermissionPolicy | None:
    return McpPermissionPolicy(type=p) if p is not None else None


def _toolset_specs() -> list[ToolSpec]:
    """Exhaustive enumeration over the precedence ladder's branch points.

    Covers, for one MCP server:

    * the **bare-spec fall-through** — ``ToolSpec.permission`` set vs ``None``, no
      ``default_config``, no ``configs`` (the deepest fall-through to the operator
      default);
    * the **default_config fall-through** — every ``(enabled, permission_policy,
      transport)`` combination, each field present vs ``None`` (``None`` inheriting from
      the next rung);
    * a **matching ``configs[]`` entry** — every ``(enabled, permission_policy,
      transport)`` combination, with and without a (deliberately *non-default*)
      ``default_config`` underneath it, so the matched-entry-``None`` short-circuit is
      exercised against a ``default_config`` it must NOT fall through to.
    """
    specs: list[ToolSpec] = []

    # 1. Bare spec.permission fall-through (no default_config, no configs).
    for sp in PERMS:
        specs.append(ToolSpec(type="mcp_toolset", mcp_server_name=SERVER, permission=sp))

    # 2. default_config fall-through (no configs).
    for en, pp, tr in itertools.product(ENABLEDS, PERMS, TRANSPORTS):
        specs.append(
            ToolSpec(
                type="mcp_toolset",
                mcp_server_name=SERVER,
                default_config=McpToolsetConfig(
                    enabled=en, permission_policy=_mpp(pp), transport=tr
                ),
            )
        )

    # 3. A matching configs[] entry for TARGET. The matched-entry-``None`` short-circuit
    #    means an entry's ``permission_policy=None`` → operator default (NOT
    #    default_config) and ``transport=None`` → ``"both"`` (NOT default_config.transport).
    #    A deliberately non-default default_config sits underneath so a regression that
    #    *did* fall through to it would diverge and turn this test red.
    underneath = McpToolsetConfig(
        enabled=True,
        permission_policy=McpPermissionPolicy(type="always_allow"),
        transport="agent_tool",
    )
    for en, pp, tr in itertools.product(ENABLEDS, PERMS, TRANSPORTS):
        cfg = McpToolConfig(name="t", enabled=en, permission_policy=_mpp(pp), transport=tr)
        specs.append(ToolSpec(type="mcp_toolset", mcp_server_name=SERVER, configs=[cfg]))
        specs.append(
            ToolSpec(
                type="mcp_toolset",
                mcp_server_name=SERVER,
                default_config=underneath,
                configs=[cfg],
            )
        )

    return specs


_SPECS = _toolset_specs()

# A small, sharp sub-corpus for the (quadratic) meet sweep — chosen to span the ladder's
# branch points so the pair-meet still hits the short-circuit, the fall-through pin, and
# transport-GLB-empty (disjoint cli/agent_tool) without exploding the case count.
_MEET_SPECS = [
    ToolSpec(type="mcp_toolset", mcp_server_name=SERVER),
    ToolSpec(type="mcp_toolset", mcp_server_name=SERVER, permission="always_ask"),
    ToolSpec(
        type="mcp_toolset",
        mcp_server_name=SERVER,
        default_config=McpToolsetConfig(
            enabled=True,
            permission_policy=McpPermissionPolicy(type="always_allow"),
            transport="cli",
        ),
        configs=[McpToolConfig(name="t")],  # permission_policy=None → matched-entry short-circuit
    ),
    ToolSpec(
        type="mcp_toolset",
        mcp_server_name=SERVER,
        configs=[
            McpToolConfig(name="t", transport="agent_tool", permission_policy=_mpp("always_ask"))
        ],
    ),
    ToolSpec(
        type="mcp_toolset",
        mcp_server_name=SERVER,
        default_config=McpToolsetConfig(enabled=False),  # disabled default → fall-through off
    ),
    ToolSpec(
        type="mcp_toolset",
        mcp_server_name=SERVER,
        configs=[McpToolConfig(name="t", enabled=False)],  # disabled override on TARGET
    ),
]


def _canon(s: Surface, *, dmp: PermissionPolicy) -> Surface:
    return canonicalize(s, default_mcp_permission=dmp, builtin_transports=BT)


def _att(d: Surface, ln: Surface, *, dmp: PermissionPolicy) -> Surface:
    return attenuate(d, ln, default_mcp_permission=dmp, builtin_transports=BT)


def _resolved_permission(
    name: str, tools: list[ToolSpec], *, dmp: PermissionPolicy
) -> PermissionPolicy:
    """The permission a live caller acts on: resolver verdict with ``None`` → operator default."""
    p = resolve_mcp_permission(name, tools)
    return p if p is not None else dmp


def _resolved_transport(name: str, tools: list[ToolSpec]) -> ToolTransport:
    """The transport a live caller acts on: resolver verdict with ``None`` → ``"both"``."""
    t = resolve_mcp_transport(name, tools)
    return t if t is not None else "both"


def _permission_meet(a: PermissionPolicy, b: PermissionPolicy) -> PermissionPolicy:
    """The lattice meet on permission: ``always_ask`` ⊏ ``always_allow`` (ask wins)."""
    return "always_ask" if "always_ask" in (a, b) else "always_allow"


def _transport_glb(a: ToolTransport, b: ToolTransport) -> ToolTransport | None:
    """GLB over ``{cli, agent_tool}`` with ``both`` as top; ``None`` when empty (disjoint)."""
    if a == "both":
        return b
    if b == "both":
        return a
    if a == b:
        return a
    return None


def _surface(spec: ToolSpec) -> Surface:
    return Surface([spec], [SRV], [])


# ── canonicalize preserves every live resolver verdict ────────────────────────


@pytest.mark.parametrize("dmp", DMPS)
@pytest.mark.parametrize("spec", _SPECS)
def test_canonicalize_preserves_resolver_verdict(spec: ToolSpec, dmp: PermissionPolicy) -> None:
    """For every spec + tool name: the resolvers agree on ``canonicalize(spec)`` vs the input.

    ``enabled`` agrees exactly. ``permission``/``transport`` are dead fields for a disabled
    tool (canonicalize collapses disabled triples to one bottom), so they are compared only
    when the tool is enabled on both forms — matching what the broker consumes (it gates on
    ``enabled`` first).
    """
    pre = _surface(spec)
    post = _canon(pre, dmp=dmp)

    for name in NAMES:
        pre_enabled = resolve_mcp_enabled(name, pre.tools)
        post_enabled = resolve_mcp_enabled(name, post.tools)
        assert pre_enabled == post_enabled, (
            f"enabled desync for {name}: pre={pre_enabled} post={post_enabled} spec={spec!r}"
        )

        if pre_enabled and post_enabled:
            assert _resolved_permission(name, pre.tools, dmp=dmp) == _resolved_permission(
                name, post.tools, dmp=dmp
            ), f"permission desync for {name} (dmp={dmp}) spec={spec!r}"
            assert _resolved_transport(name, pre.tools) == _resolved_transport(name, post.tools), (
                f"transport desync for {name} spec={spec!r}"
            )


def test_matched_entry_none_short_circuits_to_operator_default() -> None:
    """The genuinely non-obvious rule, pinned directly across both implementations.

    A matching ``configs[]`` entry with ``permission_policy=None`` resolves to the
    **operator default**, NOT ``default_config`` — even when ``default_config`` pins a
    different permission. The runtime resolver (``resolve_mcp_permission`` returns ``None``
    from the matched entry, caller substitutes the default) and the meet's ``_config_triple``
    (substitutes ``default_mcp_permission`` inline) must agree. We sweep both operator
    defaults so the assertion pins the short-circuit's *direction*, not a coincidence.
    """
    spec = ToolSpec(
        type="mcp_toolset",
        mcp_server_name=SERVER,
        default_config=McpToolsetConfig(permission_policy=McpPermissionPolicy(type="always_allow")),
        configs=[McpToolConfig(name="t")],  # permission_policy=None → short-circuit
    )
    for dmp in DMPS:
        post = _canon(_surface(spec), dmp=dmp)
        # The matched name resolves to the OPERATOR default, not default_config's always_allow.
        assert resolve_mcp_permission(TARGET, post.tools) == dmp
        # The fall-through name (no matching entry) DOES see default_config.
        assert resolve_mcp_permission(FALLTHROUGH, post.tools) == "always_allow"


# ── the meet preserves the live resolver verdict (derived via the lattice law) ─


@pytest.mark.parametrize("dmp", DMPS)
@pytest.mark.parametrize("b", _MEET_SPECS)
@pytest.mark.parametrize("a", _MEET_SPECS)
def test_meet_agrees_with_resolvers(a: ToolSpec, b: ToolSpec, dmp: PermissionPolicy) -> None:
    """For every spec pair + tool name: the resolvers on ``attenuate(a, b)`` equal the
    verdict the lattice law derives from the two operands' live resolver verdicts.

    The expected verdict is built **only** from the live ``resolve_mcp_*`` calls on each
    operand (so the precedence ladder is exercised through the runtime path) combined by the
    lattice law — permission meet (stricter wins), transport GLB, enabled = AND, with an
    empty transport GLB (disjoint ``cli``/``agent_tool``) bottoming out to disabled. This
    pins the meet's ladder re-derivation to the runtime ladder without re-implementing the
    ladder here.
    """
    sa, sb = _surface(a), _surface(b)
    out = _att(sa, sb, dmp=dmp)

    for name in NAMES:
        ea = resolve_mcp_enabled(name, sa.tools)
        eb = resolve_mcp_enabled(name, sb.tools)
        ta = _resolved_transport(name, sa.tools)
        tb = _resolved_transport(name, sb.tools)
        glb = _transport_glb(ta, tb)

        expected_enabled = ea and eb and glb is not None
        actual_enabled = resolve_mcp_enabled(name, out.tools)
        assert expected_enabled == actual_enabled, (
            f"meet enabled desync for {name} (dmp={dmp}): "
            f"expected={expected_enabled} actual={actual_enabled} a={a!r} b={b!r}"
        )

        if expected_enabled and actual_enabled:
            pa = _resolved_permission(name, sa.tools, dmp=dmp)
            pb = _resolved_permission(name, sb.tools, dmp=dmp)
            assert _permission_meet(pa, pb) == resolve_mcp_permission(name, out.tools), (
                f"meet permission desync for {name} (dmp={dmp}) a={a!r} b={b!r}"
            )
            assert glb == resolve_mcp_transport(name, out.tools), (
                f"meet transport desync for {name} a={a!r} b={b!r}"
            )
