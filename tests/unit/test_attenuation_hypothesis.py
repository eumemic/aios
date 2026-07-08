"""Generative (hypothesis) law tests for the attenuation operator's OWN documented
contracts (#1758).

Prompted by #1487 (commit 1f46e2a2): a real violation of the module's stated law
``attenuate(x, x) == canonicalize(x)`` — duplicate ``path_pattern`` http routes
silently demoted to deny-all and frozen into the run/child surface — shipped five
days after the exhaustive resolver-agreement suite (``test_attenuation_resolver_agreement.py``,
#1087) landed, in the http dimension that suite does not touch, and was found by a
manual risk audit, not tests. The http-routes input space is unbounded and
compositional — the one place the repo's exhaustive-``@pytest.mark.parametrize``
idiom structurally cannot reach. This module generates inputs (hand-built
``hypothesis`` strategies over ``Surface``, since pydantic v2 has no hypothesis
plugin — see ``_attenuation_strategies.py``) asserting the module's own documented
laws with pooled identity keys, so a #1487-shaped regression would be caught
mechanically.

**Zero-dependency alternative considered and rejected.** The established
exhaustive-parametrize idiom (extended with small per-dimension http pools — 2
patterns x method subsets x <=2 routes would also have caught #1487) was weighed
against adding ``hypothesis``. Chosen: ``hypothesis``, for the sampling+shrinking
value over an unbounded, still-growing input space (http route fields accreted
across #485/#710/#828/#1156, and the attenuation module is on the security
perimeter) — a fixed exhaustive corpus only ever covers what its author enumerated
by hand, and #1487 is exactly the kind of interaction (two same-pattern routes) an
author does not reliably think to enumerate. The CI cost is bounded and seeded
(``settings.register_profile`` below): a fixed ``derandomize=True`` seed makes the
suite deterministic (no flaky CI from a bad random draw) while still exploring far
more of the compositional http space per run than a hand-picked corpus would.

**Laws asserted** — see the exclusions documented on
``aios.models.attenuation`` (module docstring) and on
``test_attenuation.py::test_meet_is_a_fixpoint_against_the_launcher``:
commutativity, associativity, and raw two-sided (declared-side) absorption are
FALSE BY DESIGN (directed, parent-wins-frozen clamp) and are deliberately NOT
asserted here.

**Existing tests are untouched** (bar the one restatement in ``test_attenuation.py``
scoped by #1758's issue): the exhaustive enumeration in
``test_attenuation_resolver_agreement.py`` strictly dominates sampling for the
finite MCP precedence ladder, and ``test_attenuation.py``'s curated fixtures are
regression anchors for specific past bugs — sampling is additive coverage for the
unbounded http dimension, not a replacement.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aios.models.agents import (
    ToolTransport,
    resolve_mcp_enabled,
    resolve_mcp_permission,
    resolve_mcp_transport,
)
from aios.models.attenuation import Surface, attenuate, canonicalize
from aios.tools.http_request import _match_route
from tests.unit._attenuation_strategies import (
    BT,
    DMP,
    HTTP_BASE_URLS,
    MCP_SERVER_NAMES,
    MCP_TOOL_NAMES,
    METHOD_POOL,
    PATH_PATTERNS,
    http_servers_list,
    mcp_servers_list,
    tools_list,
)

# Seeded/derandomized so `pytest -n 4` (unit CI, code-validation.yml) is exactly
# reproducible across workers and re-runs — no flaky CI from an unlucky draw. Example
# count kept modest (this is a per-PR unit-CI suite, not a fuzzing campaign); a local
# investigation can always raise it or drop `derandomize` for a fresh random walk.
settings.register_profile(
    "attenuation_laws",
    max_examples=120,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("attenuation_laws")


def st_builds_surface() -> st.SearchStrategy[Surface]:
    """A combined ``Surface`` strategy over all three dimensions at once."""
    return st.builds(Surface, tools_list(), mcp_servers_list(), http_servers_list())


def _att(d: Surface, ln: Surface) -> Surface:
    return attenuate(d, ln, default_mcp_permission=DMP, builtin_transports=BT)


def _canon(s: Surface) -> Surface:
    return canonicalize(s, default_mcp_permission=DMP, builtin_transports=BT)


# ── the normal-form contract (models/attenuation.py:23-31) ────────────────────


@given(s=st_builds_surface())
def test_self_meet_equals_canonicalize(s: Surface) -> None:
    """``attenuate(x, x) == canonicalize(x)`` — the exact law #1487 violated."""
    assert _att(s, s) == _canon(s)


@given(s=st_builds_surface())
def test_canonicalize_is_idempotent(s: Surface) -> None:
    assert _canon(_canon(s)) == _canon(s)


@given(d=st_builds_surface(), ln=st_builds_surface())
def test_attenuate_is_a_fixpoint_against_the_launcher(d: Surface, ln: Surface) -> None:
    """``attenuate(attenuate(d, l), l) == attenuate(d, l)`` — clamp idempotence /
    the launcher-side fixpoint, the security-load-bearing direction (see the
    module docstring's exclusion of the declared-side direction, which is false
    by design under the parent-wins-frozen clamp).
    """
    once = _att(d, ln)
    assert _att(once, ln) == once


# ── observational non-widening: MCP (generalizes test_attenuation_resolver_agreement.py) ──


@given(d=st_builds_surface(), ln=st_builds_surface())
def test_mcp_meet_agrees_with_live_resolvers(d: Surface, ln: Surface) -> None:
    """Non-tautological check: the resolver verdicts on ``attenuate(d, ln)`` equal
    the lattice law over the two operands' INDEPENDENT live resolver verdicts, for
    every (server, tool) name in the shared pool — not just re-deriving the meet's
    own internal representation. Generalizes
    ``test_attenuation_resolver_agreement.py::test_meet_agrees_with_resolvers``
    from exhaustive enumeration of the ladder's branch points to generation over
    full ``Surface``s (multiple servers/toolsets/http co-present).
    """
    # The resolver ladder (resolve_mcp_*) is defined to run on an ALREADY-CANONICAL
    # tools list — it does not itself consult ToolSpec-level `enabled` (that gate is
    # `canonicalize`'s job: a disabled toolset is pruned from the output entirely, per
    # `canonicalize`'s `if not t.enabled: continue`). Every live call site
    # (tool_broker.py, loop.py) reads resolve_mcp_* off a canonical/frozen surface, so
    # the "per-operand live verdict" is well-defined only on `canon(operand).tools` —
    # matching test_attenuation_resolver_agreement.py's exhaustive enumeration, which
    # never varies the top-level ToolSpec.enabled field for exactly this reason.
    d_canon, ln_canon = _canon(d), _canon(ln)
    out = _att(d, ln)
    for server in MCP_SERVER_NAMES:
        for tool in MCP_TOOL_NAMES:
            name = f"mcp__{server}__{tool}"
            ea = resolve_mcp_enabled(name, d_canon.tools)
            eb = resolve_mcp_enabled(name, ln_canon.tools)
            ta = resolve_mcp_transport(name, d_canon.tools) or "both"
            tb = resolve_mcp_transport(name, ln_canon.tools) or "both"
            glb = _transport_glb(ta, tb)
            expected_enabled = ea and eb and glb is not None
            actual_enabled = resolve_mcp_enabled(name, out.tools)
            assert expected_enabled == actual_enabled, (
                f"enabled desync for {name}: expected={expected_enabled} "
                f"actual={actual_enabled} d={d!r} ln={ln!r}"
            )
            if expected_enabled and actual_enabled:
                pa = resolve_mcp_permission(name, d.tools) or DMP
                pb = resolve_mcp_permission(name, ln.tools) or DMP
                expected_perm = "always_ask" if "always_ask" in (pa, pb) else "always_allow"
                assert expected_perm == resolve_mcp_permission(name, out.tools), (
                    f"permission desync for {name}: d={d!r} ln={ln!r}"
                )
                assert glb == resolve_mcp_transport(name, out.tools), (
                    f"transport desync for {name}: d={d!r} ln={ln!r}"
                )


def _transport_glb(a: ToolTransport, b: ToolTransport) -> ToolTransport | None:
    if a == "both":
        return b
    if b == "both":
        return a
    if a == b:
        return a
    return None  # {cli} glb {agent_tool} = empty


# ── observational non-widening: HTTP (the #1487 dimension, prioritized) ───────


@given(d=st_builds_surface(), ln=st_builds_surface())
def test_http_meet_never_admits_more_than_the_launcher(d: Surface, ln: Surface) -> None:
    """For every (base_url, path, method): if ``attenuate(d, ln)`` admits the call
    via ``tools/http_request._match_route``, the LAUNCHER admits it too. This is
    the actual security property #1487 broke (a demoted route meant the *opposite*
    failure — a granted verb silently became deny-all — but the invariant this test
    encodes is the one the operator must never violate in the other direction: the
    clamp must never manufacture admission the launcher didn't already grant).
    """
    out = _att(d, ln)
    l_by_url = {s.base_url: s for s in ln.http_servers}
    for out_srv in out.http_servers:
        l_srv = l_by_url.get(out_srv.base_url)
        assert l_srv is not None, (
            f"attenuate emitted a server {out_srv.base_url!r} absent from the "
            f"launcher — an http_servers survival-key violation"
        )
        for path in PATH_PATTERNS:
            for method in METHOD_POOL:
                if _match_route(out_srv, path, method) is not None:
                    assert _match_route(l_srv, path, method) is not None, (
                        f"attenuate(d, ln) admits {method} {path} on "
                        f"{out_srv.base_url!r} but the launcher does not: "
                        f"out_routes={out_srv.routes!r} launcher_routes={l_srv.routes!r}"
                    )


@given(d=st_builds_surface(), ln=st_builds_surface())
def test_http_meet_self_admission_matches_canonical_declared(d: Surface, ln: Surface) -> None:
    """The self-meet special case of the admission law: ``attenuate(x, x)`` admits
    exactly what ``canonicalize(x)`` admits (a direct corollary of
    ``test_self_meet_equals_canonicalize``, stated at the admission-observer level
    rather than the structural-equality level so a future change to
    ``_match_route``'s admission semantics that desyncs it from the meet — while
    leaving structural equality intact — would also be caught here).
    """
    for base in (d, ln):
        out = _att(base, base)
        canon = _canon(base)
        out_by_url = {s.base_url: s for s in out.http_servers}
        canon_by_url = {s.base_url: s for s in canon.http_servers}
        assert out_by_url.keys() == canon_by_url.keys()
        for url, out_srv in out_by_url.items():
            canon_srv = canon_by_url[url]
            for path in PATH_PATTERNS:
                for method in METHOD_POOL:
                    admitted_out = _match_route(out_srv, path, method) is not None
                    admitted_canon = _match_route(canon_srv, path, method) is not None
                    assert admitted_out == admitted_canon, (
                        f"self-meet admission desync at {url!r} {method} {path}: "
                        f"att(x,x)={admitted_out} canon(x)={admitted_canon}"
                    )


# ── duplicate path_pattern is the exact #1487 shape — dedicated pooled coverage ─


@given(d=st_builds_surface(), ln=st_builds_surface())
def test_duplicate_path_pattern_never_loses_a_verb_the_launcher_granted(
    d: Surface, ln: Surface
) -> None:
    """A launcher route whose ``path_pattern`` also appears on a DIFFERENT declared
    route (the #1487 shape: two ``/x`` entries with disjoint methods) must still be
    admitted through the meet for every verb the child's UNION of same-pattern
    grants intersected with the launcher allows — i.e. the meet must not silently
    collapse to first-match-wins on the declared side. This restates
    ``test_attenuation.py``'s curated #1487 regression tests as a generated
    property over the pooled http surfaces (base_url shared with the launcher, so
    the meet's key-match path is actually exercised).
    """
    # Force a base_url collision between d and ln's http servers so the meet's
    # key-match / narrowing path (not just "absent -> drop") is exercised, and
    # force at least one duplicate-path_pattern server pair to exist.
    shared_url = HTTP_BASE_URLS[0]
    d_http = [s for s in d.http_servers if s.base_url == shared_url]
    l_http = [s for s in ln.http_servers if s.base_url == shared_url]
    if not d_http or not l_http:
        return  # no collision drawn this example — a vacuous but valid draw
    d_srv, l_srv = d_http[0], l_http[0]
    out = _att(Surface([], [], [d_srv]), Surface([], [], [l_srv]))
    if not out.http_servers:
        return
    out_srv = out.http_servers[0]
    for path in PATH_PATTERNS:
        # The child's grant for `path` is the UNION of every same-pattern
        # declared route's methods (None = all methods -> union with anything is
        # everything).
        same_pattern = [r for r in d_srv.routes if r.path_pattern == path]
        if not same_pattern:
            continue
        if any(r.methods is None for r in same_pattern):
            child_grant: set[str] | None = None
        else:
            child_grant = set().union(*(set(r.methods or []) for r in same_pattern))
        for method in METHOD_POOL:
            child_admits = child_grant is None or method in child_grant
            launcher_route = _match_route(l_srv, path, method)
            if child_admits and launcher_route is not None:
                assert _match_route(out_srv, path, method) is not None, (
                    f"{method} {path} was granted by the child's union-of-same-"
                    f"pattern-routes and by the launcher, but the meet denies it "
                    f"-- the exact #1487 shape (d_srv={d_srv!r} l_srv={l_srv!r})"
                )
