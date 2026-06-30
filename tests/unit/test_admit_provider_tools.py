"""The provider-tool clamp (``admit_provider_tools``) — pure unit tests (#1627).

Pure in-memory: no Postgres, no Docker. Covers the two pure-helper acceptance cases
from #1627: a provider tool whose ``_tool_key`` is ABSENT from the effective (frozen,
clamped) surface is dropped, and a provider tool that is PRESENT but narrowed survives
carrying the exact per-dimension meet (``always_ask`` wins, transport GLB) — byte-equal
to what ``_meet_builtin`` produces. This closes the prelude attenuation-visibility gap:
the ToolProvider seam can no longer re-grant authority a workflow run dropped.
"""

from __future__ import annotations

from aios.models.agents import PermissionPolicy, ToolSpec, ToolTransport
from aios.models.attenuation import (
    Surface,
    _canon_builtin,
    _meet_builtin,
    admit_provider_tools,
)

# Controlled defaults so the meet-logic tests are independent of the live registry.
DMP: PermissionPolicy = "always_ask"
BT: dict[str, ToolTransport] = {}


def _custom(
    name: str,
    *,
    permission: PermissionPolicy | None = None,
    transport: ToolTransport | None = None,
) -> ToolSpec:
    return ToolSpec(
        type="custom",
        name=name,
        description=f"the {name} connector tool",
        input_schema={"type": "object"},
        permission=permission,
        transport=transport,
    )


def test_admit_provider_tools_drops_absent() -> None:
    """A provider tool whose ``_tool_key`` is absent from ``effective`` is dropped."""
    provider = [_custom("X")]
    effective = Surface([_custom("Y")], [], [])  # X's key is not present
    assert (
        admit_provider_tools(provider, effective, default_mcp_permission=DMP, builtin_transports=BT)
        == []
    )


def test_admit_provider_tools_empty_effective_drops_all() -> None:
    """An empty effective surface (a maximally-clamped child) drops every provider tool."""
    provider = [_custom("X"), _custom("Z")]
    assert (
        admit_provider_tools(
            provider, Surface([], [], []), default_mcp_permission=DMP, builtin_transports=BT
        )
        == []
    )


def test_admit_provider_tools_meets_present() -> None:
    """A present-but-narrowed provider tool survives carrying the exact meet.

    ``effective`` narrows X to ``always_ask`` + ``cli`` transport; the provider entry is
    the wider ``always_allow``/``both``. The surviving entry is byte-equal to
    ``_meet_builtin`` output: ``always_ask`` wins, transport GLB → ``cli``.
    """
    provider_x = _custom("X")  # wide: permission/transport None → always_allow/both
    eff_x = _custom("X", permission="always_ask", transport="cli")
    effective = Surface([eff_x], [], [])

    got = admit_provider_tools(
        [provider_x], effective, default_mcp_permission=DMP, builtin_transports=BT
    )

    # Reference meet: canonicalize both sides exactly as the operator/helper do.
    expected = _meet_builtin(
        _canon_builtin(provider_x, transport_default="both"),
        _canon_builtin(eff_x, transport_default="both"),
    )
    assert got == [expected]
    assert got[0].permission == "always_ask"  # always_ask wins
    assert got[0].transport == "cli"  # transport GLB(both, cli) = cli


def test_admit_provider_tools_definition_from_provider() -> None:
    """The surviving entry keeps the provider tool's definition (name/description/schema)."""
    provider_x = _custom("X")
    eff_x = ToolSpec(
        type="custom",
        name="X",
        description="effective-side description (ignored by the meet emit)",
        input_schema={"type": "object", "properties": {"a": {}}},
        permission="always_ask",
    )
    [got] = admit_provider_tools(
        [provider_x], Surface([eff_x], [], []), default_mcp_permission=DMP, builtin_transports=BT
    )
    # ``_meet_builtin`` emits the *declared* (provider) definition with the met policy.
    assert got.name == "X"
    assert got.description == "the X connector tool"
    assert got.input_schema == {"type": "object"}


def test_admit_provider_tools_subset_filter() -> None:
    """Only the provider tools present in ``effective`` survive; order follows provider."""
    provider = [_custom("A"), _custom("B"), _custom("C")]
    effective = Surface([_custom("C"), _custom("A")], [], [])
    got = admit_provider_tools(
        provider, effective, default_mcp_permission=DMP, builtin_transports=BT
    )
    assert [t.name for t in got] == ["A", "C"]
