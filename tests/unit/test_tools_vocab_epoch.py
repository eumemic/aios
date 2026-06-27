"""Tests for the tools-vocab epoch belt (#1576, epic #1572).

The per-row epoch stamp is the raw-restore belt: a row records the latest
backfill its blob was known to satisfy, so a chain-bypassing ``pg_restore`` of
an old DB self-describes its staleness. These tests pin the two pure-Python
halves of the mechanism:

* :mod:`aios.retirements.epoch` — the single source of truth for the latest
  backfill rev (derived from the registry's ``contract_rev`` values, never a
  parallel constant), and
* :mod:`aios.retirements.upcast` — the restore/import hook that detects a stale
  blob and re-upcasts it to canonical vocabulary before insert.

The migration (column + per-surface partial index) is exercised against a real
Postgres in ``tests/integration/test_migrations_0124_tools_vocab_epoch.py``.
"""

from __future__ import annotations

from aios.retirements import Retirement, Surface
from aios.retirements import epoch as epoch_mod
from aios.retirements import registry as reg
from aios.retirements import upcast as upcast_mod

# A synthetic domain + descriptors so the tests are independent of how the live
# registry evolves.
_SURF = Surface(table="t", jsonb_col="tools", predicate_sql="x = :token")


def _rename(*, contract_rev: str | None, introduced_rev: str = "0001") -> Retirement:
    return Retirement(
        domain="dom",
        action="rename",
        mappings=(("old_a", "new_x"), ("old_b", "new_x"), ("old_c", "new_y")),
        surfaces=(_SURF,),
        introduced_rev=introduced_rev,
        contract_rev=contract_rev,
    )


def _drop(*, contract_rev: str | None = "0005") -> Retirement:
    return Retirement(
        domain="dom",
        action="drop",
        mappings=(("gone_a", None), ("gone_b", None)),
        surfaces=(_SURF,),
        introduced_rev="0005",
        contract_rev=contract_rev,
    )


# ── epoch: latest-backfill-rev derivation ────────────────────────────────────


def test_epoch_is_max_contract_rev_across_domain() -> None:
    registry = (_rename(contract_rev="0010"), _drop(contract_rev="0122"))
    assert epoch_mod.latest_backfill_rev("dom", registry=registry) == 122


def test_epoch_skips_expand_span_descriptors() -> None:
    # A retirement still in its expand span (contract_rev is None) is NOT a
    # backfill and contributes no epoch.
    registry = (_rename(contract_rev=None), _drop(contract_rev="0007"))
    assert epoch_mod.latest_backfill_rev("dom", registry=registry) == 7


def test_epoch_floor_is_zero_with_no_backfill() -> None:
    # No declared backfill yet → epoch floor 0, the same value the column
    # defaults to, so nothing reads as stale before the first backfill exists.
    registry = (_rename(contract_rev=None),)
    assert epoch_mod.latest_backfill_rev("dom", registry=registry) == 0


def test_epoch_ignores_other_domains() -> None:
    other = Retirement(
        domain="other",
        action="drop",
        token="z",
        surfaces=(_SURF,),
        introduced_rev="0999",
        contract_rev="0999",
    )
    registry = (_drop(contract_rev="0008"), other)
    assert epoch_mod.latest_backfill_rev("dom", registry=registry) == 8


def test_live_epoch_matches_registry_latest_tool_surface_contract() -> None:
    # The shipped constant tracks the live registry — the latest declared
    # tool_surface backfill (currently rev 0122, the goal-outcome drop).
    assert epoch_mod.latest_backfill_rev(
        reg.TOOL_SURFACE_DOMAIN
    ) == epoch_mod.TOOLS_VOCAB_EPOCH
    assert epoch_mod.TOOLS_VOCAB_EPOCH == 122


# ── registry: dropped_tokens helper ──────────────────────────────────────────


def test_dropped_tokens_collects_only_drops() -> None:
    registry = (_rename(contract_rev="0010"), _drop())
    assert reg.dropped_tokens("dom", registry=registry) == frozenset({"gone_a", "gone_b"})


def test_dropped_tokens_live_registry() -> None:
    # The live drop set is exactly the goal-outcome builtins.
    assert reg.dropped_tokens() == frozenset({"complete_goal", "fail_goal"})


# ── upcast: staleness predicate ──────────────────────────────────────────────


def test_is_stale_below_horizon() -> None:
    assert upcast_mod.is_stale(0, current=122) is True
    assert upcast_mod.is_stale(121, current=122) is True


def test_is_stale_at_or_above_horizon_is_current() -> None:
    assert upcast_mod.is_stale(122, current=122) is False
    assert upcast_mod.is_stale(200, current=122) is False


def test_is_stale_none_is_stale_fail_safe() -> None:
    # Absent a positive currency assertion, fail safe and treat as stale.
    assert upcast_mod.is_stale(None, current=122) is True


# ── upcast: canonicalize_tool_blob ───────────────────────────────────────────

_RENAMES = {"old_a": "new_x", "old_b": "new_x", "old_c": "new_y"}
_DROPS = frozenset({"gone_a", "gone_b"})


def test_canonicalize_renames_and_dedupes_builtins() -> None:
    blob = [{"type": "old_a"}, {"type": "old_b"}, {"type": "old_c"}]
    out = upcast_mod.canonicalize_tool_blob(blob, renames=_RENAMES, drops=_DROPS)
    # old_a + old_b both fold to new_x → deduped to one; old_c → new_y. Order kept.
    assert [e["type"] for e in out] == ["new_x", "new_y"]


def test_canonicalize_drops_outright() -> None:
    blob = [{"type": "gone_a"}, {"type": "old_c"}, {"type": "gone_b"}]
    out = upcast_mod.canonicalize_tool_blob(blob, renames=_RENAMES, drops=_DROPS)
    assert [e["type"] for e in out] == ["new_y"]


def test_canonicalize_only_dropped_collapses_to_empty() -> None:
    blob = [{"type": "gone_a"}, {"type": "gone_b"}]
    assert upcast_mod.canonicalize_tool_blob(blob, renames=_RENAMES, drops=_DROPS) == []


def test_canonicalize_preserves_custom_and_never_dedupes_it() -> None:
    blob = [
        {"type": "custom", "name": "a"},
        {"type": "old_a"},
        {"type": "custom", "name": "b"},
    ]
    out = upcast_mod.canonicalize_tool_blob(blob, renames=_RENAMES, drops=_DROPS)
    assert out == [
        {"type": "custom", "name": "a"},
        {"type": "new_x"},
        {"type": "custom", "name": "b"},
    ]


def test_canonicalize_clean_blob_unchanged() -> None:
    blob = [{"type": "new_x"}, {"type": "custom", "name": "x"}]
    out = upcast_mod.canonicalize_tool_blob(blob, renames=_RENAMES, drops=_DROPS)
    assert out == blob


def test_canonicalize_passes_through_non_dict_entries() -> None:
    blob = ["weird", {"type": "old_a"}]
    out = upcast_mod.canonicalize_tool_blob(blob, renames=_RENAMES, drops=_DROPS)
    assert out == ["weird", {"type": "new_x"}]


# ── upcast: reupcast_tool_surface (the hook) ─────────────────────────────────


def test_reupcast_stale_blob_canonicalized_and_restamped() -> None:
    # A stale blob carrying live-registry legacy tokens is canonicalized AND
    # stamped current — this is the restore/import hook's whole job.
    blob = [
        {"type": "invoke"},
        {"type": "complete_goal"},
        {"type": "create_run"},
        {"type": "invoke_workflow"},
        {"type": "custom", "name": "x"},
    ]
    out_blob, out_epoch = upcast_mod.reupcast_tool_surface(blob, 0)
    assert out_epoch == epoch_mod.TOOLS_VOCAB_EPOCH
    assert out_blob == [
        {"type": "call_session"},
        {"type": "call_workflow"},  # create_run + invoke_workflow fold + dedupe
        {"type": "custom", "name": "x"},
    ]


def test_reupcast_current_blob_passes_through_restamped() -> None:
    # An already-current blob is NOT rewritten (no canonicalization), only its
    # stamp is normalized to current — a forward stamp is never lowered.
    blob = [{"type": "invoke"}, {"type": "custom", "name": "x"}]
    out_blob, out_epoch = upcast_mod.reupcast_tool_surface(
        blob, epoch_mod.TOOLS_VOCAB_EPOCH
    )
    assert out_blob == blob  # untouched: a current row is trusted as-is
    assert out_epoch == epoch_mod.TOOLS_VOCAB_EPOCH


def test_reupcast_none_blob_passes_through_stamped_current() -> None:
    # A nullable surface (non-frozen sessions row) has no vocabulary to upcast;
    # it stays None and is stamped current so the boot scan skips it.
    out_blob, out_epoch = upcast_mod.reupcast_tool_surface(None, 0)
    assert out_blob is None
    assert out_epoch == epoch_mod.TOOLS_VOCAB_EPOCH
