"""The restore/import re-upcast hook — detect-and-canonicalize stale tool blobs.

Part of the per-row epoch belt (#1576, epic #1572). The epoch stamp lets a row
*self-describe* its staleness (``tools_vocab_epoch < latest-backfill-rev``); this
module is the **acting** side of that belt: a restore/import path routes a blob
through :func:`reupcast_tool_surface` BEFORE inserting it, so a stale blob is
re-upcast to canonical vocabulary at the boundary instead of being persisted
stale and waiting for the boot gate to notice.

Canonicalization is registry-driven: renames come from
:func:`aios.retirements.registry.rename_map` (including contracted descriptors,
so old snapshots remain importable) and drops come from
:func:`aios.retirements.registry.dropped_tokens` (the same set
``load_tool_specs`` drops). There is no parallel rename/drop table here — if a
token is retired in the registry it is upcast here, by construction.

**Honest limit (carry from the issue):** this hook only helps a restore/import
path that actually routes through it. A restore tool that writes directly to
Postgres bypasses it and falls back to the boot scan alone (still safe —
unready, not crash-loop — but it loses the pre-insert upcast). Restore runbooks
must route through this hook.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from aios.retirements.epoch import TOOLS_VOCAB_EPOCH
from aios.retirements.registry import dropped_tokens, rename_map


def is_stale(epoch: int | None, *, current: int = TOOLS_VOCAB_EPOCH) -> bool:
    """Does a row with ``tools_vocab_epoch == epoch`` need re-upcasting?

    A row is stale iff its stamp is below the current backfill horizon. A
    ``None`` epoch (a column not present / not read) is treated as stale: absent
    a positive assertion of currency, fail safe and re-upcast.
    """

    if epoch is None:
        return True
    return epoch < current


def canonicalize_tool_blob(
    raw: Iterable[Any],
    *,
    renames: dict[str, str] | None = None,
    drops: frozenset[str] | None = None,
) -> list[Any]:
    """Apply registry-declared renames + drops to a ``tools`` JSONB array.

    Mirrors the read-path tolerance so the persisted form a restore produces is
    identical to what the read shims would have yielded:

    * a ``rename`` token's ``type`` is rewritten to its canonical successor, then
      builtin entries are de-duplicated first-occurrence-wins (a rename can fold
      two legacy names onto one successor — ``invoke_workflow``/``create_run`` →
      ``call_workflow`` — and a list must not carry the same builtin twice);
    * a ``drop`` token's entry is removed outright (no successor exists);
    * ``custom`` / ``mcp_toolset`` entries are never de-duplicated (they are
      keyed by identity, not by ``type``);

    Original order is otherwise preserved; a list of only-dropped entries
    collapses to ``[]``. Non-dict entries pass through untouched (validation,
    not canonicalization, is their concern).

    ``renames`` / ``drops`` default to the live registry but are injectable so
    tests can scope the upcast to a synthetic descriptor set.
    """

    rename_tokens = rename_map() if renames is None else renames
    drop_set = dropped_tokens() if drops is None else drops

    out: list[Any] = []
    seen_builtin: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            out.append(entry)
            continue
        ttype = entry.get("type")
        if ttype in drop_set:
            continue
        if ttype in rename_tokens:
            new_type = rename_tokens[ttype]
            entry = {**entry, "type": new_type}
            ttype = new_type
        # De-dupe builtins (anything that is not a custom/mcp_toolset identity).
        if isinstance(ttype, str) and ttype not in ("custom", "mcp_toolset"):
            if ttype in seen_builtin:
                continue
            seen_builtin.add(ttype)
        out.append(entry)
    return out


def reupcast_tool_surface(
    blob: Iterable[Any] | None,
    epoch: int | None,
    *,
    current: int = TOOLS_VOCAB_EPOCH,
) -> tuple[list[Any] | None, int]:
    """Re-upcast a tool-surface blob at a restore/import boundary.

    Returns ``(canonical_blob, stamped_epoch)`` to persist:

    * if the row is **not** stale (``epoch >= current``), the blob is returned
      unchanged and re-stamped at ``current`` (an already-current blob needs no
      rewrite, but the stamp is normalized so a forward stamp is never lowered);
    * if the row **is** stale, the blob is canonicalized via
      :func:`canonicalize_tool_blob` and stamped at ``current`` — it is now known
      to satisfy the current backfill;
    * a ``None`` blob (a nullable surface, e.g. a non-frozen ``sessions`` row)
      passes through as ``None`` and is still stamped ``current``: there is no
      vocabulary to upcast, and the stamp marks the row current so the boot scan
      skips it.

    The caller persists both the returned blob and the returned epoch in the
    same insert, so the row is born current and the partial stale-index stays
    empty for it.
    """

    if blob is None:
        return None, current
    if not is_stale(epoch, current=current):
        return list(blob), current
    return canonicalize_tool_blob(blob), current
