"""Declarative per-column clone policy for ``clone_session``.

Historically ``clone_session`` copied a session with four
``INSERT … SELECT`` statements whose column lists were hand-enumerated.
Nothing tied those lists to the live table schemas: a migration adds a
column and the clone silently drops it (the #1676 defect class — the 0127
class-mass columns were lost on every clone this way, re-opening the #1609
composition-blind-windowing incident class on the exact session shape that
produced it).  Deliberate deviations (fresh ids, reset usage counters, a
new workspace path) were encoded as *omissions*, indistinguishable in the
code from forgotten columns.

This module replaces the hand lists with a **per-column policy** per
relation.  Each column is classified with exactly one :class:`Arm`, and
the INSERT column projection is *generated* from the policy.  A companion
integration test (``tests/integration/test_clone_policy_completeness.py``)
asserts ``policy.keys() == information_schema.columns`` on the migrated
testcontainer schema, so a future migration that adds a column without
classifying it fails CI deterministically before merge — the defect class
becomes *unauthorable* rather than merely guarded.

Copy-by-default is deliberately **not** the default: this table family's
historical failures include must-MINT columns (github-repo / trigger ids,
both global PKs) and must-RESET columns (trigger runtime state, usage
counters).  Every column names its arm explicitly; there is no implicit
fallthrough.

The generated SQL is a plain column projection (NOT a jsonb round-trip):
jsonb would add bytea / timestamptz type-fidelity risk on a table family
that carries both.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto


class Arm(Enum):
    """How a single column is produced on the clone row.

    ``COPY``            — value taken verbatim from the parent row (``s.<col>``).
    ``MINT_ID``         — a freshly minted id, supplied per-row out of band
                          (single param for the one-row sessions insert, an
                          ordinal-join array for the multi-row relations).
    ``REMAP_SESSION``   — the new clone's ``session_id`` (a bound param).
    ``RESET_DEFAULT``   — omitted from the projection so the table DEFAULT
                          (or NULL) applies.  Used for usage counters, runtime
                          state, timestamps, and the deliberate authority
                          resets (``parent_run_id``, ``origin``).
    ``NEW_VALUE``       — a caller-supplied bound param (the workspace path).
    ``MINT_INGEST_TOKEN`` — the source-conditional trigger ingest-token hash:
                          a fresh hash for ``external_event`` rows (the
                          ``triggers_ingest_token_hash`` UNIQUE index forbids
                          copying the parent's), NULL otherwise (the
                          ``triggers_ingest_token_iff_external_event`` CHECK
                          forbids a fresh hash on non-external rows).
    """

    COPY = auto()
    MINT_ID = auto()
    REMAP_SESSION = auto()
    RESET_DEFAULT = auto()
    NEW_VALUE = auto()
    MINT_INGEST_TOKEN = auto()


# Arms that contribute a column to the INSERT projection.  ``RESET_DEFAULT``
# is the sole arm that omits its column (letting the table DEFAULT / NULL win),
# so it is the only one *not* in this set.
_PROJECTED_ARMS = frozenset(
    {
        Arm.COPY,
        Arm.MINT_ID,
        Arm.REMAP_SESSION,
        Arm.NEW_VALUE,
        Arm.MINT_INGEST_TOKEN,
    }
)


@dataclass(frozen=True)
class Projection:
    """The generated ``(columns, select_exprs)`` for one relation's INSERT.

    ``columns`` is the INSERT target list; ``select_exprs`` is the matching
    SELECT list, positionally aligned.  ``RESET_DEFAULT`` columns appear in
    neither — the table DEFAULT supplies them.
    """

    columns: tuple[str, ...]
    select_exprs: tuple[str, ...]

    @property
    def insert_columns_sql(self) -> str:
        return ", ".join(self.columns)

    @property
    def select_list_sql(self) -> str:
        return ", ".join(self.select_exprs)


def build_projection(
    policy: Mapping[str, Arm],
    *,
    source_alias: str,
    new_id_expr: str,
    session_id_param: str,
    new_value_exprs: Mapping[str, str] | None = None,
) -> Projection:
    """Generate the ``(columns, select_exprs)`` for one relation from ``policy``.

    ``source_alias`` is the SQL alias of the parent-row source (``s`` in the
    ordinal-join copies).  ``new_id_expr`` is the SELECT expression producing
    the row's fresh primary key (a bound param for sessions, ``i.id`` for the
    ordinal-join relations).  ``session_id_param`` is the SELECT expression for
    the clone's ``session_id`` (a bound param).  ``new_value_exprs`` maps each
    ``NEW_VALUE`` column to its SELECT expression (e.g. the workspace-path
    param).

    Column *order* follows ``policy`` insertion order so the generated SQL is
    stable and reviewable.
    """
    new_value_exprs = new_value_exprs or {}
    columns: list[str] = []
    exprs: list[str] = []
    for col, arm in policy.items():
        if arm not in _PROJECTED_ARMS:
            continue
        columns.append(col)
        if arm is Arm.COPY:
            # A single-row copy passes ``source_alias=""`` (bare column names,
            # no table alias / ordinal join); the multi-row copies pass ``s``.
            exprs.append(f"{source_alias}.{col}" if source_alias else col)
        elif arm is Arm.MINT_ID:
            exprs.append(new_id_expr)
        elif arm is Arm.REMAP_SESSION:
            exprs.append(session_id_param)
        elif arm is Arm.NEW_VALUE:
            expr = new_value_exprs.get(col)
            if expr is None:
                raise KeyError(f"NEW_VALUE column {col!r} has no expression in new_value_exprs")
            exprs.append(expr)
        elif arm is Arm.MINT_INGEST_TOKEN:
            # Source-conditional: a fresh hash for external_event rows (bound
            # per-row via the ordinal-join array), NULL otherwise.  The unused
            # fresh hash on a non-external row is simply discarded.
            expr = new_value_exprs.get(col)
            if expr is None:
                raise KeyError(
                    f"MINT_INGEST_TOKEN column {col!r} has no expression in new_value_exprs"
                )
            exprs.append(expr)
        else:  # pragma: no cover - exhaustive over Arm
            raise AssertionError(f"unhandled projected arm {arm!r}")
    return Projection(columns=tuple(columns), select_exprs=tuple(exprs))


# ─── sessions ─────────────────────────────────────────────────────────────────
#
# One row per clone.  ``id`` and ``workspace_volume_path`` are bound params;
# everything else is COPY, a deliberate RESET, or a reviewed authority decision.
#
# Authority / isolation columns (the #794 lattice: never widen authority
# through clone):
#   - focal_channel / focal_locked   COPY  (per_chat isolation must follow the
#                                     clone — a lock-less clone escapes the
#                                     switch_channel gate; see #576)
#   - tools / mcp_servers /           COPY  (the frozen surface — surface_of
#     http_servers / surface_frozen         reads it per-session; dropping it
#                                     lets a frozen child's clone read the
#                                     agent's FULL live surface)
#   - model / litellm_extra           COPY  (the model pin / clamped inference
#                                     identity — dropping it un-pins the clone)
#   - outbound_suppression            COPY  (a side-effect gate: a clone must
#                                     not silently un-suppress outbound actions)
#
# Deliberate resets:
#   - input_tokens / output_tokens /  RESET (usage was paid on the parent;
#     cache_*_tokens / cost_microusd        must not be double-counted)
#   - parent_run_id                   RESET (a clone must not become a phantom
#                                     member of the run's sweep/cancel cascade)
#   - origin                          RESET (the table DEFAULT is 'foreground';
#                                     resetting parent_run_id + origin together
#                                     keeps the services/agents.py fail-closed
#                                     run-child guard coherent)
#   - spec_version                    RESET (consistent with the usage reset;
#                                     the clone re-stamps on first provision)
#   - snapshot_ref / snapshot_host /  RESET (the durable-sandbox snapshot
#     snapshot_bytes / snapshot_updated_at  belongs to the parent's workspace;
#                                     the clone gets a fresh workspace path, so
#                                     a copied pointer would resolve the wrong
#                                     volume)
#   - archived_at                     RESET (a clone is born un-archived)
#   - created_at / updated_at         RESET (now())
SESSIONS_POLICY: dict[str, Arm] = {
    "id": Arm.MINT_ID,
    "agent_id": Arm.COPY,
    "environment_id": Arm.COPY,
    "title": Arm.COPY,
    "metadata": Arm.COPY,
    "stop_reason": Arm.COPY,
    "workspace_volume_path": Arm.NEW_VALUE,
    "last_event_seq": Arm.COPY,
    "created_at": Arm.RESET_DEFAULT,
    "updated_at": Arm.RESET_DEFAULT,
    "archived_at": Arm.RESET_DEFAULT,
    "agent_version": Arm.COPY,
    "input_tokens": Arm.RESET_DEFAULT,
    "output_tokens": Arm.RESET_DEFAULT,
    "cache_read_input_tokens": Arm.RESET_DEFAULT,
    "cache_creation_input_tokens": Arm.RESET_DEFAULT,
    "env": Arm.COPY,
    "focal_channel": Arm.COPY,
    "focal_locked": Arm.COPY,
    "account_id": Arm.COPY,
    "parent_run_id": Arm.RESET_DEFAULT,
    "origin": Arm.RESET_DEFAULT,
    "last_reacted_seq": Arm.COPY,
    "open_tool_call_count": Arm.COPY,
    "last_error_seq": Arm.COPY,
    "last_user_seq": Arm.COPY,
    "last_stimulus_seq": Arm.COPY,
    "archive_when_idle": Arm.COPY,
    "spec_version": Arm.RESET_DEFAULT,
    "tools": Arm.COPY,
    "mcp_servers": Arm.COPY,
    "http_servers": Arm.COPY,
    "surface_frozen": Arm.COPY,
    "snapshot_ref": Arm.RESET_DEFAULT,
    "snapshot_host": Arm.RESET_DEFAULT,
    "snapshot_bytes": Arm.RESET_DEFAULT,
    "snapshot_updated_at": Arm.RESET_DEFAULT,
    "cost_microusd": Arm.RESET_DEFAULT,
    "model": Arm.COPY,
    "litellm_extra": Arm.COPY,
    "outbound_suppression": Arm.COPY,
    "tools_vocab_epoch": Arm.COPY,
}


# ─── events ───────────────────────────────────────────────────────────────────
#
# Ordinal-join copy (ids are PRIMARY KEY so they must be minted fresh);
# everything else is preserved verbatim — the context-builder semantics depend
# on a byte-identical next forward step, so this relation is COPY-everywhere
# except the id / session_id remap.  The ``cumulative_*_mass`` +
# ``cumulative_messages`` columns (migration 0127 / #1657) are COPY here: the
# #1676 live drift was exactly their absence, which made the clone's per-class
# running sums restart at ~one message and ran the #1609 R_eff blend
# composition-blind.
EVENTS_POLICY: dict[str, Arm] = {
    "id": Arm.MINT_ID,
    "session_id": Arm.REMAP_SESSION,
    "seq": Arm.COPY,
    "kind": Arm.COPY,
    "data": Arm.COPY,
    "created_at": Arm.COPY,
    "cumulative_tokens": Arm.COPY,
    "orig_channel": Arm.COPY,
    "focal_channel_at_arrival": Arm.COPY,
    "channel": Arm.COPY,
    "role": Arm.COPY,
    "tool_name": Arm.COPY,
    "is_error": Arm.COPY,
    "sender_name": Arm.COPY,
    "account_id": Arm.COPY,
    "cumulative_messages": Arm.COPY,
    "cumulative_text_mass": Arm.COPY,
    "cumulative_tool_result_mass": Arm.COPY,
    "cumulative_thinking_mass": Arm.COPY,
    "cumulative_tool_use_mass": Arm.COPY,
}


# ─── session_github_repositories ─────────────────────────────────────────────
#
# ``id`` is a global PK — minted fresh per clone via the ordinal join.  The
# attachment snapshot is otherwise copied verbatim (a clone snapshots the
# parent's attachment state at clone time, including refs archived after the
# parent attached them — by design).
SESSION_GITHUB_REPOSITORIES_POLICY: dict[str, Arm] = {
    "id": Arm.MINT_ID,
    "session_id": Arm.REMAP_SESSION,
    "rank": Arm.COPY,
    "repo_url": Arm.COPY,
    "mount_path": Arm.COPY,
    "ciphertext": Arm.COPY,
    "nonce": Arm.COPY,
    "created_at": Arm.COPY,
    "updated_at": Arm.COPY,
    "git_user_name": Arm.COPY,
    "git_user_email": Arm.COPY,
    "account_id": Arm.COPY,
}


# ─── session_memory_stores ───────────────────────────────────────────────────
#
# Composite PK (session_id, memory_store_id) — no id to mint; the session_id
# remap is the whole of the key change.  Copied verbatim (same snapshot
# semantics as the github attachments).
SESSION_MEMORY_STORES_POLICY: dict[str, Arm] = {
    "session_id": Arm.REMAP_SESSION,
    "memory_store_id": Arm.COPY,
    "rank": Arm.COPY,
    "access": Arm.COPY,
    "instructions": Arm.COPY,
    "name_at_attach": Arm.COPY,
    "description_at_attach": Arm.COPY,
    "account_id": Arm.COPY,
}


# ─── triggers ─────────────────────────────────────────────────────────────────
#
# ``id`` is a global PK — minted fresh.  The clone inherits source +
# source_spec + action + next_fire so it keeps firing on the parent's cadence,
# but RESETS all runtime state (running_since / last_fire_* /
# consecutive_failures) so it starts with fresh failure counters and no
# in-flight claim.  Carrying the FULL source_spec (not just the pre-rename
# ``schedule`` column) is what lets a clone of a one-shot-trigger owner
# succeed — the pre-rename copy carried ``schedule`` but not ``fire_at`` and
# tripped the XOR CHECK, aborting the whole clone transaction.
#
# ``ingest_token_hash`` is source-conditional (MINT_INGEST_TOKEN): the
# ``triggers_ingest_token_hash`` UNIQUE index forbids copying the parent's
# hash, and the ``triggers_ingest_token_iff_external_event`` CHECK forbids a
# fresh hash on a non-external_event row — so it is a fresh hash iff the row is
# an external_event trigger, NULL otherwise.
TRIGGERS_POLICY: dict[str, Arm] = {
    "id": Arm.MINT_ID,
    "owner_session_id": Arm.REMAP_SESSION,
    "account_id": Arm.COPY,
    "name": Arm.COPY,
    "enabled": Arm.COPY,
    "next_fire": Arm.COPY,
    "running_since": Arm.RESET_DEFAULT,
    "last_fire_at": Arm.RESET_DEFAULT,
    "last_fire_status": Arm.RESET_DEFAULT,
    "consecutive_failures": Arm.RESET_DEFAULT,
    "metadata": Arm.COPY,
    "created_at": Arm.RESET_DEFAULT,
    "updated_at": Arm.RESET_DEFAULT,
    "source": Arm.COPY,
    "source_spec": Arm.COPY,
    "action": Arm.COPY,
    "environment_id": Arm.COPY,
    "ingest_token_hash": Arm.MINT_INGEST_TOKEN,
}


# The relations ``clone_session`` copies, each mapped to its policy.  The
# completeness gate iterates this so a new copied relation is auto-covered and
# an un-copied relation is out of scope by construction (the named TABLE-drift
# non-goal from #1676: this gate closes COLUMN drift, not the arrival of a
# fifth session-attachment table).
CLONE_POLICIES: dict[str, dict[str, Arm]] = {
    "sessions": SESSIONS_POLICY,
    "events": EVENTS_POLICY,
    "session_github_repositories": SESSION_GITHUB_REPOSITORIES_POLICY,
    "session_memory_stores": SESSION_MEMORY_STORES_POLICY,
    "triggers": TRIGGERS_POLICY,
}
