"""Drop the redundant value-enum ``CHECK`` constraints now single-sourced from
the Python ``Literal`` via a typed writer (#1081).

Every persisted enum's value set used to be hand-synced in TWO independently
editable places — the Python ``Literal`` and a DB ``CHECK`` widened in lockstep
by a paired migration. A forgotten paired migration was a *prod-only* runtime
``CHECK`` violation (every insert raises), invisible to CI. This migration
removes the duplication for the columns whose write path now *provably* flows
from a ``Literal``-typed parameter, so the ``Literal`` becomes the single
editable source:

  (A) writer was already ``Literal``-typed and the value flows from it — the
      CHECK was genuinely redundant:
        * ``vault_credentials.auth_type``   (insert ... auth_type: AuthType)
        * ``triggers.last_fire_status``     (record_trigger_fire(status: TriggerFireStatus))
        * ``wf_run_events.type``            (append_run_event(type: WfRunEventType))
        * ``wf_run_signals.kind``           (insert_run_signal(kind: WfRunSignalKind))

  (B) writer was plain ``str`` and is tightened to its ``Literal`` in the same
      change *first*, so the formerly-silent illegal write is now a static type
      error; only *then* is the CHECK redundant:
        * ``wf_runs.status``
              (set_run_status / set_run_terminal / _commit_terminal_and_dispatch
               now ``status: WfRunStatus``)
        * ``session_memory_stores.access``  (insert_session_memory_store(access: Access))
        * ``memory_versions.created_by_type``
              (the ``actor_type: ActorType`` writers, via ``_actor_columns``)

Deliberately NOT dropped (out of scope — these stay pinned to their ``Literal``
by the (C) drift-test that reads the live CHECK):

  * ``memory_versions.operation`` — written as a *hardcoded SQL literal*
    (``'created'``/``'modified'``/``'deleted'``), so no bad param can drift it;
    the CHECK is its only value-set guard, kept and drift-tested.
  * ``memory_versions.redacted_by_type`` — a nullable ``IS NULL OR IN (...)``
    value-enum; kept and drift-tested.
  * ``trigger_runs.status`` — a *wider lifecycle* enum (``pending``/``running``
    plus the ``TriggerFireStatus`` terminals) with no single backing ``Literal``
    and several ``str``/hardcoded writers; not a single-sourced value-enum, so
    out of scope entirely.
  * the ``memory_stores`` *structural* CHECKs (path regex,
    ``content_size_bytes = octet_length(content)``, ``rank BETWEEN 0 AND 7``,
    ``length(instructions) <= 4096``) — not value-enums, untouched.

Pure DDL, no data rewrite. Downgrade re-adds each CHECK with the value set it
held at HEAD (the current ``get_args`` of each ``Literal``); a later widening
of a ``Literal`` would make ``downgrade`` re-create a *narrower* CHECK than the
data, but downgrade is a rollback-to-this-revision operation, so the value set
frozen here is the correct one for that revision.

Revision ID: 0111
Revises: 0110
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0111"
down_revision: str = "0110"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# (constraint_name, table, CHECK body to restore on downgrade)
_DROPPED_CHECKS: tuple[tuple[str, str, str], ...] = (
    # (A) rows
    (
        "vault_credentials_auth_type_check",
        "vault_credentials",
        "auth_type IN ('bearer_header','oauth2_refresh','basic',"
        "'custom_header','environment_variable')",
    ),
    (
        "triggers_last_fire_status_check",
        "triggers",
        "last_fire_status IN ('ok','error','timeout','skipped')",
    ),
    (
        "wf_run_events_type_check",
        "wf_run_events",
        "type IN ('run_started','call_started','call_result','run_completed',"
        "'annotation','frontier_deferred','request_response')",
    ),
    (
        "wf_run_signals_kind_check",
        "wf_run_signals",
        "kind IN ('gate_resume','child_done','cancel','tool_result')",
    ),
    # (B) rows (writers tightened to the Literal in the same change)
    (
        "wf_runs_status_check",
        "wf_runs",
        "status IN ('pending','running','suspended','completed','errored','cancelled')",
    ),
    (
        "session_memory_stores_access_check",
        "session_memory_stores",
        "access IN ('read_only','read_write')",
    ),
    (
        "memory_versions_created_by_type_check",
        "memory_versions",
        "created_by_type IN ('api_actor','session_actor')",
    ),
)


def upgrade() -> None:
    for name, table, _body in _DROPPED_CHECKS:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT {name}")


def downgrade() -> None:
    for name, table, body in _DROPPED_CHECKS:
        op.execute(f"ALTER TABLE {table} ADD CONSTRAINT {name} CHECK ({body})")
