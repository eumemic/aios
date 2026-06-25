# `create_goal` — the canonical reflexive-OPEN verb (terminology of record)

*unify-obligations epic #1516, child #1523. Decision + documentation child (minimal code).*

> **Companion:** the source of record for this decision is the module docstring of
> [`src/aios/tools/goal_management.py`](../../src/aios/tools/goal_management.py); this note is the
> epic-linked design record so the #1414 naming is not re-litigated.

## Decision

The unify-obligations epic retires every goal-specific lifecycle verb **except OPEN** into the
uniform, source-agnostic obligation verbs. What remains goal-named is **exactly one verb**:
**`create_goal`** — the canonical *reflexive-OPEN* verb. It is kept **as-is** (no rename to
`set_goal`, no signature change).

A **GOAL is the reflexive call**: a self-referential **awaited** obligation whose
`caller == servicer == self` (the #1414 self-goal path — `caller={kind:"session", id:<this session>}`,
target = the same session, `awaited=true`). It is a *label* for that one shape of obligation, **not a
subsystem**: there is no separate goal store, gate, or lifecycle. The quiescence guard already refuses
to let a session quiesce past any open awaited obligation, self-goals included.

## Terminology of record (the verb mapping)

| Goal action | Canonical verb(s) | Authority |
|-------------|-------------------|-----------|
| **OPEN** a goal | `create_goal` | the session opens its own reflexive edge |
| **CLOSE** a goal | `return` / `error` (by `goal_id` as `request_id`) | servicer authority, schema-validated |
| **LIST** open goals | `list_obligations` / `list_calls` (origin=self) | — |
| **DROP / CANCEL** a goal | `cancel_call` (by `tool_call_id`) | caller authority |

- **`create_goal`** opens the self-referential awaited edge via `sessions_service.invoke` with
  `target=session_id` and `caller={kind:"session", id:session_id}`. `output_schema` is **REQUIRED**
  (#1512/#1513) — there is no schemaless goal; the schema persists on the `request_opened` frame the
  same way `call_*` carry it, and is the completion contract `return` validates servicer-side. The
  open is a **no-park self-edge** (parking would self-deadlock; the quiescence guard, not a park, holds
  the session) and enforces the per-session open-goal admission cap
  (`Settings.session_open_goals_max`, counted via `_open_self_goals`). It returns a `goal_id`
  (= the edge's `request_id`).
- **Closing** a goal is `return(request_id=<goal_id>, value=…)` (the persisted `output_schema` is
  enforced by `return`'s own schema gate — a non-conforming value is rejected as
  `output_schema_violation` and the goal stays open) or `error(request_id=<goal_id>, message=…)`.
  The self-only `complete_goal`/`fail_goal` were retired (#1518 → #1525).
- **Listing** open goals is the general open-obligation enumeration filtered to self-caller edges:
  `list_obligations` / `list_calls` with `origin=self` (epic children #6/#3). The legacy `list_goals`
  shim is retained only until child #3 folds it in.
- **Dropping** a goal is `cancel_call` by `tool_call_id` (epic child #5). There is no `cancel_goal`.

There is **no** `set_goal`, **no** `cancel_goal`, **no** `complete_goal`/`fail_goal`, **no**
`update_goal`, and **no** long-term `list_goals`. `create_goal` is the only goal-named verb.
Revision stays explicit: `error` the goal, then `create_goal` a revised one — the value is goals that
*don't move*.

## #1414 disposition (declined)

Open issue **#1414** ("set_goal / cancel_goal: self-issued awaited-request 'goals' for always-on
agents") proposed a `set_goal`/`cancel_goal` naming. The chairman-ratified decision in epic #1516 is
to **keep `create_goal` and NOT rename to `set_goal`**, and to **NOT add `cancel_goal`** — dropping a
goal is the uniform `cancel_call` (child #5). #1414 can be closed as **resolved-by-#1516** by the
seat/chairman; this note records that disposition so the naming is not re-litigated.

## Scope of this child

Decision + documentation only. `create_goal`'s behavior ships correct at origin/master `7869686f`
(#1508/#1511 + #1512/#1513) and is unchanged here: the no-park self-edge open, the required
`output_schema`, and the open-goal admission cap all stay exactly as-is. This child only pins the
terminology of record and the #1414 disposition, and cleans up the docstring/description references.

## Sequencing

The description-reference cleanup depends on children #2 (close verbs retired — landed, #1525),
#3 (`list_goals` → `list_obligations`/`list_calls`), and #5 (`cancel_call` exists). This child lands
the decision and the terminology of record now; the docstring/description name the canonical verbs of
record even where #3/#5 have not yet merged, and the legacy `list_goals` shim is explicitly flagged as
transitional until #3 retires it.
