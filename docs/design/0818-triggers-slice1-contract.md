# Design contract — `triggers` slice 1 (#818)

**Status: FROZEN (rev 2).** Schema contract for renaming `session_scheduled_tasks` → `triggers`
and generalizing it into one entity carrying a `source` (what fires it) and an `action` (what runs
at fire time). Transcribe-ready: the migration, models, and query layer are built against the
exact SQL/JSON/Pydantic below.

Provenance: rev 1 was adversarially red-teamed by five lenses (one executing every SQL statement
against a throwaway Postgres 16, including NOTIFY-gate probes and the backfill); 6 confirmed
findings (2 blockers, 4 majors) and the substantive minors are folded in below. Empirically
verified: the full §4 migration sequence runs clean end-to-end; the §5 NOTIFY gate passes all nine
edge probes; the §4.1 backfill round-trips through Pydantic; today's clone-crash on one-shot rows
reproduces verbatim; croniter's occurrence horizon behaves as §7 states.

Prime directive: **additive future**. Nothing here may force a jsonb data-migration when #819's
`workflow` action kind / `run_completion` source land, nor when later kinds (`spawn_session`),
target-version pinning, or cron timezones arrive. Second directive: **zero behavioral regression**
— every backfilled row behaves byte-for-byte as today.

---

## 1. The envelopes

### 1.1 `action` — tagged union, discriminator key `kind`

`kind` is the discriminator (matches the ratified sketch in #818). Stored as **one `action` jsonb
column** with `kind` inside it (the runner is the only reader; SQL never filters on action kind —
see §3 for the source asymmetry rationale). Defaults are **materialized at write time** (the row
is self-describing; the runner carries no defaults knowledge), which is what lets the DB CHECK
require the kind's **first-shipped** keys (see the retro-add rule below).

```jsonc
// Slice-1 member 1 — verbatim today's bash behavior:
{
  "kind": "sandbox_command",
  "command": "tool wake_self '{\"content\":\"poll done\"}'",  // 1..16_384 chars
  "timeout_seconds": 300,        // 1..3600, default 300 — materialized at write
  "max_output_bytes": 65536     // 1024..1_048_576, default 65536 — materialized at write
}

// Slice-1 member 2 — folds in schedule_wake's delivery (renamed from the
// issue-sketch "wake_session"; see decision D1):
{
  "kind": "wake_owner",
  "content": "[Your scheduled wake fired. Reason: check the deploy]"  // 1..16_384 chars
}
```

Field-name note: the issue sketch says `timeout_s`; this contract keeps **`timeout_seconds`** /
**`max_output_bytes`** — they are today's Pydantic field names, column names, and tool-schema
parameter names, so keeping them eliminates a gratuitous rename across the tool surface and makes
"verbatim" auditable by string equality.

`wake_owner` carries **no `session_id`**: the target is implicitly the trigger's owner session
(`owner_session_id`, NOT NULL in this slice). A redundant `session_id` that must equal a column is
a divergence liability, and an *un*constrained one is an uncapped cross-session wake — a bypass of
the `wake_session` tool's depth/rate caps (#373). When slice 4 makes ownership nullable
(operator-owned triggers), an explicit-target wake is a **new kind**, not a new field.

**Additive-growth rules (the contract for future kinds):**

- A new execution behavior is always a **new `kind`**, never a flag/mode on an existing kind.
  Reserved next: `workflow` (#819), `spawn_session` (slice 4 — the Managed-Agents-deployment
  shape: fresh session per fire; per the wire probe it is NOT a variant of `wake_owner`).
- Adding a kind = swap the `triggers_action_shape` CHECK (DROP + ADD CONSTRAINT, one table scan,
  **zero row rewrites**) + add a Pydantic union member + add a `oneOf` branch to the tool schema.
- **Retro-add rule (CHECK scope).** A kind's CHECK requires exactly the keys of the kind's
  **first-shipped** shape. A field retro-added to an existing kind NEVER joins that kind's CHECK:
  its default lives in the Pydantic member model, and the resulting absent-vs-present mixed key
  corpus is deliberate and harmless (the read path validates both to the same value). Promoting a
  retro-added key to CHECK-required would demand exactly the jsonb backfill the prime directive
  forbids — don't.
- **Version-pin reservation:** any kind that references a versioned resource MUST carry
  `<noun>_version: int | null` **from its first-shipped shape**, where `null` = resolve-latest-at-
  fire (float) and an integer = frozen pin (resolve-latest-at-*write*, Anthropic's probed
  pin-and-freeze). e.g. #819: `{"kind":"workflow","workflow_id":"wf_…","input":{…},
  "workflow_version":3}`. The CHECK idiom for such a required-but-nullable key is
  `(action ? 'workflow_version') AND jsonb_typeof(action -> 'workflow_version') IN ('number','null')`
  — the bare `= 'number'` idiom would reject every float-pin row (materialized jsonb `null`), and
  dropping the `?` guard stops requiring presence (NULL satisfies a CHECK; see §2.1).
- **FK-enforced references are the one exception to per-kind-jsonb placement.** A kind field that
  must be FK-enforced (it references a deletable resource) lands as a **nullable first-class FK
  column** plus a per-kind presence/absence conjunct in the action-CHECK swap (existing rows are
  NULL, so the swap is zero-rewrite). Reserved next: `environment_id` (workflow, #819), `agent_id`
  (spawn_session, slice 4).
- Member models use `extra="forbid"` (repo style, fail-hard). This assumes API+worker deploy in
  lockstep (they do — single promote). Consequence, accepted: landing an additive key is one-way
  for app code — rolling back below the key-introducing version makes rows carrying the key
  unreadable. Consistent with forward-only migrations.

### 1.2 `source` — text column + `source_spec` jsonb

Wire shape is a tagged object (gives the discriminated-union 422 at the API boundary); storage
splits the tag into a **`source text` column** (SQL branches on it: the claim loop's cron-advance,
future filtered indexes) plus a **`source_spec jsonb`** holding the rest. Read path reconstructs
`{"kind": source, **source_spec}`.

```jsonc
// wired in slice 1:
{"kind": "cron",     "schedule": "*/5 * * * *"}            // → source='cron',     source_spec={"schedule":"*/5 * * * *"}
{"kind": "one_shot", "fire_at": "2026-06-11T09:00:00Z"}    // → source='one_shot', source_spec={"fire_at":"2026-06-11T09:00:00Z"}
```

- `schedule`: standard 5-field cron, **UTC-only this slice**. Grammar- and occurrence-validated
  at write time only (§7).
- `fire_at`: ISO-8601 UTC string inside the jsonb (the *definition*); the runtime copy lives in
  the `next_fire` column (§3). Past `fire_at` at create is allowed and fires immediately (today's
  documented semantic); any update that leaves an **enabled** row one-shot with a past `fire_at`
  is rejected (today's semantic — see the update matrix in §2.4; this covers more than just
  re-enable).
- **Reserved source values** (vocabulary only, not in any CHECK until wired): `webhook`,
  `connector_inbound`, `run_completion` (#819). Wiring one = constraint swap + union member at the
  schema layer — schema-only, no data migration. **But the fire-dispatch machinery does NOT carry
  over for event-shaped sources:** the slice-1 claim+defer tail (per-trigger `queueing_lock`
  dedup, `running_since` claim gate, task_id-only job payload) is single-flight-coalescing by
  design — correct for cron, where a dropped fire self-heals at the next slot, but silently lossy
  for events, which have no next slot. Event-shaped fires must carry per-event context to the
  runner (an additive optional kwarg on the fire job, e.g. `trigger_run_id` or the firing event's
  id — needed anyway to render #819's input template) and must not dedupe or claim on the bare
  trigger id. Whether they fire job-per-event or inline at the signal is #819 design space; this
  contract only reserves that they do not reuse the slice-1 tail. (Same reservation applies to a
  future manual "fire now".)
- **Timezone is additive later:** a future `"timezone": "America/New_York"` key on the cron spec.
  The CHECK below asserts required-present + foreign-absent, NOT a closed key set, precisely so
  this lands without touching the constraint. (A forward-skewed client sending `timezone` to a
  pre-timezone server gets a clean 422 from `extra="forbid"` — the deliberate alternative to
  silent UTC misinterpretation.)
- **Orthogonality:** any source × any action. The runner dispatches execution on `action.kind`
  ONLY; it branches lifecycle (one-shot delete-before-run vs cron record-after) on `source` ONLY.
  Nothing infers one from the other. A cron `wake_owner` (recurring model wake — the
  deployment-style heartbeat) and a one-shot `sandbox_command` are both legal and tested.

### 1.3 Storage type for `source`: text + CHECK, not a native enum

Repo style (`last_fire_status` is text + CHECK), and additions are constraint swaps rather than
`ALTER TYPE … ADD VALUE` (which has transaction-visibility caveats). The shape CHECK's
`ELSE false` already rejects unknown sources, so there is **one** CHECK, not two.

---

## 2. The integrity layers

Replaces the `schedule`-vs-`fire_at` nullable-XOR (`sched_tasks_schedule_xor_fire_at`, migration
0059) with spec-shape-matches-source. All three layers are part of the ratified contract: typed
422 at the API boundary, CHECK at the DB, diagnostic SELECT in the migration.

### 2.1 DB CHECK constraints

The predicates are defined ONCE as Python string constants in the migration and interpolated into
both `ADD CONSTRAINT` and the validating SELECT, so the two cannot drift. **The outer
`COALESCE(…, false)` is load-bearing** (red-team blocker, empirically proven): `->` on an absent
key returns SQL NULL, `jsonb_typeof` is STRICT so it propagates NULL through the AND-chain and
CASE, and **a NULL CHECK is satisfied** per the SQL standard — without the COALESCE, a row missing
a required key (`one_shot` with `{}`, `sandbox_command` without materialized defaults) inserts
successfully and the validating SELECT is blind to it. With it, both layers reject/flag exactly
those rows while additive extra keys (e.g. a future `timezone`) still pass.

```python
# migrations/versions/0083_triggers_rename_and_action_union.py  (numbered 0083 at land time — master added 0080/0081 first)
SOURCE_SPEC_PREDICATE = """COALESCE((
    CASE source
        WHEN 'cron' THEN
            jsonb_typeof(source_spec -> 'schedule') = 'string'
            AND NOT (source_spec ? 'fire_at')
        WHEN 'one_shot' THEN
            jsonb_typeof(source_spec -> 'fire_at') = 'string'
            AND NOT (source_spec ? 'schedule')
        ELSE false
    END
), false)"""

ACTION_PREDICATE = """COALESCE((
    CASE action ->> 'kind'
        WHEN 'sandbox_command' THEN
            jsonb_typeof(action -> 'command') = 'string'
            AND jsonb_typeof(action -> 'timeout_seconds') = 'number'
            AND jsonb_typeof(action -> 'max_output_bytes') = 'number'
            AND NOT (action ? 'content')
        WHEN 'wake_owner' THEN
            jsonb_typeof(action -> 'content') = 'string'
            AND NOT (action ? 'command')
        ELSE false
    END
), false)"""
```

```sql
ALTER TABLE triggers ADD CONSTRAINT triggers_source_spec_shape CHECK ({SOURCE_SPEC_PREDICATE});
ALTER TABLE triggers ADD CONSTRAINT triggers_action_shape      CHECK ({ACTION_PREDICATE});
```

Notes: when the key IS present, `jsonb_typeof(x -> 'k') = '<type>'` asserts non-json-null and
typed; presence enforcement comes from the COALESCE collapsing the absent-key NULL to `false`.
The `NOT (… ? '…')` clauses are the "foreign keys absent" rule — each spec may not carry the
*other* source's/kind's keys, but the key set is NOT closed (timezone et al. stay additive).
`ELSE false` makes unknown sources/kinds unrepresentable (fail-hard; new values are deliberate
constraint swaps). A NULL `source` or missing `kind` falls to `ELSE false`. Both predicates use
only IMMUTABLE jsonb operators — legal in CHECK. Constraint tests must include the two must-reject
probes the unwrapped predicates silently accepted: `one_shot` + `{}` spec, and `sandbox_command`
missing `timeout_seconds`/`max_output_bytes`.

### 2.2 Pydantic models (`src/aios/models/triggers.py`)

Two structural rules, both red-team-confirmed:

- **All cron validation is write-side.** The union members carry structure only (types, lengths,
  discriminator); `TriggerCreate`/`TriggerUpdate` run the cron grammar + occurrence check in
  `model_validator`s. Rev 1 had the occurrence check as a `CronSource` field validator — but reads
  go through the same union (`TriggerEcho`, query-layer `TypeAdapter`s), so a legally-persisted
  quadrennial cron row (valid under today's grammar-only check) would become **unreadable** after
  migration and 500 every `GET /v1/sessions` via the embedded echoes. The read path must accept
  every row the write path ever accepted. (`SandboxCommandAction`'s `ge`/`le` bounds are safe on
  read: byte-identical since the feature shipped, every persisted row satisfies them, the backfill
  copies verbatim.)
- **Update replaces unions wholesale, with no silent re-defaulting.** `TriggerUpdate.action` uses
  a Replace variant whose optional-at-create fields are **required**: sending
  `{"kind":"sandbox_command","command":…}` without `timeout_seconds` on update is a 422, not a
  silent reset to 300 (today's PATCH leaves omitted fields alone; materialized defaults would have
  inverted that for the same JSON keys). Defaults apply at CREATE only.

```python
from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from croniter import CroniterBadDateError, croniter
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_TRIGGERS_PER_SESSION = 32          # carried verbatim from MAX_SCHEDULED_TASKS_PER_SESSION

DEFAULT_TIMEOUT_SECONDS = 300
MIN_TIMEOUT_SECONDS = 1
MAX_TIMEOUT_SECONDS = 3600
DEFAULT_MAX_OUTPUT_BYTES = 65536
MIN_MAX_OUTPUT_BYTES = 1024
MAX_MAX_OUTPUT_BYTES = 1_048_576
MAX_COMMAND_CHARS = 16_384
MAX_SCHEDULE_CHARS = 128
MAX_NAME_CHARS = 64
MAX_WAKE_CONTENT_CHARS = 16_384        # ≈ today's implicit bound (content rode inside command)

CRON_OCCURRENCE_HORIZON_YEARS = 1

TriggerFireStatus = Literal["ok", "error", "timeout", "skipped"]  # was ScheduledTaskStatus


def _validate_cron_expression(value: str) -> str:
    """WRITE-PATH ONLY (TriggerCreate/TriggerUpdate) — never on read models."""
    if not croniter.is_valid(value):
        raise ValueError(f"invalid cron expression: {value!r}")
    # Occurrence-existence: grammar-valid expressions with no real fire within
    # the horizon (e.g. `0 0 30 2 *`) fail at create instead of sitting
    # silently dead. croniter raises CroniterBadDateError when no match exists
    # within max_years_between_matches (verified against the pinned croniter).
    try:
        croniter(
            value,
            datetime.now(UTC),
            max_years_between_matches=CRON_OCCURRENCE_HORIZON_YEARS,
        ).get_next(datetime)
    except CroniterBadDateError:
        raise ValueError(
            f"cron expression {value!r} produces no occurrence within the next "
            f"{CRON_OCCURRENCE_HORIZON_YEARS} year(s)"
        ) from None
    return value


class CronSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["cron"] = "cron"
    schedule: str = Field(min_length=1, max_length=MAX_SCHEDULE_CHARS)
    # structure-only here — grammar/occurrence checks live on the write models.
    # future additive field: timezone (IANA name; absent = UTC) — out of scope this slice


class OneShotSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["one_shot"] = "one_shot"
    fire_at: datetime

    @field_validator("fire_at")
    @classmethod
    def _validate_fire_at_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError(
                "fire_at must be timezone-aware (e.g. ISO 8601 with a `Z` or explicit "
                "offset) — naive datetimes are ambiguous against the `timestamptz` column"
            )
        return v


TriggerSource = Annotated[CronSource | OneShotSource, Field(discriminator="kind")]


class SandboxCommandAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["sandbox_command"] = "sandbox_command"
    command: str = Field(min_length=1, max_length=MAX_COMMAND_CHARS)
    timeout_seconds: int = Field(
        default=DEFAULT_TIMEOUT_SECONDS, ge=MIN_TIMEOUT_SECONDS, le=MAX_TIMEOUT_SECONDS
    )
    max_output_bytes: int = Field(
        default=DEFAULT_MAX_OUTPUT_BYTES, ge=MIN_MAX_OUTPUT_BYTES, le=MAX_MAX_OUTPUT_BYTES
    )


class SandboxCommandActionReplace(SandboxCommandAction):
    """Update-side variant: optional-at-create fields are REQUIRED, so a
    partial action 422s instead of silently resetting stored values to
    defaults. (Create keeps the defaults for tool ergonomics.)"""
    timeout_seconds: int = Field(ge=MIN_TIMEOUT_SECONDS, le=MAX_TIMEOUT_SECONDS)
    max_output_bytes: int = Field(ge=MIN_MAX_OUTPUT_BYTES, le=MAX_MAX_OUTPUT_BYTES)


class WakeOwnerAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["wake_owner"] = "wake_owner"
    content: str = Field(min_length=1, max_length=MAX_WAKE_CONTENT_CHARS)


TriggerAction = Annotated[SandboxCommandAction | WakeOwnerAction, Field(discriminator="kind")]
TriggerActionReplace = Annotated[
    SandboxCommandActionReplace | WakeOwnerAction, Field(discriminator="kind")
]


class TriggerCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        min_length=1, max_length=MAX_NAME_CHARS, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$"
    )
    source: TriggerSource
    action: TriggerAction
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_cron_write_path(self) -> TriggerCreate:
        if isinstance(self.source, CronSource):
            _validate_cron_expression(self.source.schedule)
        return self


class TriggerUpdate(BaseModel):
    """Update body. `source`/`action` are replaced WHOLESALE when provided (a
    cron↔one-shot or sandbox↔wake conversion is just a different object).
    `None` = leave alone; there is no clear-to-null (both columns are NOT
    NULL). This deletes the EllipsisType sentinel and the merged-XOR
    re-validation — invalid SHAPES are unrepresentable. The next_fire/cap/
    past-fire_at business rules are PORTED, not deleted (§2.4)."""
    model_config = ConfigDict(extra="forbid")
    source: TriggerSource | None = None
    action: TriggerActionReplace | None = None
    enabled: bool | None = None
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_cron_write_path(self) -> TriggerUpdate:
        if isinstance(self.source, CronSource):
            _validate_cron_expression(self.source.schedule)
        return self


class TriggerEcho(BaseModel):
    id: str
    name: str
    source: TriggerSource
    action: TriggerAction
    enabled: bool
    next_fire: datetime | None
    last_fire_at: datetime | None
    last_fire_status: TriggerFireStatus | None
    consecutive_failures: int
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


def compute_next_fire(schedule: str, from_time: datetime) -> datetime:
    ...  # unchanged body (croniter default horizon — NOT the 1y write-time horizon)


def compute_initial_next_fire(source: CronSource | OneShotSource, now: datetime) -> datetime:
    """Cron: next slot strictly after now. One-shot: fire_at verbatim (even if
    past — due immediately, today's semantic)."""
    if isinstance(source, OneShotSource):
        return source.fire_at
    return compute_next_fire(source.schedule, now)
```

Persistence mapping (service/queries layer): `source` column = `spec.source.kind`; `source_spec` =
`spec.source.model_dump(mode="json", exclude={"kind"})`; `action` =
`spec.action.model_dump(mode="json")` (defaults materialized). Read path:
`TypeAdapter(TriggerSource).validate_python({"kind": row["source"], **parse_jsonb(row["source_spec"])})`
and `TypeAdapter(TriggerAction).validate_python(parse_jsonb(row["action"]))` via module-level
`TypeAdapter`s — **structure-only; the read path must accept every row the write path ever
accepted.** (`fire_at` serialization format between Pydantic — `…Z` without trailing zeros — and
the backfill's `to_char` — six-digit microseconds — differs cosmetically; nothing string-compares,
both parse identically; round-trip verified empirically.)

### 2.3 In-migration validating SELECT

Runs after the backfill, **before** `ADD CONSTRAINT`, sharing the predicate constants (which carry
the COALESCE — without it this SELECT is blind to missing-key rows, proven empirically). Its value
over `ADD CONSTRAINT`'s own full-table validation is the diagnostic: it names offending rows
instead of aborting on the first.

```python
bad = op.get_bind().execute(
    sa.text(f"""
        SELECT id, source, source_spec, action
        FROM triggers
        WHERE NOT {SOURCE_SPEC_PREDICATE} OR NOT {ACTION_PREDICATE}
        LIMIT 20
    """)
).fetchall()
if bad:
    raise RuntimeError(
        f"trigger backfill produced rows violating the shape contract: {bad!r}"
    )
```

(No `@dataclass` anywhere in the migration — known alembic synthetic-module crash; rows are plain
`Row` tuples. The f-string interpolates only module-level developer-controlled constants — keep
them module-level so that property stays structurally true.)

### 2.4 Update semantics — carried verbatim from today's `update_task`

The shape simplification (§2.2) does NOT simplify the business rules. Keyed on the **merged final
state** (update fields merged over the current row):

| Update | `next_fire` | Cap check | Past-`fire_at` rule |
|---|---|---|---|
| `enabled` true→false | := NULL | — | — |
| `enabled` false→true (re-enable) | recompute from merged source | **per-account enabled-rows cap under the advisory lock** (disabled rows don't consume a slot) | reject if merged source is one-shot with `fire_at <= now()` (typed ValidationError) |
| `source` replaced (row enabled in final state) | recompute from new source — wholesale replacement **always** recomputes | no re-check (already-enabled rows hold their slot) | same rejection — applies here too, not just re-enable (today's behavior; rev 1 understated this) |
| `action` / `metadata` / no-op | untouched | — | — |
| any update incl. no-op | — | — | `updated_at` always bumps (external `updated_at > since` pollers) |

Past `fire_at` remains legal at CREATE and on rows whose final state is disabled. Create caps,
unchanged: per-session = COUNT(*) of ALL rows (enabled + disabled) vs `MAX_TRIGGERS_PER_SESSION`;
per-account = COUNT of enabled rows on non-archived sessions vs `triggers_per_account_max`; both
serialized by `pg_advisory_xact_lock(hashtextextended('aios_st_cap:' || account_id, 0))` — **the
lock key text `'aios_st_cap:'` stays byte-identical** (cross-process string; renaming buys
nothing — same stance as the NOTIFY channel). Regression test to add: update
`source={kind:'one_shot', fire_at:<past>}` on an ENABLED cron trigger → 422 (no existing test
covers that cell).

---

## 3. Storage layout — columns vs jsonb

Hard constraint honored: the scheduler hot path (`fetch_and_claim_due_triggers`,
`fetch_next_trigger_event`) reads **only first-class columns** (`next_fire`, `enabled`,
`running_since` + the `sessions.archived_at` join); the partial index
`triggers_due ON triggers (next_fire) WHERE enabled AND next_fire IS NOT NULL` is unchanged in
shape. The single jsonb access on the hot path is the post-claim cron advance — the claim SELECT
projects **`source_spec ->> 'schedule' AS schedule`** (text extraction in SQL, keeping the Python
advance loop byte-identical and avoiding asyncpg's jsonb-as-raw-string decode trap) — per
*claimed* row (≤100/tick), negligible.

**Contract invariant (load-bearing for #819's reactive sources):** the claim and MIN queries keep
their explicit `next_fire IS NOT NULL` WHERE clause — a row without `next_fire` is unschedulable
by the tick **by predicate**, not merely by index shape.

| Field | Storage | Why |
|---|---|---|
| `id` | column (PK) | unchanged; new rows mint `trig_` (D6), old `sched_` ids remain valid |
| `owner_session_id` | column, FK→sessions ON DELETE CASCADE, NOT NULL | **renamed from `session_id`** now, while the table rename already forces a full query pass; slice 4 makes it nullable (operator-owned) — name is right from day one |
| `account_id` | column NOT NULL | unchanged (denormalized tenant scope) |
| `name` | column NOT NULL, UNIQUE(owner_session_id, name), CHECK pattern | unchanged addressable identifier |
| `source` | **column** text NOT NULL | SQL branches on it (claim-loop cron advance); discriminator-as-column is the tagged-union-in-SQL norm |
| `source_spec` | **jsonb** NOT NULL | definition-only; per-source shape enforced by CHECK; additive keys (timezone) free |
| `action` | **jsonb** NOT NULL (kind inside) | definition-only, opaque to SQL queries (the shape CHECK may branch on kind) — the runner is the sole reader, post-fetch in Python. Keeping `command`/`timeout_seconds`/`max_output_bytes` as columns would force nullable columns + a command-present-iff-sandbox CHECK, i.e. recreate the XOR disease this slice exists to kill. This call moves ~half the query-layer diff: `add/update_trigger` take one validated jsonb instead of five scalars |
| `enabled` | column NOT NULL | hot path + partial-index predicate |
| `next_fire` | column timestamptz | **hot path** — claim WHERE + MIN(); runtime copy of the definition |
| `running_since` | column timestamptz | hot path — overlap prevention + stale recovery |
| `last_fire_at`, `last_fire_status`, `consecutive_failures` | columns | runtime fire-outcome trio, written per fire; status CHECK (`ok/error/timeout/skipped`) unchanged |
| `metadata` | jsonb NOT NULL | unchanged (CLI `wake:`-rendering tag lives here) |
| `created_at`, `updated_at` | columns | unchanged |
| ~~`schedule`, `fire_at`~~ | **dropped** → `source_spec` | definition fields |
| ~~`command`, `timeout_seconds`, `max_output_bytes`~~ | **dropped** → `action` | definition fields |

Query-layer shapes pinned: `TriggerRow` NamedTuple carries `source: str` + `source_spec: dict`
**raw** (the scheduler/runner branch lifecycle on the source string) and `action: TriggerAction`
**typed** via the module-level TypeAdapter (the runner dispatches on `action.kind`), plus the
unchanged runtime/identity fields incl. `session_archived_at`. List ordering preserved:
`ORDER BY created_at` (`list_triggers`) and `ORDER BY owner_session_id, created_at` (batch echo
loader) — echo stability and CLI output depend on it. The renamed `unscoped_get_trigger_row` stays
**intentionally unscoped** — the single audited cross-tenant read (the fire-job runs cross-tenant
in the worker; each row carries `account_id` denormalized). Do not "fix" it with account scoping.

**Slice-4 obligations (recorded now so slice 4 stays additive; no slice-1 code):** when
`owner_session_id` goes nullable: (1) NULL-owner name addressing needs a partial unique index
`(account_id, name) WHERE owner_session_id IS NULL` — never `NULLS NOT DISTINCT` on the existing
UNIQUE (cross-tenant name collision); (2) the `JOIN sessions` hot queries become LEFT JOINs with
the archived-at predicate guarded on owner presence — **four** of them since slice 2: the claim,
the MIN, the unscoped getter, and the run_completion completion matcher
(`insert_run_completion_fires`); (3) `triggers_action_shape` gains `owner_session_id IS NOT NULL`
conjuncts for `sandbox_command` and `wake_owner` in the same constraint swap (zero rewrites).
Slice-2 additions to this list: (4) the
TriggerRow `session_parent_run_id` projection (the timer-fire lineage) rides the same JOINs as (2); (5)
`triggers_environment_id_iff_workflow` swaps to the `IN ('workflow','spawn_session')` form if
`spawn_session` also binds an environment; (6) owner-NULL `workflow` fires degrade to
`launcher_session_id=None` = UNATTENUATED operator authority via `create_run`'s existing operator
path — semantically right for operator-owned triggers, but the slice-4 review must make that an
explicit decision, not an accident of the threading.

---

## 4. Migration 0083 — exact sequence + backfill

One alembic migration, one transaction. The full sequence below was executed end-to-end against a
scratch Postgres 16 with synthetic rows covering cron / one-shot / schedule_wake-style / disabled /
in-flight shapes — every statement runs clean, and the auto-generated constraint names in step 3
were verified against the real catalog.

```text
 1. ALTER TABLE session_scheduled_tasks RENAME TO triggers
 2. ALTER TABLE triggers RENAME COLUMN session_id TO owner_session_id
 3. Hygiene renames (catalog-local, zero-risk; names verified on PG16 — re-confirm via \d):
      ALTER INDEX sched_tasks_by_session RENAME TO triggers_by_owner_session
      ALTER INDEX sched_tasks_due        RENAME TO triggers_due
      ALTER TABLE triggers RENAME CONSTRAINT session_scheduled_tasks_pkey
                                          TO triggers_pkey
      ALTER TABLE triggers RENAME CONSTRAINT session_scheduled_tasks_session_id_fkey
                                          TO triggers_owner_session_id_fkey
      ALTER TABLE triggers RENAME CONSTRAINT session_scheduled_tasks_session_id_name_key
                                          TO triggers_owner_session_id_name_key
      ALTER TABLE triggers RENAME CONSTRAINT session_scheduled_tasks_name_check
                                          TO triggers_name_check
      ALTER TABLE triggers RENAME CONSTRAINT session_scheduled_tasks_last_fire_status_check
                                          TO triggers_last_fire_status_check
 4. ALTER TABLE triggers
        ADD COLUMN source text,
        ADD COLUMN source_spec jsonb,
        ADD COLUMN action jsonb
 5. CREATE OR REPLACE FUNCTION notify_scheduled_tasks_due()  -- NEW body, SAME name (§5)
 6. DROP TRIGGER session_scheduled_tasks_notify ON triggers;
    CREATE TRIGGER triggers_notify AFTER INSERT OR UPDATE OR DELETE ON triggers
        FOR EACH ROW EXECUTE FUNCTION notify_scheduled_tasks_due()
 7. Backfill UPDATE (below) — fires the new NOTIFY function; pg_notify queues until
    commit; the scheduler wakes once post-migration and recomputes MIN. Harmless.
 8. Validating SELECT (§2.3) — fail hard with offending ids
 9. ALTER TABLE triggers
        ALTER COLUMN source SET NOT NULL,
        ALTER COLUMN source_spec SET NOT NULL,
        ALTER COLUMN action SET NOT NULL
10. ADD CONSTRAINT triggers_source_spec_shape …; ADD CONSTRAINT triggers_action_shape …
11. ALTER TABLE triggers DROP CONSTRAINT sched_tasks_schedule_xor_fire_at;
    ALTER TABLE triggers
        DROP COLUMN schedule, DROP COLUMN fire_at, DROP COLUMN command,
        DROP COLUMN timeout_seconds, DROP COLUMN max_output_bytes
12. Agent tool-name rewrite in agents.tools + agent_versions.tools jsonb (§6.2)
```

### 4.1 Backfill mapping (step 7)

Every existing row → `action = sandbox_command` **verbatim** — including rows `schedule_wake`
created (their command is `tool wake_self '…'` bash; behaviorally identical to today, so
`wake_owner` is opt-in going forward, never backfilled):

```sql
UPDATE triggers SET
    source = CASE WHEN schedule IS NOT NULL THEN 'cron' ELSE 'one_shot' END,
    source_spec = CASE
        WHEN schedule IS NOT NULL THEN jsonb_build_object('schedule', schedule)
        ELSE jsonb_build_object(
            'fire_at',
            to_char(fire_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')
        )
    END,
    action = jsonb_build_object(
        'kind', 'sandbox_command',
        'command', command,
        'timeout_seconds', timeout_seconds,
        'max_output_bytes', max_output_bytes
    )
```

The 0059 XOR CHECK guarantees exactly one of `schedule`/`fire_at` is set on every pre-migration
row, so the CASE is total; `command`/`timeout_seconds`/`max_output_bytes` are NOT NULL since 0058,
so the action object never gets nulls. Zero semantic change table:

| Today | After backfill |
|---|---|
| `schedule='*/5 * * * *'`, `command=C`, timeouts T/B | `source='cron'`, `source_spec={"schedule":"*/5 * * * *"}`, `action={kind:sandbox_command, command:C, timeout_seconds:T, max_output_bytes:B}` |
| `fire_at=2026-06-12T08:00:00Z`, `command=C` | `source='one_shot'`, `source_spec={"fire_at":"2026-06-12T08:00:00.000000Z"}`, same action |
| `next_fire`, `running_since`, `last_fire_*`, `consecutive_failures`, `enabled`, `metadata` | untouched |

Pre-existing cron rows are NOT re-validated against the §7 occurrence horizon (write-time-only
rule; the read path accepts them). Optional, flagged for sign-off rather than included: a
non-fatal diagnostic SELECT listing legacy cron rows with no occurrence in the next year, for
operator visibility.

### 4.2 Downgrade

Reconstruct old columns from jsonb (mechanical reverse), **fail hard if any `wake_owner` row
exists** (it has no `command`; inventing one would be a silent lie):

```python
n = bind.execute(sa.text(
    "SELECT count(*) FROM triggers WHERE action->>'kind' <> 'sandbox_command'"
)).scalar()
if n:
    raise RuntimeError(f"cannot downgrade: {n} non-sandbox_command trigger rows")
```

then re-add `schedule`/`fire_at`/`command`/`timeout_seconds`/`max_output_bytes` from
`source_spec`/`action`, restore the XOR CHECK + 0059 trigger body, drop the new columns, reverse
the renames. (Repo precedent allows partial downgrades — 0059 leaves `schedule` nullable — but
this one happens to be fully reconstructible for sandbox-only data.)

---

## 5. The NOTIFY landmine — rewritten trigger, byte-identical channel + function name

Verified in code: migration 0059's gate references `OLD.schedule`/`NEW.schedule` and
`OLD.fire_at`/`NEW.fire_at` — exactly the columns step 11 drops. Without the rewrite, the failure
is **loud and total** (empirically verified): plpgsql is late-bound, so after the column drop
every UPDATE on the table errors with `record "old" has no field "schedule"` — claims,
record-fire, enable/disable all break outright. (Rev 1 called this "silent degradation to
hourly"; the real mode is worse and more obvious. The rewrite is mandatory for basic operation,
not just for cadence.) **Channel string `aios_scheduled_tasks_due` and function name
`notify_scheduled_tasks_due()` stay byte-identical** (`db/listen.py: SCHEDULED_TASKS_DUE_CHANNEL`;
renaming buys nothing and opens a deploy-window where an old worker listens on a channel nothing
notifies; the Python *helper* `listen_for_scheduled_tasks_due` → `listen_for_triggers_due` is a
free in-process rename).

Faithful translation of the 0059 gate — `schedule`/`fire_at` watches become `source`/`source_spec`
watches; the load-bearing runner-clear edge (short-period cron cadence) and the INSERT/DELETE arms
are preserved verbatim. All nine gate probes pass empirically (source_spec edit / enabled flip /
runner-clear / INSERT / DELETE notify; action-only edit / metadata edit / claim write /
identical-value write don't):

```sql
CREATE OR REPLACE FUNCTION notify_scheduled_tasks_due() RETURNS TRIGGER AS $$
BEGIN
    IF (TG_OP = 'INSERT') THEN
        PERFORM pg_notify('aios_scheduled_tasks_due', NEW.id);
    ELSIF (TG_OP = 'UPDATE') THEN
        IF (
            OLD.source IS DISTINCT FROM NEW.source
            OR OLD.source_spec IS DISTINCT FROM NEW.source_spec
            OR OLD.enabled IS DISTINCT FROM NEW.enabled
            OR (OLD.running_since IS NOT NULL AND NEW.running_since IS NULL)
        ) THEN
            PERFORM pg_notify('aios_scheduled_tasks_due', NEW.id);
        END IF;
    ELSIF (TG_OP = 'DELETE') THEN
        PERFORM pg_notify('aios_scheduled_tasks_due', OLD.id);
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;
```

Equivalence audit: definition edits (schedule/fire_at) → `source_spec` distinct → notify (as
today). cron↔one-shot conversion → `source` distinct → notify. enable/disable → notify (as
today). Runner-clear (`record_trigger_fire` sets `running_since=NULL`) → notify (as today).
`action`-only updates (command/timeout edits) → **no** notify — matching 0059, which never gated
on `command`/`timeout_seconds`. Post-claim `next_fire` advance → no notify (as today; gate
deliberately excludes it).

---

## 6. Cross-cutting fixes the migration/code MUST carry

### 6.1 `clone_session` (pre-existing latent bug — NOT in #818's file list)

`db/queries/__init__.py:1917-1935`: the clone INSERT copies `schedule` but not `fire_at`. Cloning
a session that owns a one-shot row inserts a both-NULL row → trips the 0059 XOR CHECK → **the
entire clone transaction aborts today** (reproduced empirically; the bug is a clone *crash*, not a
quietly-invalid row — the CHECK already protects the data). Fix in this slice (the INSERT is being
rewritten anyway):

```sql
INSERT INTO triggers (
    id, owner_session_id, account_id, name, source, source_spec, action,
    enabled, next_fire, metadata
)
SELECT i.id, $2, s.account_id, s.name, s.source, s.source_spec, s.action,
       s.enabled, s.next_fire, s.metadata
  FROM (
    SELECT *, row_number() OVER (ORDER BY created_at) AS rn
      FROM triggers WHERE owner_session_id = $1
  ) s
  JOIN unnest($3::text[]) WITH ORDINALITY AS i(id, rn) USING (rn)
```

Runtime state stays reset (running_since/last_fire_*/consecutive_failures default), `next_fire`
copied — the documented clone principle, now correct for one-shots too. (`clone_session` is
same-account-gated, so the copied denormalized `account_id` equals the destination account — no
tenant leak.) **New regression test:** clone a session owning a one-shot trigger; assert the clone
succeeds and carries an identical `source_spec`.

### 6.2 Agent tool-name rewrite (NOT in #818's file list)

`ToolSpec.type` is validated against `BuiltinToolType` on every agent read. Renaming the Literal
members without rewriting stored configs bricks any agent that declares them (validation error on
load). Migration step 12, applied to BOTH `agents.tools` AND `agent_versions.tools`
(order-preserving — tool order feeds the model schema and prompt cache):

```sql
UPDATE agents SET tools = (
    SELECT jsonb_agg(
        CASE elem ->> 'type'
            WHEN 'schedule_task_add'    THEN jsonb_set(elem, '{type}', '"trigger_create"')
            WHEN 'schedule_task_remove' THEN jsonb_set(elem, '{type}', '"trigger_remove"')
            WHEN 'schedule_task_update' THEN jsonb_set(elem, '{type}', '"trigger_update"')
            WHEN 'schedule_task_list'   THEN jsonb_set(elem, '{type}', '"trigger_list"')
            ELSE elem
        END ORDER BY ord
    )
    FROM jsonb_array_elements(tools) WITH ORDINALITY AS t(elem, ord)
)
WHERE tools::text LIKE '%schedule_task_%';
-- identical statement against agent_versions
```

**The WHERE guard is load-bearing correctness, not an updated_at nicety** (empirically verified):
`jsonb_agg` over an empty set returns NULL, so without the WHERE an agent with `tools = '[]'`
would have its NOT NULL `tools` column nulled and abort the migration. The CASE keys on `type`,
which for built-ins IS the model-facing name and for custom/mcp_toolset tools is the discriminator
(`"custom"`/`"mcp_toolset"`) — a custom tool whose `name` happens to be `"schedule_task_add"` has
`type="custom"` and is correctly left untouched; do NOT extend this rewrite to match on `name`.
Migration-equivalence test rows: an agent with empty tools, and one carrying a custom tool named
`schedule_task_add` — both byte-identical after the rewrite.

The **event log is NOT rewritten** (append-only; monotonicity invariant). Historical `tool_call`
events keep the old names — they're raw history the model tolerates.

### 6.3 `BuiltinToolType` (`models/agents.py`)

`"schedule_task_add" | "schedule_task_remove" | "schedule_task_update"` →
`"trigger_create" | "trigger_remove" | "trigger_update"` **plus `"trigger_list"`** — verified:
`tools/schedule_task_list.py` registers a handler today, but the Literal omits it, so no agent can
declare it. Fixed as part of the rename. `"schedule_wake"`, `"wake_session"`, `"wake_self"` stay.

---

## 7. Cron validation hardening

In §2.2's `_validate_cron_expression`, **write-path only** (`TriggerCreate`/`TriggerUpdate`
model_validators — never on read models or the query-layer TypeAdapters; see §2.2 for why): after
`croniter.is_valid`, require ≥1 real occurrence within **1 year**
(`max_years_between_matches=1`; raises `CroniterBadDateError` — verified empirically against the
pinned croniter). Fail shape: `ValueError` → Pydantic → typed 422 / raw tool error:
`cron expression '0 0 30 2 *' produces no occurrence within the next 1 year(s)`.

Semantics, stated precisely:

- The check is **relative to the create/update call time**: `0 0 29 2 *` (quadrennial leap-day)
  is rejected in June 2026 (next occurrence ~1.7y away) but accepted during the 12 months before
  each leap day. Matches the probed reference-platform behavior; a schedule firing less often than
  annually is presumed dead.
- It applies to any **newly-submitted** source object on update too — re-sending an unchanged rare
  cron can therefore 422; accepted (callers omit `source` to leave it alone).
- Pre-existing rows are never re-validated (read path is structure-only).
- DoS: empirically negative — worst-case adversarial 5-field expressions validate in ~0.3 ms under
  the 1-year horizon + `MAX_SCHEDULE_CHARS=128`; safe on the request thread. Don't re-litigate.
- `compute_next_fire` (steady-state advance) keeps croniter's default horizon.

---

## 8. Tool surface

Granular ops only (per #270). All four registered names + transports carry over; parameter schemas
express the unions as `oneOf` with `const` kinds, so #819's `workflow` kind is one added branch.

| Tool | Parameters (JSON Schema sketch) | Notes |
|---|---|---|
| `trigger_create` | `{name, source: oneOf[{kind:'cron', schedule}, {kind:'one_shot', fire_at}], action: oneOf[{kind:'sandbox_command', command, timeout_seconds?, max_output_bytes?}, {kind:'wake_owner', content}], enabled?, metadata?}` — required `[name, source, action]`, `additionalProperties:false` throughout | handler: `TriggerCreate.model_validate(arguments)` → `services.triggers.add_trigger` |
| `trigger_remove` | `{name}` | unchanged semantics |
| `trigger_update` | `{name, source?, action?, enabled?, metadata?}` — source/action replace wholesale; the sandbox_command branch marks `command`, `timeout_seconds`, `max_output_bytes` ALL required (§2.2 Replace variant). Description states: "source/action replace the stored object wholesale — send the complete object (fetch current values via trigger_list)" | conversion = send a different object |
| `trigger_list` | `{}` | now declarable (§6.3) |
| `schedule_wake` | **unchanged params** (`delay_seconds`/`at`/`tz`/`reason`, dateparser NL parsing, max-delay clamp) | becomes sugar: emits `trigger_create(source={kind:'one_shot', fire_at}, action={kind:'wake_owner', content:'[Your scheduled wake fired. Reason: …]'}, name='wake-…', metadata={kind:'wake', reason})`. Result payload: `{scheduled, trigger_id, name, fire_at, reason}` (`task_id` → `trigger_id`); description's `schedule_task_*` cross-references become `trigger_*`. The CLI's `wake: <reason>` rendering keys off `metadata.kind` — preserved |

Tool descriptions follow the runtime-vocabulary rule (no issue numbers); `sandbox_command`'s
description keeps the `tool wake_self '…'` escalation guidance verbatim from `schedule_task_add`.
Test to pin: `wake_owner` triggers route through the same advisory-locked COUNT+INSERT caps as
`sandbox_command` (per-session 32 / per-account enabled cap) — so a future "self-delivery is
cheap, skip the cap" refactor can't silently remove the only standing-row bound.

Behavior delta, deliberate and documented: **new** `schedule_wake` calls deliver in-worker
(append + defer_wake) instead of provisioning a sandbox to run `tool wake_self` — which also
removes today's trap where a scheduled wake 404s at fire time if `wake_self` isn't declared on the
agent. Old rows are untouched (backfilled as `sandbox_command`) — zero regression on existing
data; strictly-better reliability going forward. The delivered marker string is byte-identical
(the new path executes the `wake_self` handler body with the same pre-formatted content). Nothing
else read the old wake rows' `timeout_seconds=30` (verified).

### 8.1 Runner routing (`harness/trigger_runner.py`, rename-and-extend of `scheduled_task_runner.py`)

Two orthogonal branches:

- **Lifecycle on `source`** (unchanged semantics, every cell of today's matrix carried):
  `skip_deleted` (row gone between claim and execute → log, exit) unchanged; `skip_archived` and
  `skip_disabled` keep their one-shot-DELETE vs cron-record-skip split; `one_shot` → DELETE row
  before executing (at-most-once, crash loses the fire rather than duplicating); `cron` →
  `record_trigger_fire` after, auto-disable at 5 consecutive failures. The auto-disable surface
  message MUST append **after** the record/disable transaction commits (pool-deadlock rationale,
  today's comment carries over). Both failure-marker strings stay **byte-identical** —
  `[Scheduled wake '<name>' failed to deliver: <detail>]` and `[Scheduled task '<name>'
  auto-disabled after 5 consecutive failures: <detail>]` are runtime strings that backfilled rows
  keep producing ("Scheduled wake/task" remains truthful runtime vocabulary).
- **Execution on `action.kind`**:
  - `sandbox_command` → `sandbox_registry.get_or_provision` + exec, byte-identical to today
    (statuses ok/error/timeout, stderr-tail summary, one-shot failure surfacing).
  - `wake_owner` → **self-delivery path**: `sessions_service.append_user_message(pool,
    owner_session_id, action.content, account_id=…)` + `defer_wake(cause="message")` — the
    `wake_self` handler body executed in-worker. Explicitly NOT the `wake_session` tool handler:
    no depth counter, no per-pair rate limit, no cross-session reach (the #373 caps exist for
    cross-session graphs; a trigger waking its own owner is the `wake_self`-class primitive,
    already uncapped today via cron bash — red-team confirmed: no capability delta, only
    cheaper/more reliable). No wake_depth/wake_source metadata stamped — correct by design (this
    is self-delivery, not a chain link). No sandbox, no broker. Statuses: ok/error (timeout N/A).
    One-shot `wake_owner` failure = the append itself failed (DB-level), so the
    `_surface_one_shot_failure` machinery (which appends through the same path) is skipped — log
    and exit; cron `wake_owner` failures use the standard counter/auto-disable, whose surfacing is
    best-effort through the same append path (acceptable — `_surface_failure` already swallows).

### 8.2 API routes + CLI porcelain

Routes (D2): `GET /v1/sessions/{id}/triggers` → 200 `ListResponse[TriggerEcho]`
(`list_triggers`); `POST` → 201 `TriggerEcho` (`create_trigger`);
`DELETE /v1/sessions/{id}/triggers/{name}` → 204 (`delete_trigger`);
**`PUT` `/v1/sessions/{id}/triggers/{name}`** → 200 `TriggerEcho` (`update_trigger`) — today's
update route is PUT (not PATCH), and PUT is now actually truthful for wholesale source/action
replacement. The CLI `@covers` decorator strings follow these operation_ids; the
`tests/unit/test_cli_coverage.py` drift guard enforces the mapping against openapi.json.

CLI (`aios sessions triggers …`): `list` columns = name, source cell (cron expr or one-shot time;
`wake: <reason>` synthesized from `metadata.kind`), action cell (kind + command/content preview),
enabled, next_fire, last status. `add` keeps convenience flags for the common shapes —
`--cron <expr>` / `--at <iso>` for source, `--command <bash>` / `--wake-content <text>` for action
— with `--file/--stdin/--data` for full JSON. `update` drops per-field flags (no
`--timeout-seconds` etc.) in favor of the JSON payload path (thin-wire, server-validates — repo
CLI philosophy), keeping `--enabled/--disabled` since `enabled` remains top-level granular. The
sub-app help text is rewritten without the `#636` issue reference (runtime-vocabulary rule).

---

## 9. Decision table

| # | Decision | Recommendation | Rationale |
|---|---|---|---|
| D1 | Action-kind name vs #373 collision | **`wake_owner`** (not `wake_session`) | The cross-session `wake_session` tool (caps, tests, BuiltinToolType) stays untouched; an identically-named action with *different* cap semantics is a model-facing footgun and a security ambiguity. `wake_owner` states the target; slice-4's fresh-session shape is a separate `spawn_session` kind per the wire probe |
| D2 | Session wire field | **Rename `scheduled_tasks` → `triggers`** on `SessionCreate` + `Session` now | This slice already forces openapi+SDK regen (routes/operation_ids/model renames, §8.2) — one coordinated break instead of two. Cost: CLI subcommand group + console (aios-ui BFF) must follow; surface is low-traffic ("built-but-unused in prod" per #818). SDK regen of the discriminated unions has working precedent (Session.resources already round-trips a discriminator union through openapi-python-client). Don't-deprecate-delete |
| D3 | Per-fire audit table | **Defer** (not slice 1) | Zero-regression scope. The reservation is structural: `source` is a first-class column, so a later `trigger_runs` table stamps `trigger_context` by echoing it (plus `manual` for fire-now) at claim time — additive. Two constraints recorded now: `trigger_runs` is also the natural per-event claim/dedup carrier for event-shaped sources (§1.2) — land it in or immediately before #819; and it must reference triggers by plain id + denormalized context, NOT an enforcing FK (one-shot rows are deleted pre-execution; a cascade would eat the audit, a RESTRICT would block at-most-once) |
| D4 | `sandbox_command` permanent or transitional? | **Permanent** | It's the zero-interpreter, zero-model-token poller — structurally cheaper than #819's workflow action (which pays interpreter spawn per fire, per #819's own caveat). "Transitional" in the chairman thread reads as *no longer the only kind*, not *scheduled for removal*. Revisit only if workflow fires become as cheap |
| D5 | procrastinate task name | **Rename `harness.run_scheduled_task` → `harness.run_trigger`** (+ `queueing_lock="trigger:{id}"`) | Delete-don't-deprecate. Deploy-window caveat: a job enqueued pre-restart under the old name fails task-lookup; the claimed row recovers via stale-recovery ≈2h later (one late fire, worst case; window sub-second on lockstep drain-and-restart). Scope note: the queueing_lock dedup + task_id-only payload apply to **scheduler-tick-originated fires only** — event-shaped sources and manual fire-now must not reuse them (§1.2) |
| D6 | ID prefix | **`TRIGGER = "trig"`** for new rows; keep `"sched"` in `_PREFIXES` (legacy-parse comment) | IDs are opaque text PKs; mixed prefixes have precedent (`bnd_` backfill note in ids.py). Keeps `split_id` working on old ids without a data migration |
| D7 | Settings field | **Rename `scheduled_tasks_per_account_max` → `triggers_per_account_max`** (`AIOS_TRIGGERS_PER_ACCOUNT_MAX`); `schedule_wake_max_delay_seconds` unchanged; advisory-lock key text `'aios_st_cap:'` **unchanged** (cross-process string, §2.4) | Consistency; **deploy gate**: grep eumemic-ops for the old env var before promote — a silently-ignored old var would revert prod to the default cap |
| D8 | `source` storage | text + CHECK, not native enum | Repo norm; constraint-swap additions; one CHECK total (shape CASE `ELSE false` subsumes value validation) |
| D9 | Beyond ship-the-table? | **No** new capability in slice 1 (no `trigger_runs`, no manual-fire endpoint, no `spawn_session`) — but the slice DOES carry the three correctness fixes that the rename itself exposes (clone INSERT §6.1, agent-jsonb rewrite §6.2, `trigger_list` Literal §6.3) | Broken-windows: all three are flagged by this design pass and sit directly on the renamed surface |

---

## 10. Corrected file-impact list (vs #818's "Files")

#818 corrections confirmed in-tree: `scheduled_task_runner.py` already exists (rename-and-extend,
not new); the slice also touches `api/routers/sessions.py` (paths + operation_ids → openapi+SDK
regen), `db/listen.py`, `models/sessions.py`, `models/agents.py`, plus the items below that #818
missed entirely (CLI, config, ids, services/sessions, harness/tasks, wake_self description, clone
fix, agent-jsonb rewrite).

**Migration**
- `migrations/versions/0083_triggers_rename_and_action_union.py` (new; §4 sequence incl. §6.1-6.2 SQL)

**Renamed modules**
- `src/aios/models/scheduled_tasks.py` → `src/aios/models/triggers.py`
- `src/aios/services/scheduled_tasks.py` → `src/aios/services/triggers.py`
- `src/aios/harness/scheduled_task_runner.py` → `src/aios/harness/trigger_runner.py`
- `src/aios/tools/schedule_task_{add,remove,update,list}.py` → `src/aios/tools/trigger_{create,remove,update,list}.py`

**Edited**
- `src/aios/db/queries/__init__.py` — all `*scheduled_task*` functions → `*trigger*`; new
  jsonb-shaped INSERT/UPDATE; `TriggerRow` NamedTuple (§3 pinned shape); claim SELECT projects
  `source_spec->>'schedule' AS schedule`; the unscoped getter keeps its intentionally-unscoped
  shape (§3); **`clone_session` INSERT fix (§6.1)**; list `ORDER BY` preserved (§3)
- `src/aios/db/listen.py` — helper rename only; channel constant VALUE unchanged
- `src/aios/harness/scheduler.py` — claim/advance branches on `source`; renamed query imports
- `src/aios/harness/tasks.py` — task name `harness.run_trigger` (D5)
- `src/aios/harness/worker.py` — comment refs
- `src/aios/tools/schedule_wake.py` — emit `one_shot` + `wake_owner`; result key
  `task_id`→`trigger_id`; description refs `trigger_*` (§8); keep NL-time parsing
- `src/aios/tools/wake_self.py` — model-facing description vocabulary only ("scheduled-task cron
  commands" → trigger/sandbox_command wording)
- `src/aios/tools/__init__.py` — registration imports
- `src/aios/models/agents.py` — `BuiltinToolType` (§6.3)
- `src/aios/models/sessions.py` — `SessionCreate.triggers` / `Session.triggers` (D2)
- `src/aios/services/sessions.py` — create-time attachment path renames
- `src/aios/api/routers/sessions.py` — `/sessions/{id}/triggers` routes per §8.2 (PUT update)
- `src/aios/cli/commands/sessions.py` — `aios sessions triggers …` per §8.2 (incl. `@covers`
  strings; drop `#636` from help text)
- `src/aios/config.py` — `triggers_per_account_max` (D7)
- `src/aios/ids.py` — `TRIGGER = "trig"`; `"sched"` retained in `_PREFIXES` as legacy (D6)
- `src/aios/errors.py`, `src/aios/sandbox/tool_broker.py`, `src/aios/db/queries/workflows.py`,
  `src/aios/services/wake.py` — docstring/comment refs only (verified: nothing functional)

**Regenerated (CI invariants — run before pushing)**
- `openapi.json` (`./scripts/regen-openapi.sh`)
- `packages/aios-sdk/aios_sdk/_generated/**` (`./scripts/regen-client.sh`; rsync --delete cleans
  the renamed module files)

**Tests**
- `tests/unit/test_scheduled_tasks_models.py` → `test_triggers_models.py` (+ union validation,
  occurrence-horizon incl. read-path acceptance of a rare cron, wholesale-update 422 on partial
  sandbox_command action, must-reject CHECK probes from §2.1)
- `tests/unit/test_schedule_wake_handler.py` — port: asserts emitted trigger is
  `one_shot`+`wake_owner`, result carries `trigger_id`
- `tests/unit/cli/test_cli_sessions.py` — subcommand renames (coverage drift guard
  `tests/unit/test_cli_coverage.py` self-enforces, no direct edit)
- `tests/e2e/test_scheduled_tasks.py` → `test_triggers.py` (cron sandbox_command e2e verbatim;
  NEW: one-shot `wake_owner` delivers user-role marker + self-deletes; NEW: migration backfill
  equivalence incl. empty-tools + custom-tool-named-schedule_task_add agent rows §6.2; NEW:
  wake_owner cap-enforcement §8; NEW: past-fire_at source-replace-on-enabled-row 422 §2.4)
- `tests/e2e/test_session_status_pending.py` — fixture renames
- NEW: clone-with-one-shot regression test (§6.1)
- **Explicitly untouched:** `tests/unit/test_wake_session_handler.py` (cross-session tool is out
  of scope — its survival is the D1 acceptance check) and
  `tests/integration/test_spec_version_triggers.py` (Postgres spec-version triggers; word
  collision only — implementers grepping "trigger" will hit it)

**Out-of-repo flags**
- eumemic-ops: grep `AIOS_SCHEDULED_TASKS_PER_ACCOUNT_MAX` before promote (D7)
- aios-console (aios-ui): reads `session.scheduled_tasks` + the scheduled-tasks routes → follow-up

---

## 11. First commit

**One atomic commit: migration + full code rename + runner dispatch + tests.** A migration-only
commit cannot be green: the old code reads dropped columns, so every e2e (which runs
`alembic upgrade head`) breaks — the rename is indivisible. This also matches the repo's
pre-commit reality (hook validates the whole worktree; stage everything). SDK/openapi regen lands
in the same commit (CI drift gates). The PR is one substantial conventional commit:
`feat(triggers): rename session_scheduled_tasks→triggers; source enum + action union (#818)`.
