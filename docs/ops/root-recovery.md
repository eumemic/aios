# Recovering from lost root credentials

This procedure recovers operator access when the root account's API
keys have all been lost or revoked, or when the root account has been
accidentally archived. It requires direct Postgres access — there is
no in-band recovery path, by design (a re-openable bootstrap endpoint
would be a permanent re-entry mechanism that defeats the purpose of
the one-shot model).

## Symptom

Every API call returns `401 unauthorized`. No working root key is
known. `AIOS_BOOTSTRAP_TOKEN` is set but `POST /v1/accounts/bootstrap`
returns `404` (a root row exists in the table, just with no usable
key).

## Procedure

You need shell access to the host running aios and credentials for
the aios Postgres database.

### 1. Confirm the root account exists

```sql
SELECT id, display_name, archived_at, created_at
FROM accounts
WHERE parent_account_id IS NULL;
```

You expect exactly one row. If the row is archived (`archived_at IS
NOT NULL`), continue to step 2. Otherwise skip to step 3.

### 2. Un-archive the root if it was accidentally archived

```sql
UPDATE accounts
SET archived_at = NULL
WHERE id = '<root_account_id>'
  AND parent_account_id IS NULL;
```

The `accounts_one_active_root` partial unique index blocks two active
roots from coexisting — if you somehow ended up with a stale active
root *and* a real one, you'll see a unique-violation here. In that
case, archive the stale row first and re-run the un-archive.

### 3. Generate a fresh API key and its sha256 hash

Run this on the host as the aios operator:

```bash
PLAINTEXT=$(python3 -c '
import secrets
print("aios_" + secrets.token_urlsafe(32))
')

# print the plaintext so you can capture it — this is your new root key
echo "ROOT_KEY: $PLAINTEXT"

# compute the sha256 hash (raw bytes, hex-encoded for psql)
HEX_HASH=$(python3 -c "
import hashlib, sys
print(hashlib.sha256(sys.argv[1].encode()).hexdigest())
" "$PLAINTEXT")

echo "HEX_HASH: $HEX_HASH"
```

Capture `ROOT_KEY` into your secret store *immediately* — you cannot
recover it from the database after this step.

### 4. Insert the key row

```sql
INSERT INTO account_keys (key_id, account_id, hash, label)
VALUES (
    'key_' || replace(gen_random_uuid()::text, '-', ''),  -- any unique key_id
    '<root_account_id>',
    decode('<HEX_HASH from step 3>', 'hex'),
    'manual-recovery'
);
```

The `key_id` value can be any string; the rest of the system only
identifies keys via their hash. Using a descriptive `label` like
`manual-recovery` makes the recovery audit-visible in later
`GET /v1/accounts/{id}/keys` calls.

### 5. Verify and restart

```bash
# verify the key works against the running api
curl -sS -H "Authorization: Bearer $PLAINTEXT" \
  http://localhost:8090/v1/accounts/self
```

Once the recovery key is verified, restart api + worker so any cached
auth state is cleared.

### 6. Rotate the recovery key

A manually-injected key sidesteps the normal mint flow and isn't
labeled or scoped like a production key. Once you're back in, use it
to mint a new key via `POST /v1/accounts/<root_id>/keys` and revoke
the recovery key with `DELETE /v1/accounts/keys/<key_id>`.

## Why this is deliberately painful

Root represents full operator authority over every account in the
system. A re-runnable bootstrap path or an "emergency override" env
var would mean any future compromise of that secret can re-mint root
indefinitely. The Postgres-side recipe requires database credentials,
which are themselves a high-trust artifact — the recovery surface is
exactly as protected as the database itself, and no more.


## Multi-tenancy migration (0042 → 0044)

When upgrading a populated aios deployment from a pre-multi-tenancy
build to a build that includes migrations 0042–0044, the migration
script enforces a per-row `account_id NOT NULL` constraint on every
tenant-scoped resource table. That constraint requires a root account
to exist before the backfill can succeed. The order is therefore:

1. **Bootstrap the root account first.** Before running `alembic
   upgrade head`, start the API server, set `AIOS_BOOTSTRAP_TOKEN`,
   and hit `POST /v1/accounts/bootstrap` to create the root row. Save
   the plaintext key returned in the response.
2. **Then run the migrations.** Migration 0044's backfill subquery
   resolves to the root account's `id` and stamps it onto every
   pre-existing row that doesn't have an `account_id` yet. The
   `SET NOT NULL` step then succeeds because every row has a value.
3. **Restart api + worker.** The auth dep changes shape between
   pre-multi-tenancy and post-multi-tenancy (the bearer now resolves
   to a 3-tuple including `account_id`); a restart clears any cached
   FastAPI dependency state from the old shape.

### What happens if you run migrations before bootstrapping

Migration 0044 will fail at the `SET NOT NULL` step on the first table
that contains pre-existing rows: the backfill subquery returns `NULL`
(no root account), the `UPDATE` leaves the column as-is, and `SET
NOT NULL` rejects the operation with a Postgres `23502` error. Roll
back to the previous migration, bootstrap the root, and re-run.

### Per-tenant resource recycling

Migration 0045 reshapes the `display_name` partial unique indexes on
`agents`, `environments`, `credentials`, and `session_templates` from
global to `(account_id, name)`. After this migration, two tenants can
mint a resource named `"default"` independently. There's no data
migration for this — existing rows are left in place; the partial
unique index just allows what was previously rejected.

### Fresh installs

On a fresh install (empty tables), the backfill in 0044 matches zero
rows, `SET NOT NULL` succeeds trivially, and the bootstrap step
happens after migrations in the usual order. The "bootstrap first"
ordering only matters on upgrades of populated databases.
