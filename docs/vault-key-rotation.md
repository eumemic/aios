# Vault key rotation runbook

This page covers rotating `AIOS_VAULT_KEY` after the egress CA cutover.
Runtime messages name only the environment variables and commands operators use.

## One-time egress CA cutover

1. Mint the new CA seed: `openssl rand -base64 32`.
2. Add it to the production environment as `AIOS_EGRESS_CA_KEY` before deploy. Boot fails loudly if it is missing.
3. Deploy/restart API and worker.
4. Run the single managed recycle wave:
   - Sessions with env-var credentials: bump env-var credential `updated_at` so the existing drift probe snapshot-recycles each affected session, or accept next-step TLS failure followed by recycle.
   - Runs: let the run finish or the idle reaper end exposure; a mid-run TLS failure is branchable and is not a leak.
5. Done. After this cutover, rotating `AIOS_VAULT_KEY` changes no sandbox CA and no secret placeholder.

## Rotating `AIOS_VAULT_KEY`

1. Mint the new key: `openssl rand -base64 32`.
2. Set `AIOS_VAULT_KEY` to the new value and set `AIOS_VAULT_KEY_PREVIOUS` to the old value.
3. Restart API/worker so new writes use the new key.
4. Run `aios rekey`. It decrypts encrypted rows with current-then-previous and re-encrypts with current. It does not bump `updated_at` because re-encryption is not an observable credential change.
5. Unset `AIOS_VAULT_KEY_PREVIOUS` and restart.

Secret placeholders are a pure function of `(account placeholder salt, owner, credential)`. The salt value is re-encrypted during `aios rekey` but does not change, so placeholders an agent persisted into `/workspace` continue resolving across master-key rotation.
