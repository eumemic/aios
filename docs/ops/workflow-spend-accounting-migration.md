# Workflow spend accounting migration

Migration 0168 repairs the historical period in which raw workflow `call_llm`
cost was stored on `wf_runs` but was not automatically projected into
`accounts.spent_microusd`. Account totals cannot prove whether a particular
workflow charge was already included: manual adjustments and deleted sessions
can produce the same aggregate value. Migration 0167 therefore creates an
explicit operator watermark and 0168 refuses to guess when one is missing.

## Procedure

1. Upgrade only through the watermark revision:

   ```console
   uv run alembic upgrade 0167
   ```

2. List accounts with retained raw-workflow cost:

   ```sql
   SELECT account_id,
          SUM(call_llm_cost_microusd)::bigint AS retained_run_cost_microusd
     FROM wf_runs
    GROUP BY account_id
   HAVING SUM(call_llm_cost_microusd) > 0
    ORDER BY account_id;
   ```

3. For every returned account, determine how much of that retained run cost is
   already represented in `accounts.spent_microusd`, then record that amount:

   ```sql
   INSERT INTO workflow_spend_accounting_watermarks (
       account_id,
       accounted_run_cost_microusd
   ) VALUES (
       'acc_example',
       0
   );
   ```

   Use `0` only after confirming no historical raw-workflow charge was manually
   added for that account. Use the exact retained amount already included for a
   full or partial prior repair. Do not infer this value from aggregate equality.
   Do not manually adjust `accounts.spent_microusd` between recording the
   watermark and completing 0168. Raw workflow writers may continue: 0168 locks
   their run meter at cutover and includes any later pre-cutover increment in
   the unaccounted delta.

4. Upgrade through 0168 or head:

   ```console
   uv run alembic upgrade head
   ```

0168 adds only `retained run cost - accounted watermark` and advances the
watermark to the retained run meter in the same transaction. It fails before
changing account spend if a required watermark is absent or exceeds retained
run cost. The post-migration database trigger advances the account meter and
watermark together for every new raw-workflow charge.
