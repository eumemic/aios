# Connection change cursors

Connection discovery cursors ar scoped to `(account_id, connector)`.
Writers hold a per-stream transaction advisory lock before allocating
`seq`, so visible sequence order matches commit order within the stream.

The durable pruning horizon survives ledger retention. A fresh cursor is the
maximum of the visible ledger high-water mark and that horizon. This keeps the
fresh to tail handoff valid even when retention has emptied the ledger.
