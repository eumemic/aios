# Connector storage migration notes

## Per-instance cloister (#238)

Pre-#238 deployments stored connector state at `~/.aios/connectors/<name>/`
(spool databases, signal-cli config, etc.). That path is now per-instance:
`~/.aios/instances/<instance_id>/connectors/<name>/`. Concurrent dev worktrees
no longer clobber each other's spools.

For an existing `instance_id="default"` deployment (the production
single-instance case), one-time move:

```
mkdir -p ~/.aios/instances/default
mv ~/.aios/connectors ~/.aios/instances/default/connectors
```

`aios dev bootstrap` writes `AIOS_CONNECTORS_DIR=~/.aios/instances/<id>/connectors`
into the per-worktree `.env` automatically — dev instances cloister without
operator action. Operators that explicitly set `AIOS_CONNECTORS_DIR` keep their
override (escape hatch for ops that pre-bake connector state elsewhere).
