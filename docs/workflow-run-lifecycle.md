# Stale suspended workflow runs

The workflow sweep cancel-signals a suspended run after `workflow_suspended_reap_seconds` (24 hours by default) only when all unresolved awaited agent sessions or workflow runs are terminal, archived, or missing. A live awaited child keeps its parent parked.

Cancellation uses the durable run-signal path, so the workflow step remains the only journal writer. Operators can also free an outstanding-run slot with `POST /v1/tasks/{run_id}/cancel?request_id={request_id}`.
