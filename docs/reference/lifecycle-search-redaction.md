# `lifecycle_search` redaction invariant

`lifecycle_search.detail_text` may render non-sensitive lifecycle payload fields, but its migration-maintained strip vocabulary must include every cumulative usage counter and every cost field in the search-view denylist. The integration test structurally aligns the migration keys, denylist, and execution sentinels, and exercises all allowlisted lifecycle event kinds.

PostgreSQL's `jsonb - text[]` operation removes **top-level keys only**. Lifecycle writers must therefore keep usage counters and cost data at the payload's top level. A nested value such as `{"usage": {"cumulative_tokens": 1}}` is not removed by this fence; introducing nested counters requires updating the view expression and its tests in the same change.
