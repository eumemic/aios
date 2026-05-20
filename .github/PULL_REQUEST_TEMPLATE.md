## Summary

<!-- 1-3 sentences on what + why. -->

## Substrate state changes

<!--
If your PR touches ANY of the following, list the coordinated host-side / Coolify-side / dependency state changes needed alongside the merge. If none apply, leave this section as: "None — code-only change".
-->

- [ ] Required env-var contract changed (added, removed, renamed, validation tightened)
- [ ] File-mount content shape changed (e.g. `/config/config.yaml`)
- [ ] Container UID, working-dir, or volume-permission expectations changed
- [ ] Persistent-volume layout changed (new mount points, names, ownership)
- [ ] External-service contract changed (DB schema, downstream API surface, public auth shape)
- [ ] First-boot / migration grandfather hook needed for live consumers

If ANY box above is checked, paste the post-merge ops checklist:

```
1. ...
2. ...
```

## Test plan

<!-- How was this verified? Real traffic? CI? Smoke? Unit + integration? -->

## Risk / rollback

<!--
Optional. If non-trivial: how to roll back if this breaks prod. For substrate changes, this section is mandatory.
-->
