# Done

- Root cause: the trigger-swap e2e command used curl's default address-family
  selection. `api.github.com` can resolve IPv6, but sandbox credential-host DNAT
  is IPv4-only, so the request could bypass DNAT/secret-egress and the recorder
  saw zero requests. This affected both Limited and Unrestricted trigger legs.
- Fix: force curl IPv4 (`-4`) in the trigger swap command, ensuring the request
  traverses the IPv4 DNAT and secret-egress proxy while preserving the existing
  security behavior.
- Commit: `4d2d6661` (`test(e2e): force IPv4 for trigger secret-egress swap`)
- Rebased onto `origin/master` at `0b07495e`; not pushed.
- Verified: Python bytecode compilation of the modified e2e module succeeds.
  Docker e2e and pytest were unavailable in this environment (`pytest` and
  `python` commands are not installed); CI Docker validation remains required.
