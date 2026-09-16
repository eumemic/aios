Rebased `gvisorgrn` onto `origin/master` (`e4679395`).

Resolved the gVisor workflow conflict using master’s runsc-capability scoping,
retaining the required CI configuration and updating its unit drift test. The
sandbox resolver remains hosts-first (`/etc/hosts`, then BusyBox `nslookup`),
with no `getent` path. Focused workflow and DNS-resolution tests pass.
